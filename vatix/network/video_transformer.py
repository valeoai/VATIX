import torch
from torch import nn

from einops import rearrange

from vatix.network.transformer_block import RMSNorm, Attention, FeedForward, TimestepEmbedder
from vatix.network.trajectory_embedder import PerFrameTrajectoryEmbedder

# Block whose output feeds the trajectory aux head (clamped to the last block for shallow models).
AUX_FEAT_LAYER = 8


def modulate(x, gamma):
    return x * (1 + gamma)


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout=0.):
        super().__init__()

        self.spatial_attn = Attention(dim, heads, dropout=dropout)
        self.ln_spatial = RMSNorm(dim, linear=True, bias=False, eps=1e-5)

        self.temporal_attn = Attention(dim, heads, dropout=dropout)
        self.ln_temporal = RMSNorm(dim, linear=True, bias=False, eps=1e-5)

        self.ff = FeedForward(dim, mlp_dim, dropout=dropout)
        self.ln_mlp = RMSNorm(dim, linear=True, bias=False, eps=1e-5)

    def forward(self, x, ada_cond, t, s, temporal_mask=None):
        """
        Args:
            x: main hidden states [B, N, D]
            text_cond: optional key/values for cross-attention
            ada_cond: conditioning tensor [B, D] 
        """ 

        (
            gamma_spatial, alpha_spatial,
            gamma_temporal, alpha_temporal,
            gamma_mlp, alpha_mlp,
        ) = ada_cond

        b, n, d = x.shape
        n_video = t * s
        n_reg = n - n_video

        if n_reg < 0:
            raise ValueError(f"Invalid token layout: got N={n}, expected at least T*S={n_video}")

        x_reg = x[:, :n_reg] if n_reg > 0 else None
        x_video = x[:, n_reg:]

        # --- Spatial self-attention (within frame) ---
        x_spatial = modulate(self.ln_spatial(x_video), gamma_spatial[:, n_reg:])
        x_spatial = rearrange(x_spatial, 'b (t s) d -> (b t) s d', t=t, s=s)
        x_spatial = self.spatial_attn(x_spatial)
        x_spatial = rearrange(x_spatial, '(b t) s d -> b (t s) d', b=b, t=t, s=s)
        x_video = x_video + alpha_spatial[:, n_reg:] * x_spatial

        # --- Temporal self-attention (across time, per spatial location) ---
        x_temporal = modulate(self.ln_temporal(x_video), gamma_temporal[:, n_reg:])
        x_temporal = rearrange(x_temporal, 'b (t s) d -> (b s) t d', t=t, s=s)
        x_temporal = self.temporal_attn(x_temporal, mask=temporal_mask)
        x_temporal = rearrange(x_temporal, '(b s) t d -> b (t s) d', b=b, t=t, s=s)
        x_video = x_video + alpha_temporal[:, n_reg:] * x_temporal

        x = torch.cat([x_reg, x_video], dim=1) if n_reg > 0 else x_video

        # --- Feed-forward with AdaLN modulation ---
        x = x + alpha_mlp * self.ff(modulate(self.ln_mlp(x), gamma_mlp))

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([Block(dim, heads, mlp_dim, dropout=dropout) for _ in range(depth)])
        self.feat_layer = None  # index of the block whose output is also returned (trajectory aux head)

    def forward(self, x, ada_cond, t, s, temporal_mask=None):
        feat = None
        for i, block in enumerate(self.layers):
            x = block(x, ada_cond=ada_cond, t=t, s=s, temporal_mask=temporal_mask)
            if i == self.feat_layer:
                feat = x
        return x, feat


class Transformer(nn.Module):
    """ DiT-like transformer with adaRMSNorm with zero initializations """
    def __init__(self, input_size=(16, 8, 16, 16), hidden_dim=768,
                 depth=12, heads=16, mlp_dim=3072, dropout=0.,
                 register=1, proj=1, is_causal=False,
                 use_trajectory_cond=False, trajectory_length=25,
                 use_trajectory_aux_head=False,
                 trajectory_fuse_mode="chunk_sum",
                 traj_aux_head_norm=False):
        super().__init__()

        self.input_size = input_size                                    # Number of tokens as input
        self.c, self.t, self.h, self.w = self.input_size                #
        self.hidden_dim = hidden_dim                                    # Hidden dimension of the transformer
        self.proj = proj                                                # Projection
        self.register = register                                        # add register token
        self.is_causal = is_causal                                      # use temporal causal mask in the transformer
        self.use_trajectory_cond = use_trajectory_cond                  # condition generation on an ego-trajectory
        self.trajectory_length = trajectory_length                      # number of (x, y) waypoints per clip
        self.trajectory_fuse_mode = trajectory_fuse_mode                # "chunk_sum" or "chunk_sum_rms"

        # Separate temporal and spatial (learned) positional embeddings
        self.temporal_pos = nn.Embedding(self.t, hidden_dim)
        self.spatial_pos = nn.Embedding((self.h * self.w)//(proj**2), hidden_dim)

        # number of spatial tokens
        self.num_spatial = (self.h * self.w) // (proj ** 2)

        self.time_embed = TimestepEmbedder(in_dim=hidden_dim, out_dim=hidden_dim*6)

        # project the input to a smaller space
        self.in_proj = nn.Conv2d(self.c, hidden_dim, kernel_size=self.proj, stride=self.proj)
        self.out_proj = nn.Linear(hidden_dim, self.c*proj**2)

        # Transformer Archi
        self.transformer = TransformerEncoder(dim=hidden_dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout)
        self.last_norm = RMSNorm(dim=hidden_dim, linear=True, bias=True)

        if self.register > 0:
            self.reg_tokens = nn.Embedding(self.register, hidden_dim)

        self.trajectory_aux_head = None
        if self.use_trajectory_cond:
            if trajectory_fuse_mode not in ("chunk_sum", "chunk_sum_rms"):
                raise ValueError(f"trajectory_fuse_mode must be 'chunk_sum' or 'chunk_sum_rms', got {trajectory_fuse_mode!r}")
            # Six AdaLN modulation chunks per latent frame, summed onto the timestep chunks.
            self.trajectory_embed = PerFrameTrajectoryEmbedder(
                trajectory_length=trajectory_length,
                hidden_dim=hidden_dim,
                latent_length=self.t,
                out_dim=hidden_dim * 6,
            )

            # Training-only aux head: predicts each frame's waypoints from a mid-depth block, so
            # the conditioning pathway gets gradient at high noise. Built only when used (DDP).
            if use_trajectory_aux_head:
                self.transformer.feat_layer = min(AUX_FEAT_LAYER, depth - 1)
                self.trajectory_aux_head = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, self.trajectory_embed.group_size * 2),  # (x, y) per waypoint
                )
                # Own attribute, not element 0 of the Sequential: inserting there would
                # renumber trajectory_aux_head.0/.2 and break old checkpoints.
                if traj_aux_head_norm:
                    self.trajectory_aux_norm = RMSNorm(dim=hidden_dim, linear=True, bias=True)

        self.initialize_weights()  # Init weight

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Init embedding
        nn.init.normal_(self.temporal_pos.weight, std=0.02)
        nn.init.normal_(self.spatial_pos.weight, std=0.02)

        # Init proj layer
        if self.proj > 1:
            nn.init.xavier_uniform_(self.in_proj.weight)
            # nn.init.xavier_uniform_(self.out_proj.weight)

        # Init register
        if self.register > 0:
            nn.init.normal_(self.reg_tokens.weight, std=0.02)

        # AdaLN-Zero style init: start modulation close to identity / zero residual scaling
        nn.init.constant_(self.time_embed.mlp[-1].weight, 0)
        nn.init.constant_(self.time_embed.mlp[-1].bias, 0)
        if self.use_trajectory_cond:
            # AdaLN-Zero: at step 0 the trajectory contributes nothing, so fine-tuning an
            # unconditional checkpoint is safe.
            nn.init.constant_(self.trajectory_embed.mlp[-1].weight, 0)
            nn.init.constant_(self.trajectory_embed.mlp[-1].bias, 0)
        if hasattr(self, "trajectory_aux_norm"):
            # Opt-in only: zeroing the aux head's output changes its iter-0 prediction.
            nn.init.constant_(self.trajectory_aux_head[-1].weight, 0)
            nn.init.constant_(self.trajectory_aux_head[-1].bias, 0)


    def forward(self, x, ada_cond, return_feat=False, trajectory_cond=None, trajectory_keep_mask=None):
        b, c, t, h, w = x.size()
        S = (h * w) // (self.proj ** 2) # spatial token per frames

        if self.is_causal:
            temporal_mask = torch.full((t, t), float("-inf"), device=x.device)
            temporal_mask = torch.triu(temporal_mask, diagonal=1)
        else:
            temporal_mask = None
        
        x = rearrange(x, 'b c t h w -> (b t) c h w', b=b, t=t, c=c, h=h, w=w).contiguous()
        x = self.in_proj(x)
        _, c, h, w = x.shape
        x = rearrange(x, '(b t) c h w -> b (t h w) c', b=b, t=t, c=c, h=h, w=w).contiguous()

        # Positional embeddings
        t_pos = torch.arange(t, device=x.device).repeat_interleave(h * w)
        s_pos = torch.arange(h * w, device=x.device).repeat(t)
        pos = self.temporal_pos(t_pos) + self.spatial_pos(s_pos)
        x = x + pos

        # timestep embeddings
        if ada_cond.dim() == 1:
            ada_cond = ada_cond[:, None]
        if ada_cond.shape[1] == 1 and t > 1:
            ada_cond = ada_cond.expand(-1, t)
        
        t_emb = self.time_embed(ada_cond).chunk(6, dim=-1)

        if self.use_trajectory_cond:
            if trajectory_cond is None or trajectory_cond.shape[0] != b:
                raise ValueError(
                    f"trajectory conditioning is enabled: expected trajectory_cond with batch size {b}, "
                    f"got {None if trajectory_cond is None else tuple(trajectory_cond.shape)}"
                )
            # (B, T, 6D): one modulation set per latent frame, summed onto the timestep chunks.
            traj_emb = self.trajectory_embed(trajectory_cond).to(dtype=t_emb[0].dtype)
            if trajectory_keep_mask is not None:
                # Dropped samples add zero: the unconditional pathway used for classifier-free guidance.
                traj_emb = traj_emb * trajectory_keep_mask.to(traj_emb).view(b, 1, 1)
            traj_chunks = traj_emb.chunk(6, dim=-1)
            if self.trajectory_fuse_mode == "chunk_sum_rms":
                # Scale each traj chunk by the detached RMS of its timestep chunk (fine-tuning only).
                fused = []
                for time_chunk, traj_chunk in zip(t_emb, traj_chunks):
                    scale = time_chunk.float().pow(2).mean(dim=-1, keepdim=True).sqrt().detach()
                    # Cap the traj/time RMS ratio at 1; the clamp also avoids a NaN grad on zero chunks.
                    scale = scale / traj_chunk.float().pow(2).mean(dim=-1, keepdim=True).clamp(min=1.0).sqrt()
                    fused.append((time_chunk.float() + traj_chunk.float() * scale).to(time_chunk.dtype))
                t_emb = fused
            else:
                t_emb = [time_chunk + traj_chunk for time_chunk, traj_chunk in zip(t_emb, traj_chunks)]
        elif trajectory_cond is not None or trajectory_keep_mask is not None:
            raise ValueError("trajectory_cond/trajectory_keep_mask given but trajectory conditioning is disabled for this model")

        t_emb_expanded = [e.repeat_interleave(S, dim=1) for e in t_emb]  # (B, N, D)

        
        if self.register > 0:
            reg = torch.arange(0, self.register, dtype=torch.long, device=x.device)
            x = torch.cat([self.reg_tokens(reg).expand(b, self.register, self.hidden_dim), x], dim=1)
            t_emb_expanded = [torch.cat([torch.zeros(b, self.register, self.hidden_dim, dtype=x.dtype, device=x.device), e], dim=1)
                                for e in t_emb_expanded]

        x, feat = self.transformer(x=x, ada_cond=t_emb_expanded, t=t, s=S, temporal_mask=temporal_mask)

        # drop the register(s)
        x = x[:, self.register:].contiguous()

        x = self.last_norm(x)
        x = self.out_proj(x)
        x = rearrange(x, 'b (t h w) (c s1 s2) -> b c t (h s1) (w s2)', s1=self.proj, s2=self.proj, b=b, c=self.c, h=h, w=w).contiguous()

        if return_feat:
            traj_aux_pred = None
            if self.trajectory_aux_head is not None:
                # Drop the register, mean-pool over space, predict the non-origin frames' waypoints.
                feat = feat[:, self.register:].reshape(b, t, S, -1).mean(dim=2).float()  # (B, T, D)
                if hasattr(self, "trajectory_aux_norm"):
                    feat = self.trajectory_aux_norm(feat)
                traj_aux_pred = self.trajectory_aux_head(feat[:, 1:])                    # (B, T-1, G*2)
            return x, traj_aux_pred

        return x

