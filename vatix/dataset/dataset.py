import os
import random
import json
import importlib
import bisect
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
try:
    from torchcodec.decoders import VideoDecoder
except:
    VideoDecoder = None
import imageio.v2 as imageio

import torch.nn.functional as F

from vatix.trajectory_utils import smooth_trajectory as _smooth_trajectory

def make_cam_path(front_path, cam):
    cam = cam.upper()
    if cam == "FRONT" and os.path.exists(front_path):
        return front_path

    candidates = []

    # Generic placeholder format.
    candidates.append(front_path.replace("CAM_FOLDER", f"{cam}_FOLDER").replace("CAM_", f"{cam}_"))
    # Natix FRONT naming.
    candidates.append(front_path.replace("FRONT_FOLDER", f"{cam}_FOLDER").replace("FRONT_", f"{cam}_"))

    if cam == "LEFT":
        candidates.append(front_path.replace("FRONT_FOLDER", "LEFT_REPEATER_FOLDER").replace("FRONT_", "LEFT_REPEATER_"))
    elif cam == "RIGHT":
        candidates.append(front_path.replace("FRONT_FOLDER", "RIGHT_REPEATER_FOLDER").replace("FRONT_", "RIGHT_REPEATER_"))

    # Keep only paths that actually changed from FRONT path and de-duplicate.
    seen = set()
    deduped = []
    for p in candidates:
        if p == front_path:
            continue
        if p not in seen:
            deduped.append(p)
            seen.add(p)

    for p in deduped:
        if os.path.exists(p):
            return p

    raise FileNotFoundError(f"No valid camera path for {cam}:\n{front_path}\nTried: {deduped}")


def make_cam_latent_path(front_path, cam):
    cam = cam.upper()
    if cam == "FRONT" and os.path.exists(front_path):
        return front_path

    cam_aliases = {
        "FRONT": "CAM_FRONT",
        "LEFT": "CAM_FRONT_LEFT",
        "RIGHT": "CAM_FRONT_RIGHT",
        "REAR": "CAM_REAR",
        "REAR_LEFT": "CAM_REAR_LEFT",
        "REAR_RIGHT": "CAM_REAR_RIGHT",
    }

    candidates = []

    # Generic placeholders used in some list files.
    candidates.append(front_path.replace("CAM_FOLDER", f"{cam}_FOLDER").replace("CAM_", f"{cam}_"))

    # Natix-style FRONT naming.
    candidates.append(front_path.replace("FRONT_FOLDER", f"{cam}_FOLDER").replace("FRONT_", f"{cam}_"))

    # NuScenes-style channels.
    if cam in cam_aliases:
        candidates.append(front_path.replace("CAM_FRONT", cam_aliases[cam]))

    # LEFT/RIGHT repeater fallback.
    if cam == "LEFT":
        candidates.append(front_path.replace("FRONT_FOLDER", "LEFT_REPEATER_FOLDER").replace("FRONT_", "LEFT_REPEATER_"))
        candidates.append(front_path.replace("CAM_FRONT", "CAM_FRONT_LEFT"))
    elif cam == "RIGHT":
        candidates.append(front_path.replace("FRONT_FOLDER", "RIGHT_REPEATER_FOLDER").replace("FRONT_", "RIGHT_REPEATER_"))
        candidates.append(front_path.replace("CAM_FRONT", "CAM_FRONT_RIGHT"))

    # Preserve order while removing duplicates; ignore unchanged FRONT path for non-FRONT cams.
    seen = set()
    deduped = []
    for p in candidates:
        if p == front_path:
            continue
        if p not in seen:
            deduped.append(p)
            seen.add(p)

    for p in deduped:
        if os.path.exists(p):
            return p

    raise FileNotFoundError(
        f"No valid latent path for camera {cam}. Base path: {front_path}.\n"
        f"Tried: {deduped}"
    )


class CodeDataset(Dataset):
    def __init__(self, file_list, nb_latents_frame=7, samples_per_file=None, cameras=None):
        """
        Args:
            folder_path (str): Path to the folder containing .pth files.
            samples_per_file (int|None): If set, expose this many deterministic
                samples per file. Useful when each file stores multiple clips.
        """
        self.file_list = file_list
        self.nb_latents_frame = nb_latents_frame
        self.samples_per_file = samples_per_file
        self.cameras = cameras if cameras is not None else ["FRONT"]
        
    def __len__(self):
        if self.samples_per_file is not None:
            return len(self.file_list) * int(self.samples_per_file)
        return len(self.file_list)

    def load_latents(self, file_path):
        data = torch.load(file_path) # Ensure the file can be loaded as a PyTorch tensor
        current_t = data["latents"].shape[2]
        target_t = self.nb_latents_frame
        if current_t > target_t:
            data["latents"] = data["latents"][:, :, :target_t, ...]
        elif current_t < target_t:
            pad_count = target_t - current_t
            last_slice = data["latents"][:, :, -1:, ...]
            data["latents"] = torch.cat([data["latents"], last_slice.repeat(1, 1, pad_count, 1, 1)], dim=2)
        return data
    
    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the file to load.

        Returns:
            dict: A dictionary containing "latent" and "y".
        """
        max_attempts = min(len(self.file_list), 32)

        for attempt in range(max_attempts):
            if self.samples_per_file is not None:
                probe_idx = (idx + attempt) % len(self)
                file_idx = probe_idx // int(self.samples_per_file)
                sample_idx = probe_idx % int(self.samples_per_file)
            else:
                file_idx = (idx + attempt) % len(self)
                sample_idx = None

            file_path = self.file_list[file_idx]

            try:
                cam_paths = [make_cam_latent_path(file_path, cam) for cam in self.cameras]
                cam_data = [self.load_latents(path) for path in cam_paths]
                base_data = cam_data[0]

                if sample_idx is None:
                    sample_idx = random.randint(0, base_data["latents"].shape[0] - 1)
                else:
                    # Deterministic selection with wraparound if a file has fewer clips.
                    sample_idx = sample_idx % base_data["latents"].shape[0]

                latents_per_cam = []
                for d in cam_data:
                    local_idx = sample_idx % d["latents"].shape[0]
                    latents_per_cam.append(d["latents"][local_idx].contiguous().clone())

                # Stack cameras in width dimension: (C, T, H, W * nb_cam)
                out = {
                    "latents": torch.cat(latents_per_cam, dim=-1).contiguous().clone(),
                }

                return out
            except Exception as e:
                print(f"CodeDataset skip file {file_path}: {e}")

        raise RuntimeError(
            f"CodeDataset failed to produce a valid sample after {max_attempts} attempts. "
            f"Last probed index: {idx}"
        )


class VideoDataset(Dataset):
    def __init__(self, video_paths, num_frames=8, transform=None, sample_stride=4, shuffle=True, cameras=None):
        self.video_paths = np.array(video_paths)
        self.num_frames = num_frames
        self.transform = transform
        self.sample_stride = sample_stride
        self.shuffle = shuffle

        self.cameras = cameras if cameras is not None else ["FRONT"]
    
    def __len__(self):
        return len(self.video_paths)

    def _load_video(self, path, start, needed=None):
        # torchcodec need torch 2.10 and torchcodec 0.10
        decoder = VideoDecoder(path, transforms=[self.transform], seek_mode="approximate")

        total_frames = decoder.metadata.num_frames

        if total_frames <= 0:
            raise RuntimeError(f"Video has no frames: {path}")

        needed = needed if needed is not None else self.num_frames * self.sample_stride
        actual_start = min(start, total_frames - needed) if total_frames > needed else 0
        frames = decoder.get_frames_in_range(actual_start, actual_start + needed, self.sample_stride).data

        if len(frames) == 0:
            raise RuntimeError(f"Decoder returned no frames: {path}")

        if torch.is_tensor(frames):
            if frames.shape[0] < self.num_frames:
                pad_count = self.num_frames - frames.shape[0]
                pad = frames[-1:].repeat(pad_count, 1, 1, 1)
                frames = torch.cat([frames, pad], dim=0)
            else:
                frames = frames[:self.num_frames]
        else:
            if len(frames) < self.num_frames:
                padding = [frames[-1]] * (self.num_frames - len(frames))
                frames.extend(padding)
            else:
                frames = frames[:self.num_frames]

        del decoder
        return frames

    def __getitem__(self, idx):
        needed = self.num_frames * self.sample_stride
        max_attempts = min(len(self.video_paths), 32)
        last_error = None

        for attempt in range(max_attempts):
            if self.shuffle:
                probe_idx = random.randint(0, len(self.video_paths) - 1)
            else:
                probe_idx = (idx + attempt) % len(self.video_paths)

            init_path = self.video_paths[probe_idx]
            cam_tensors = []

            # Use a random start offset (approximate across all cams) for 1min video at 36fps, we have >2160 frame
            start = random.randint(0, 2000) if self.shuffle else 0

            try:
                for cam in self.cameras:
                    path = make_cam_path(init_path, cam)
                    tensor = self._load_video(path, start, needed)
                    cam_tensors.append(tensor.permute(1, 0, 2, 3))

                return {"images": torch.cat(cam_tensors, dim=-1)}  # (C, T, H, W*nb_cams)
            except Exception as e:
                last_error = e
                continue

        raise RuntimeError(
            f"VideoDataset failed to decode a valid sample after {max_attempts} attempts. "
            f"Last probed index: {idx}"
        ) from last_error


class VideoDatasetV2(VideoDataset):
    """Video dataset variant that avoids torchcodec and decodes with imageio/ffmpeg."""

    def _load_video(self, path, start, needed=None):
        reader = imageio.get_reader(path, format="ffmpeg")
        try:
            total_frames = reader.count_frames()
            if total_frames <= 0:
                raise RuntimeError(f"Video has no frames: {path}")

            needed = needed if needed is not None else self.num_frames * self.sample_stride
            actual_start = min(start, total_frames - needed) if total_frames > needed else 0
            frame_indices = list(range(actual_start, actual_start + needed, self.sample_stride))[:self.num_frames]

            frames = []
            for frame_idx in frame_indices:
                if frame_idx >= total_frames:
                    break
                frames.append(reader.get_data(frame_idx))

            if len(frames) == 0:
                raise RuntimeError(f"Decoder returned no frames: {path}")

            while len(frames) < self.num_frames:
                frames.append(frames[-1])

            frames_np = np.stack(frames[:self.num_frames], axis=0)  # [T,H,W,C], uint8
            frames_t = torch.from_numpy(frames_np).permute(0, 3, 1, 2).contiguous()  # [T,C,H,W]

            if self.transform is not None:
                frames_t = self.transform(frames_t)

            return frames_t
        finally:
            reader.close()


class RawVideoDataset(VideoDatasetV2):
    """Raw MP4 dataset for plain video paths without camera path remapping."""

    def __init__(self, video_paths, num_frames=8, transform=None, sample_stride=1, shuffle=True):
        super().__init__(
            video_paths=video_paths,
            num_frames=num_frames,
            transform=transform,
            sample_stride=sample_stride,
            shuffle=shuffle,
            cameras=["FRONT"],
        )

    def __getitem__(self, idx):
        path = self.video_paths[idx]
        needed = self.num_frames * self.sample_stride

        # Random start for train, deterministic start for eval.
        start = random.randint(0, 2000) if self.shuffle else 0
        tensor = self._load_video(path, start, needed)

        # Return trainer-compatible shape: (C, T, H, W)
        return {"images": tensor.permute(1, 0, 2, 3)}


class MP4WithTrajectoryDataset(RawVideoDataset):
    """Raw MP4 dataset that also returns the clip's ego-trajectory.

    Each `clip.mp4` must have a sibling `clip.npy` holding a `(trajectory_length, 2)` float32
    array of cumulative-meter waypoints in the camera frame (x=right, y=forward), relative to
    the clip origin so waypoint 0 is (0, 0).

    Output format:
        {"images": Tensor[C, T, H, W], "trajectory": Tensor[trajectory_length, 2]}
    """

    def __init__(self, video_paths, num_frames=8, transform=None, sample_stride=1, trajectory_length=25,
                 smooth_trajectory=False, horizontal_flip_aug=False):
        """
        Args:
            smooth_trajectory: SavGol-smooth the waypoints (train and val alike).
            horizontal_flip_aug: 50% mirror the clip and negate the trajectory's x (train only).
        """
        # shuffle=False forces a deterministic start at frame 0: the trajectory describes the
        # clip from its first frame, so a random start would desynchronise the two.
        super().__init__(
            video_paths=video_paths,
            num_frames=num_frames,
            transform=transform,
            sample_stride=sample_stride,
            shuffle=False,
        )
        self.trajectory_length = int(trajectory_length)
        self.smooth_trajectory = bool(smooth_trajectory)
        self.horizontal_flip_aug = bool(horizontal_flip_aug)

    @staticmethod
    def trajectory_path_for(video_path):
        """Return the sibling `.npy` trajectory path for a video path."""
        return os.path.splitext(video_path)[0] + ".npy"

    def _load_trajectory(self, video_path):
        traj_path = self.trajectory_path_for(video_path)
        if not os.path.exists(traj_path):
            raise FileNotFoundError(
                f"No trajectory found for {video_path}: expected a sibling file {traj_path} "
                f"holding a ({self.trajectory_length}, 2) float array of waypoints."
            )

        traj = np.load(traj_path).astype(np.float32, copy=False)
        if traj.shape != (self.trajectory_length, 2):
            raise ValueError(
                f"Expected trajectory of shape ({self.trajectory_length}, 2) in {traj_path}, got {traj.shape}"
            )
        if self.smooth_trajectory:
            traj = _smooth_trajectory(traj)
        return torch.from_numpy(traj)

    def __getitem__(self, idx):
        path = self.video_paths[idx]
        sample = super().__getitem__(idx)
        trajectory = self._load_trajectory(path)

        # Mirror augmentation: flip W of (C, T, H, W) and negate the lateral coordinate.
        if self.horizontal_flip_aug and random.random() < 0.5:
            sample["images"] = sample["images"].flip(-1)
            trajectory[:, 0] = -trajectory[:, 0]

        sample["trajectory"] = trajectory
        return sample


class NuScenesDataset(Dataset):
    """NuScenes image-sequence dataset that returns trainer-compatible video tensors.

    Output format:
        {"images": Tensor[C, T, H, W * nb_cam]} with uint8 values in [0, 255].
    """

    CAM_ALIASES = {
        "FRONT": "CAM_FRONT",
        "LEFT": "CAM_FRONT_LEFT",
        "RIGHT": "CAM_FRONT_RIGHT",
        "BACK": "CAM_BACK",
        "BACK_LEFT": "CAM_BACK_LEFT",
        "BACK_RIGHT": "CAM_BACK_RIGHT",
    }

    def __init__(
        self,
        data_root,
        num_frames=8,
        transform=None,
        sample_stride=1,
        split="train",
        version="v1.0-trainval",
        cameras=None,
        target_fps=10.0,
        max_time_error_us=None,
        single_clip_from_start=True,
    ):
        self.data_root = data_root
        self.num_frames = int(num_frames)
        self.transform = transform
        self.sample_stride = int(sample_stride)
        self.split = split
        self.version = version
        self.target_fps = float(target_fps)
        self.step_us = int(round(1_000_000.0 / max(self.target_fps, 1e-6)))
        self.max_time_error_us = int(max_time_error_us) if max_time_error_us is not None else max(self.step_us, 60_000)
        self.single_clip_from_start = bool(single_clip_from_start)
        self.cameras = self._resolve_cameras(cameras)
        self.windows = self._build_windows()

    def _resolve_cameras(self, cameras):
        if cameras is None:
            cameras = ["FRONT"]

        resolved = []
        for cam in cameras:
            cam_name = self.CAM_ALIASES.get(cam.upper(), cam)
            resolved.append(cam_name)
        return resolved

    def _load_table(self, table_name):
        table_path = os.path.join(self.data_root, self.version, f"{table_name}.json")
        with open(table_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _select_scene_names(self, all_scene_names):
        # Try official split definitions first when nuScenes devkit is available.
        split_key = self.split.lower()
        try:
            splits_mod = importlib.import_module("nuscenes.utils.splits")
            official = splits_mod.create_splits_scenes(verbose=False)
            if split_key in official:
                return set(official[split_key])
        except Exception:
            pass

        if split_key == "all":
            return set(all_scene_names)

        # Deterministic fallback split for environments without devkit.
        ordered = sorted(all_scene_names)
        cut = int(0.9 * len(ordered))
        if split_key == "train":
            return set(ordered[:cut])
        if split_key == "val":
            return set(ordered[cut:])
        return set(ordered)

    def _build_windows(self):
        scenes = self._load_table("scene")
        samples = self._load_table("sample")
        sample_data = self._load_table("sample_data")
        calibrated_sensors = self._load_table("calibrated_sensor")
        sensors = self._load_table("sensor")

        sample_by_token = {row["token"]: row for row in samples}
        scene_by_token = {row["token"]: row for row in scenes}

        calibrated_by_token = {row["token"]: row for row in calibrated_sensors}
        sensor_by_token = {row["token"]: row for row in sensors}

        def resolve_channel(sd_row):
            # Primary path: direct channel in sample_data.
            channel = sd_row.get("channel")
            if channel:
                return channel

            # Fallback path: calibrated_sensor -> sensor -> channel.
            calib_token = sd_row.get("calibrated_sensor_token")
            if not calib_token:
                return None
            calib = calibrated_by_token.get(calib_token)
            if calib is None:
                return None
            sensor = sensor_by_token.get(calib.get("sensor_token"))
            if sensor is None:
                return None
            return sensor.get("channel")

        # Build per-scene and per-camera timelines from raw sample_data.
        timelines = {}  # key: (scene_token, channel), value: list[(timestamp, img_path)]
        for sd in sample_data:
            if sd.get("fileformat", "").lower() != "jpg":
                continue

            sample_token = sd.get("sample_token")
            channel = resolve_channel(sd)
            filename = sd.get("filename")
            if not sample_token or not channel or not filename:
                continue

            if channel not in self.cameras:
                continue

            sample_row = sample_by_token.get(sample_token)
            if sample_row is None:
                continue

            scene_token = sample_row.get("scene_token")
            if not scene_token:
                continue

            timestamp = sd.get("timestamp", sample_row.get("timestamp", 0))
            if not timestamp:
                continue

            img_path = os.path.join(self.data_root, filename)
            timelines.setdefault((scene_token, channel), []).append((int(timestamp), img_path))

        all_scene_names = [scene["name"] for scene in scenes]
        selected_scene_names = self._select_scene_names(all_scene_names)
        selected_scene_tokens = {
            token for token, row in scene_by_token.items()
            if row.get("name") in selected_scene_names
        }

        required_len = (self.num_frames - 1) * self.sample_stride + 1
        windows = []

        for scene_token in selected_scene_tokens:
            anchor = timelines.get((scene_token, self.cameras[0]), [])
            if len(anchor) < required_len:
                continue

            per_cam_ts = {}
            per_cam_paths = {}
            missing_any_camera = False
            for cam in self.cameras:
                frames = sorted(timelines.get((scene_token, cam), []), key=lambda x: x[0])
                if len(frames) < required_len:
                    missing_any_camera = True
                    break
                per_cam_ts[cam] = [t for t, _ in frames]
                per_cam_paths[cam] = [p for _, p in frames]
            if missing_any_camera:
                continue

            anchor_ts = per_cam_ts[self.cameras[0]]
            # Do not start so late that a full window cannot be formed.
            max_start = len(anchor_ts) - required_len + 1
            if max_start <= 0:
                continue

            seen_clip_keys = set()
            start_indices = [0] if self.single_clip_from_start else range(max_start)
            for start_idx in start_indices:
                start_t = anchor_ts[start_idx]
                cam_clip = {cam: [] for cam in self.cameras}
                valid_clip = True

                for offset in range(self.num_frames):
                    target_t = start_t + offset * self.step_us * self.sample_stride
                    for cam in self.cameras:
                        ts_list = per_cam_ts[cam]
                        paths = per_cam_paths[cam]

                        pos = bisect.bisect_left(ts_list, target_t)
                        candidates = []
                        if pos < len(ts_list):
                            candidates.append(pos)
                        if pos > 0:
                            candidates.append(pos - 1)
                        if not candidates:
                            valid_clip = False
                            break

                        best_i = min(candidates, key=lambda i: abs(ts_list[i] - target_t))
                        if abs(ts_list[best_i] - target_t) > self.max_time_error_us:
                            valid_clip = False
                            break

                        frame_path = paths[best_i]
                        if not os.path.exists(frame_path):
                            valid_clip = False
                            break

                        cam_clip[cam].append(frame_path)

                    if not valid_clip:
                        break

                if valid_clip:
                    # Deduplicate identical clips produced by nearest timestamp matching.
                    clip_key = tuple(
                        frame_path
                        for cam in self.cameras
                        for frame_path in cam_clip[cam]
                    )
                    if clip_key in seen_clip_keys:
                        continue
                    seen_clip_keys.add(clip_key)
                    windows.append(cam_clip)
                    if self.single_clip_from_start:
                        break

        return windows

    def __len__(self):
        return len(self.windows)

    def _read_frame(self, path):
        with Image.open(path) as img:
            rgb = img.convert("RGB")
            arr = np.array(rgb, dtype=np.uint8)
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()

    def __getitem__(self, idx):
        cam_clip = self.windows[idx]
        cam_tensors = []

        for cam in self.cameras:
            frames = []
            for frame_path in cam_clip[cam]:
                frames.append(self._read_frame(frame_path))

            frames_t = torch.stack(frames, dim=0)  # [T,C,H,W]
            if self.transform is not None:
                frames_t = self.transform(frames_t)

            cam_tensors.append(frames_t.permute(1, 0, 2, 3))  # [C,T,H,W]

        return {"images": torch.cat(cam_tensors, dim=-1)}
