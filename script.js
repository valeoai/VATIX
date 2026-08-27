const header = document.querySelector('.site-header');
const backTop = document.getElementById('back-top');
const navToggle = document.querySelector('.nav-toggle');
const themeToggle = document.getElementById('theme-toggle');
const navLinks = document.getElementById('nav-links');
const counterTargets = document.querySelectorAll('[data-counter]');
const sceneButtonsRoot = document.getElementById('scene-buttons');
const compareToggle = document.getElementById('compare-toggle');
const compareSourceVideo = document.getElementById('compare-source-video');
const compareCanvases = Array.from(document.querySelectorAll('.compare-canvas'));
const compareWindows = Array.from(document.querySelectorAll('.comparison-window'));
const reduceMotionQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
const darkSchemeQuery = window.matchMedia('(prefers-color-scheme: dark)');
let compareRenderRaf = null;

function setTheme(theme) {
  document.body.setAttribute('data-theme', theme);
  if (themeToggle) {
    const isDark = theme === 'dark';
    themeToggle.setAttribute('aria-checked', String(isDark));
    themeToggle.setAttribute('aria-label', isDark ? 'Switch to light mode' : 'Switch to dark mode');
    themeToggle.setAttribute('title', isDark ? 'Switch to light mode' : 'Switch to dark mode');
  }
}

const savedTheme = localStorage.getItem('vatix-theme');
if (savedTheme === 'dark' || savedTheme === 'light') {
  setTheme(savedTheme);
} else {
  setTheme('dark');
}

if (themeToggle) {
  themeToggle.addEventListener('click', () => {
    const nextTheme = document.body.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
    setTheme(nextTheme);
    localStorage.setItem('vatix-theme', nextTheme);
  });
}

function onScroll() {
  const y = window.scrollY;
  if (header) {
    header.classList.toggle('scrolled', y > 10);
  }
  if (backTop) {
    backTop.classList.toggle('show', y > 600);
  }
}

window.addEventListener('scroll', onScroll, { passive: true });
onScroll();

if (backTop) {
  backTop.addEventListener('click', () => {
    window.scrollTo({ top: 0, behavior: reduceMotionQuery.matches ? 'auto' : 'smooth' });
  });
}

if (navToggle && navLinks) {
  const closeNav = () => {
    navLinks.classList.remove('open');
    navToggle.setAttribute('aria-expanded', 'false');
  };

  navToggle.addEventListener('click', () => {
    const isOpen = navLinks.classList.toggle('open');
    navToggle.setAttribute('aria-expanded', String(isOpen));
  });

  navLinks.querySelectorAll('a').forEach((link) => {
    link.addEventListener('click', () => {
      closeNav();
    });
  });

  document.addEventListener('click', (event) => {
    const target = event.target;
    if (!(target instanceof Element)) return;
    if (!navLinks.classList.contains('open')) return;
    if (!navLinks.contains(target) && !navToggle.contains(target)) {
      closeNav();
    }
  });

  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') {
      closeNav();
      navToggle.focus();
    }
  });
}

function animateCounter(node) {
  const raw = node.getAttribute('data-counter');
  if (!raw) return;

  const value = Number(raw);
  if (!Number.isFinite(value)) return;

  const isFloat = raw.includes('.');
  const initial = (node.textContent || '').trim();
  const suffix = initial.includes('%') ? '%' : initial.includes('h') ? ' h' : initial.includes('+') ? '+' : '';

  if (reduceMotionQuery.matches) {
    const finalValue = isFloat ? value.toFixed(1) : value >= 1000 ? value.toLocaleString() : String(Math.round(value));
    node.textContent = `${finalValue}${suffix}`;
    return;
  }

  const duration = 1100;
  const start = performance.now();

  function tick(now) {
    const t = Math.min(1, (now - start) / duration);
    const eased = 1 - Math.pow(1 - t, 3);
    const current = value * eased;
    const display = isFloat ? current.toFixed(1) : Math.round(current).toString();
    if (value >= 1000 && !isFloat) {
      node.textContent = `${Number(display).toLocaleString()}${suffix}`;
    } else {
      node.textContent = `${display}${suffix}`;
    }
    if (t < 1) requestAnimationFrame(tick);
  }

  requestAnimationFrame(tick);
}

const observer =
  'IntersectionObserver' in window
    ? new IntersectionObserver(
        (entries, obs) => {
          entries.forEach((entry) => {
            if (entry.isIntersecting) {
              animateCounter(entry.target);
              obs.unobserve(entry.target);
            }
          });
        },
        { threshold: 0.4 }
      )
    : null;

if (observer) {
  counterTargets.forEach((counter) => observer.observe(counter));
} else {
  counterTargets.forEach((counter) => animateCounter(counter));
}

function setPlaybackState(state) {
  if (!compareToggle) return;
  compareToggle.dataset.state = state;
  compareToggle.textContent = state === 'playing' ? 'Pause' : 'Play';
}

function setComparisonAspectFromSource() {
  if (!compareSourceVideo || !compareSourceVideo.videoWidth || !compareSourceVideo.videoHeight) return;
  const tileAspect = (compareSourceVideo.videoWidth / 4) / compareSourceVideo.videoHeight;
  compareWindows.forEach((node) => {
    node.style.setProperty('--compare-aspect', String(tileAspect));
  });
}

function renderComparisonFrame() {
  if (!compareSourceVideo || compareCanvases.length === 0) return;
  if (compareSourceVideo.readyState < 2) {
    compareRenderRaf = requestAnimationFrame(renderComparisonFrame);
    return;
  }

  const sourceWidth = compareSourceVideo.videoWidth;
  const sourceHeight = compareSourceVideo.videoHeight;
  const tileWidth = Math.floor(sourceWidth / 4);

  compareCanvases.forEach((canvas) => {
    const slot = Number(canvas.dataset.slot || 0);
    const sx = slot * tileWidth;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const renderW = Math.max(1, Math.floor(canvas.clientWidth));
    const renderH = Math.max(1, Math.floor(canvas.clientHeight));
    if (canvas.width !== renderW || canvas.height !== renderH) {
      canvas.width = renderW;
      canvas.height = renderH;
    }

    ctx.drawImage(
      compareSourceVideo,
      sx,
      0,
      tileWidth,
      sourceHeight,
      0,
      0,
      renderW,
      renderH
    );
  });

  if (compareToggle?.dataset.state === 'playing') {
    compareRenderRaf = requestAnimationFrame(renderComparisonFrame);
  }
}

function startComparisonRenderLoop() {
  if (compareRenderRaf) cancelAnimationFrame(compareRenderRaf);
  compareRenderRaf = requestAnimationFrame(renderComparisonFrame);
}

function setScene(source) {
  if (!compareSourceVideo) return;
  compareSourceVideo.src = source;
  compareSourceVideo.load();
  const playPromise = compareSourceVideo.play();
  if (playPromise && typeof playPromise.catch === 'function') {
    playPromise.catch(() => {});
  }
  setPlaybackState('playing');
  startComparisonRenderLoop();
}

if (compareSourceVideo) {
  compareSourceVideo.addEventListener('loadedmetadata', () => {
    setComparisonAspectFromSource();
    startComparisonRenderLoop();
  });
  window.addEventListener('resize', startComparisonRenderLoop);
}

if (sceneButtonsRoot && compareSourceVideo && compareCanvases.length > 0) {
  sceneButtonsRoot.addEventListener('click', (event) => {
    const btn = event.target.closest('.scene-btn');
    if (!btn) return;

    const source = btn.getAttribute('data-src');
    if (!source) return;

    sceneButtonsRoot.querySelectorAll('.scene-btn').forEach((node) => {
      node.classList.remove('is-active');
    });
    btn.classList.add('is-active');

    setScene(source);
  });
}

if (compareToggle && compareSourceVideo && compareCanvases.length > 0) {
  setPlaybackState('playing');
  compareToggle.addEventListener('click', () => {
    const isPlaying = compareToggle.dataset.state === 'playing';
    if (isPlaying) {
      compareSourceVideo.pause();
      setPlaybackState('paused');
      if (compareRenderRaf) {
        cancelAnimationFrame(compareRenderRaf);
        compareRenderRaf = null;
      }
    } else {
      const playPromise = compareSourceVideo.play();
      if (playPromise && typeof playPromise.catch === 'function') {
        playPromise.catch(() => {});
      }
      setPlaybackState('playing');
      startComparisonRenderLoop();
    }
  });
}
