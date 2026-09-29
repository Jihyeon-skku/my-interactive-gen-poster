import io

import numpy as np
import matplotlib
matplotlib.use("Agg")                       # draw without a screen (needed on servers)
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import to_rgb
from matplotlib.patches import Ellipse, Polygon
from scipy.ndimage import gaussian_filter
from PIL import Image
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

# ----------------------------------------------------------
# 1. SETTINGS (change these freely)
# ----------------------------------------------------------
POSTER_W, POSTER_H = 10, 14          # poster size in "poster units"
W_PX, H_PX = 400, 560                # resolution of the stripe image (higher = sharper, slower)
SEED = 42                            # same seed = same layout on every redraw
POSTER_TITLE = "Generative Poster"   # title printed at the bottom-left
DISPLAY_W = 480                      # width of the poster on the web page, in pixels

# Each theme has:
#   bg    -> background color
#   warm  -> stripe colors used mostly toward the poster edges
#   cool  -> stripe colors used mostly toward the poster center
#   glow  -> colors of the big soft glowing circles
#   lens  -> color of the small lens / hexagon shapes
#   ink   -> color of the title text and thin lines
THEMES = {
    "Sunrise": {
        "bg": "#ede4cf",
        "warm": ["#e8583c", "#f08a4b", "#f4b24c", "#f6cfa8", "#c8402f", "#f3e0b8"],
        "cool": ["#1f4e6b", "#2d6a7a", "#173a52", "#5b9aa0", "#7fb5b0"],
        "glow": ["#ffe9a8", "#f6a04d", "#f4a3a0", "#ffd9b0"],
        "lens": "#fff6e6",
        "ink": "#2a1d16",
    },
    "Twilight": {
        "bg": "#14172b",
        "warm": ["#ff6b6b", "#ff9f5a", "#ffd166", "#f78fb3", "#ffb8a1"],
        "cool": ["#3a86ff", "#5e60ce", "#48bfe3", "#2b2d6e", "#80ffdb"],
        "glow": ["#ffd6a5", "#ff9ecb", "#ffe66d", "#9bf6ff"],
        "lens": "#fff3d6",
        "ink": "#f6ecd9",
    },
    "Sage": {
        "bg": "#eef0e3",
        "warm": ["#e07a5f", "#f2cc8f", "#d4a373", "#e9c46a", "#f4a261"],
        "cool": ["#264653", "#2a9d8f", "#3d5a40", "#588157", "#a3b18a"],
        "glow": ["#f9f1c9", "#e9c46a", "#f2cc8f", "#dad7cd"],
        "lens": "#fffdf2",
        "ink": "#22301f",
    },
    "Rosewood": {
        "bg": "#f4e7e1",
        "warm": ["#c9184a", "#ff4d6d", "#ff758f", "#ffb3c1", "#f4a261"],
        "cool": ["#590d22", "#800f2f", "#3c1642", "#6d597a", "#b56576"],
        "glow": ["#ffccd5", "#ffb3c1", "#ffe5d9", "#fcd5ce"],
        "lens": "#fff5f0",
        "ink": "#3b0a1e",
    },
}
THEME_NAMES = list(THEMES.keys())

# Big glowing circles: (x, y, radius, opacity, glow color slot)
BIG_ORBS = [
    (2.3, 10.6, 2.1, 0.55, 0),
    (7.2,  8.6, 2.2, 0.60, 0),
    (4.1,  4.7, 1.3, 0.70, 1),
    (1.5,  3.3, 1.0, 0.60, 2),
    (8.3, 11.6, 1.2, 0.40, 2),
    (7.6,  3.9, 0.8, 0.50, 3),
]

# Coordinates of every pixel of the stripe image, in poster units (built once).
_cols, _rows = np.meshgrid(np.arange(W_PX), np.arange(H_PX))
XX = (_cols + 0.5) / W_PX * POSTER_W
YY = POSTER_H - (_rows + 0.5) / H_PX * POSTER_H

# Paper grain texture, created once so it does not flicker on redraw.
GRAIN = np.random.default_rng(1).normal(0, 1, (H_PX, W_PX, 1))

# ----------------------------------------------------------
# 2. HELPERS (unchanged)
# ----------------------------------------------------------
def paint_stripes(canvas, n_stripes, wobble, theme):
    """Paint vertical stripes that fade out at both ends.
    Every stripe has its own seed, so adding layers never moves existing stripes."""
    rows = np.arange(H_PX)[:, None]
    for i in range(n_stripes):
        rng = np.random.default_rng(SEED + i)
        x = rng.uniform(0.03, 0.97)                       # horizontal position (0..1)
        closeness = 1 - abs(x - 0.5) * 2                  # 1 at center, 0 at the edges
        # Cool colors are more likely near the center, warm colors near the edges.
        pool = theme["cool"] if rng.random() < 0.15 + 0.65 * closeness else theme["warm"]
        color = np.array(to_rgb(pool[rng.integers(len(pool))]))

        half = max(1, int(rng.uniform(2, 15) / 2))        # half of the stripe width in pixels
        c0 = max(int(x * W_PX) - half, 0)
        c1 = min(int(x * W_PX) + half, W_PX)
        if c1 <= c0:
            continue

        # Where the stripe starts and ends. Wobble makes the ends more irregular.
        top = (rng.uniform(-0.05, 0.40) + wobble * rng.uniform(-0.08, 0.08)) * H_PX
        bottom = (rng.uniform(0.45, 0.88) + wobble * rng.uniform(-0.08, 0.08)) * H_PX
        fade_top = rng.uniform(0.02, 0.12) * H_PX * (1 + wobble)
        fade_bottom = rng.uniform(0.06, 0.25) * H_PX * (1 + wobble)

        # Opacity along the stripe: 0 -> 1 -> 0 (smooth fade at both ends).
        profile = (np.clip((rows - top) / fade_top, 0, 1) *
                   np.clip((bottom - rows) / fade_bottom, 0, 1))
        profile = profile * profile * (3 - 2 * profile)
        a = (rng.uniform(0.5, 0.95) * profile)[..., None]

        canvas[:, c0:c1] = canvas[:, c0:c1] * (1 - a) + color * a

def add_orb(canvas, cx, cy, radius, color, alpha, wobble, seed):
    """Paint one soft glowing circle onto the stripe image.
    The edge is blurry, the center is brighter, and wobble makes the outline lumpy."""
    r_max = radius * 1.5
    c0 = max(int((cx - r_max) / POSTER_W * W_PX), 0)
    c1 = min(int((cx + r_max) / POSTER_W * W_PX) + 1, W_PX)
    r0 = max(int((POSTER_H - (cy + r_max)) / POSTER_H * H_PX), 0)
    r1 = min(int((POSTER_H - (cy - r_max)) / POSTER_H * H_PX) + 1, H_PX)
    if c1 <= c0 or r1 <= r0:
        return

    dx = XX[r0:r1, c0:c1] - cx
    dy = YY[r0:r1, c0:c1] - cy
    dist = np.hypot(dx, dy)
    angle = np.arctan2(dy, dx)

    # Lumpy radius: a few sine waves around the circle.
    rng = np.random.default_rng(seed)
    rad = np.full_like(angle, radius)
    for k in range(2, 6):
        rad += (wobble * radius * 0.12 * rng.uniform(0.3, 1.0) / (k - 1)
                * np.sin(k * angle + rng.uniform(0, 2 * np.pi)))

    d = dist / rad                                        # 0 at center, 1 at the edge
    mask = np.clip((1.0 - d) / 0.35, 0, 1)                # soft edge in the outer 35%
    mask = mask * mask * (3 - 2 * mask)
    core = np.clip(1 - d / 0.6, 0, 1)[..., None]          # brighter core

    base = np.array(to_rgb(color))
    rgb = base + (1 - base) * 0.45 * core                 # blend toward white in the core
    a = (alpha * mask)[..., None]
    canvas[r0:r1, c0:c1] = canvas[r0:r1, c0:c1] * (1 - a) + rgb * a

# ----------------------------------------------------------
# 3. POSTER DRAWING LOGIC
#    Same design as before. The only change: clicked blobs now arrive
#    through the `blobs` argument instead of a global variable.
#    Each blob: (x, y, radius, seed, color_index)
# ----------------------------------------------------------
def draw_original_poster(n_layers, wobble, theme, blobs):
    canvas = np.ones((H_PX, W_PX, 3)) * np.array(to_rgb(theme["bg"]))

    paint_stripes(canvas, n_layers * 6, wobble, theme)    # 6 stripes per layer

    for i, (x, y, r, alpha, gi) in enumerate(BIG_ORBS):
        add_orb(canvas, x, y, r, theme["glow"][gi % len(theme["glow"])],
                alpha, wobble, SEED + 500 + i)

    # Blobs placed by clicking are painted as glowing circles too.
    for (x, y, radius, seed, color_idx) in blobs:
        add_orb(canvas, x, y, radius, theme["glow"][color_idx % len(theme["glow"])],
                0.8, wobble, seed)

    # Dreamy glow: mix the sharp image with a blurred copy of itself.
    blurred = gaussian_filter(canvas, sigma=(7, 7, 0))
    canvas = 0.82 * canvas + 0.18 * blurred

    canvas = canvas + GRAIN * 0.015                       # paper grain
    return np.clip(canvas, 0, 1)

# ----------------------------------------------------------
# 4. SMALL LENS SHAPES, HIGHLIGHTS, AND TEXT (drawn on top)
# ----------------------------------------------------------
def draw_lenses(ax, theme):
    """Small pale ellipses and tall hexagons floating over the stripes."""
    rng = np.random.default_rng(SEED + 777)
    lens = theme["lens"]
    for _ in range(18):
        x = rng.uniform(0.8, POSTER_W - 0.8)
        y = rng.uniform(2.4, POSTER_H - 0.8)              # keep the title area clear
        opacity = rng.uniform(0.55, 0.95)
        if rng.random() < 0.7:                            # ellipse
            ax.add_patch(Ellipse((x, y), rng.uniform(0.5, 1.0), rng.uniform(0.3, 0.5),
                                 facecolor=lens, edgecolor="none",
                                 alpha=opacity, zorder=10))
        else:                                             # tall hexagon
            sx, sy = rng.uniform(0.18, 0.3), rng.uniform(0.4, 0.7)
            pts = np.array([(0, 1), (0.6, 0.5), (0.6, -0.5),
                            (0, -1), (-0.6, -0.5), (-0.6, 0.5)]) * [sx, sy] + [x, y]
            ax.add_patch(Polygon(pts, closed=True, facecolor=lens,
                                 edgecolor="white", linewidth=0.6,
                                 alpha=opacity, zorder=10))

def draw_added_highlights(ax, theme, blobs):
    """A small bright spot on each clicked blob so it looks like glass."""
    for (x, y, radius, seed, color_idx) in blobs:
        ax.add_patch(Ellipse((x - radius * 0.25, y + radius * 0.25),
                             radius * 0.45, radius * 0.3,
                             facecolor=theme["lens"], edgecolor="none",
                             alpha=0.85, zorder=12))

def draw_text(ax, theme, theme_name, n_layers, wobble):
    ink = theme["ink"]
    # Title: bottom-left, with a thin outline in the background color for readability.
    ax.text(0.9, 1.6, POSTER_TITLE, color=ink, fontsize=24,
            fontweight="bold", va="bottom", ha="left", zorder=61,
            path_effects=[pe.withStroke(linewidth=4, foreground=theme["bg"])])
    ax.plot([0.9, 6.0], [1.42, 1.42], color=ink, alpha=0.8, linewidth=1.2, zorder=61)
    ax.text(0.9, 0.95, f"{theme_name}  ·  {n_layers} layers  ·  wobble {wobble:.2f}",
            color=ink, fontsize=8, alpha=0.75, va="bottom", zorder=61)

# ----------------------------------------------------------
# 5. RENDER THE WHOLE POSTER TO PNG BYTES
#    st.cache_data remembers results, so identical settings are not redrawn.
# ----------------------------------------------------------
@st.cache_data(show_spinner=False, max_entries=50)
def render_poster_png(n_layers, wobble, theme_name, blobs, dpi=150):
    theme = THEMES[theme_name]
    image = draw_original_poster(n_layers, wobble, theme, blobs)

    fig, ax = plt.subplots(figsize=(6, 8.4))
    fig.subplots_adjust(0, 0, 1, 1)                       # poster fills the whole figure
    ax.set_axis_off()
    ax.imshow(image, extent=[0, POSTER_W, 0, POSTER_H],
              origin="upper", interpolation="bilinear", zorder=0)
    draw_lenses(ax, theme)
    draw_added_highlights(ax, theme, blobs)
    draw_text(ax, theme, theme_name, n_layers, wobble)
    ax.set_xlim(0, POSTER_W)
    ax.set_ylim(0, POSTER_H)
    ax.set_aspect("equal")

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi)
    plt.close(fig)                                        # free memory
    return buffer.getvalue()

# ----------------------------------------------------------
# 6. STREAMLIT PAGE
# ----------------------------------------------------------
st.set_page_config(page_title="Generative Poster", page_icon="🎨", layout="centered")
st.title("Generative Poster")

# session_state keeps values between reruns (Streamlit reruns the script on every interaction).
if "blobs" not in st.session_state:
    st.session_state.blobs = []          # each item: (x, y, radius, seed, color_index)
if "last_click" not in st.session_state:
    st.session_state.last_click = None   # remembers the last click so it is not added twice

# ----- Sidebar controls -----
with st.sidebar:
    st.header("Controls")
    n_layers = st.slider("Layers", 1, 30, 16)
    wobble = st.slider("Wobble", 0.0, 1.5, 0.5, 0.05)
    theme_name = st.select_slider("Palette", options=THEME_NAMES, value=THEME_NAMES[0])

    col1, col2 = st.columns(2)
    if col1.button("Undo last blob") and st.session_state.blobs:
        st.session_state.blobs.pop()
    if col2.button("Clear blobs"):
        st.session_state.blobs.clear()

    st.caption("Click anywhere on the poster to place a new glowing blob.")

# ----- Draw the poster -----
png_bytes = render_poster_png(n_layers, wobble, theme_name, tuple(st.session_state.blobs))
poster_image = Image.open(io.BytesIO(png_bytes))

# This shows the image AND reports where the user clicked on it.
click = streamlit_image_coordinates(poster_image, width=DISPLAY_W, key="poster")

# ----- Handle a click: add a blob at that spot -----
if click is not None:
    click_id = (click["x"], click["y"], click.get("unix_time"))
    if click_id != st.session_state.last_click:          # ignore old clicks on reruns
        st.session_state.last_click = click_id

        # Convert pixel position -> poster units (y is flipped: image top = poster top).
        x = click["x"] / click["width"] * POSTER_W
        y = POSTER_H - click["y"] / click["height"] * POSTER_H

        rng = np.random.default_rng()
        st.session_state.blobs.append((
            float(x), float(y),
            float(rng.uniform(0.5, 1.4)),                # random size
            int(rng.integers(0, 10**6)),                 # random shape seed
            int(rng.integers(0, 4)),                     # random glow color slot
        ))
        st.rerun()                                       # redraw with the new blob

st.download_button("Download PNG", data=png_bytes,
                   file_name="generative_poster.png", mime="image/png")
