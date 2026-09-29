# Interactive Generative Poster (Streamlit version) + click-to-place blobs
# Based on the class code: blob(), make_palette(), and the layer loop are unchanged.
import io
import random, math
import numpy as np
import matplotlib
matplotlib.use("Agg")                        # draw without a screen (needed on servers)
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from PIL import Image
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

POSTER_TITLE = "Generative Poster"           # text printed on the poster
FIG_W, FIG_H, DPI = 6, 8, 100                # figure size (same as the class code)

# Blob shape (unchanged)
def blob(center=(0.5, 0.5), r=0.3, points=200, wobble=0.15):
    angles = np.linspace(0, 2 * math.pi, points, endpoint=False)
    radii = r * (1 + wobble * (np.random.rand(points) - 0.5))
    x = center[0] + radii * np.cos(angles)
    y = center[1] + radii * np.sin(angles)
    return x, y

# Simple palette generator (unchanged)
def make_palette(k=6, mode="pastel", base_h=0.60):
    cols = []
    for _ in range(k):
        if mode == "pastel":
            h = random.random(); s = random.uniform(0.15, 0.35); v = random.uniform(0.9, 1.0)
        elif mode == "vivid":
            h = random.random(); s = random.uniform(0.8, 1.0); v = random.uniform(0.8, 1.0)
        elif mode == "mono":
            h = base_h; s = random.uniform(0.2, 0.6); v = random.uniform(0.5, 1.0)
        else:  # random
            h = random.random(); s = random.uniform(0.3, 1.0); v = random.uniform(0.5, 1.0)
        cols.append(tuple(hsv_to_rgb([h, s, v])))
    return cols

# Main drawing function: returns a figure instead of calling plt.show()
# clicked_blobs: list of (x, y, radius, seed) placed by mouse clicks
def draw_poster(n_layers=8, wobble=0.15, palette_mode="pastel", seed=0, clicked_blobs=()):
    random.seed(seed)
    np.random.seed(seed)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.axis("off")
    ax.set_facecolor((0.97, 0.97, 0.97))

    palette = make_palette(6, mode=palette_mode)

    # --- Original layer loop (unchanged) ---
    for _ in range(n_layers):
        cx, cy = random.random(), random.random()
        rr = random.uniform(0.15, 0.45)
        x, y = blob((cx, cy), r=rr, wobble=wobble)
        color = random.choice(palette)
        alpha = random.uniform(0.3, 0.6)
        ax.fill(x, y, color=color, alpha=alpha, edgecolor=(0, 0, 0, 0))

    # --- NEW: freeze the axes limits so a click can be converted to a position ---
    ax.autoscale_view()
    ax.set_xlim(ax.get_xlim())
    ax.set_ylim(ax.get_ylim())

    # --- NEW: blobs placed by clicking (each has its own seed, so the
    #     original layers above never change when you add or remove blobs) ---
    for (bx, by, br, bseed) in clicked_blobs:
        random.seed(bseed)
        np.random.seed(bseed)
        x, y = blob((bx, by), r=br, wobble=wobble)
        color = random.choice(palette)
        alpha = random.uniform(0.3, 0.6)
        ax.fill(x, y, color=color, alpha=alpha, edgecolor=(0, 0, 0, 0))

    ax.text(0.05, 0.95, f"{POSTER_TITLE} • {palette_mode}",
            transform=ax.transAxes, fontsize=12, weight="bold")
    return fig

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Interactive Generative Poster", layout="centered")
st.title("Interactive Generative Poster")
st.caption("Arts and Advanced Big Data | From Colab to the Web")

# session_state keeps values between reruns (Streamlit reruns the script on every interaction).
if "blobs" not in st.session_state:
    st.session_state.blobs = []          # each item: (x, y, radius, seed)
if "last_click" not in st.session_state:
    st.session_state.last_click = None   # remembers the last click so it is not added twice

st.sidebar.header("Controls")
n_layers = st.sidebar.slider("Layers", min_value=3, max_value=20, value=8, step=1)
wobble = st.sidebar.slider("Wobble", min_value=0.01, max_value=0.30, value=0.15, step=0.01)
palette_mode = st.sidebar.selectbox("Palette mode", ["pastel", "vivid", "mono", "random"])
seed = st.sidebar.slider("Seed", min_value=0, max_value=9999, value=0, step=1)

if st.sidebar.button("Undo last blob") and st.session_state.blobs:
    st.session_state.blobs.pop()
if st.sidebar.button("Clear blobs"):
    st.session_state.blobs.clear()
st.sidebar.caption("Click anywhere on the poster to place a new blob.")

# Draw the poster and turn it into an image.
fig = draw_poster(n_layers, wobble, palette_mode, seed, tuple(st.session_state.blobs))
buffer = io.BytesIO()
fig.savefig(buffer, format="png", dpi=DPI)
image = Image.open(buffer).convert("RGB")
img_w, img_h = image.size
inverse = fig.axes[0].transData.inverted()   # pixel position -> poster position
plt.close(fig)                               # free memory

# Show the image AND get the position of the user's click on it.
click = streamlit_image_coordinates(image, width=480, key="poster")

if click is not None:
    click_id = (click["x"], click["y"], click.get("unix_time"))
    if click_id != st.session_state.last_click:          # ignore old clicks on reruns
        st.session_state.last_click = click_id

        # Scale the click to the real image size (y is measured from the top).
        px = click["x"] / click["width"] * img_w
        py = click["y"] / click["height"] * img_h
        data_x, data_y = inverse.transform((px, img_h - py))

        rng = random.Random()
        st.session_state.blobs.append((
            float(data_x), float(data_y),
            rng.uniform(0.1, 0.3),                       # random size
            rng.randrange(10**6),                        # random shape/color seed
        ))
        st.rerun()                                       # redraw with the new blob
