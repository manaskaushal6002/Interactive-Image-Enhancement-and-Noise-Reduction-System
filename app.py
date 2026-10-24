
import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Smart Image Enhancement System",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# 2. Custom dashboard styling
# --------------------------------------------------

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #10141f 0%, #171d2b 100%);
    }

    [data-testid="stSidebar"] {
        background-color: #121824;
        border-right: 1px solid #2b3548;
    }

    .hero {
        padding: 28px;
        border-radius: 18px;
        background: linear-gradient(120deg, #202c43, #263957);
        border: 1px solid #354966;
        margin-bottom: 24px;
    }

    .hero h1 {
        margin: 0;
        font-size: 2.15rem;
        color: #f4f7ff;
    }

    .hero p {
        margin: 10px 0 0 0;
        color: #c0cce0;
        font-size: 1rem;
    }

    .section-title {
        font-size: 1.3rem;
        font-weight: 650;
        margin-top: 12px;
        margin-bottom: 14px;
    }

    div[data-testid="stMetric"] {
        background-color: #1b2433;
        border: 1px solid #303d52;
        padding: 16px;
        border-radius: 12px;
    }

    div[data-testid="stMetricLabel"] {
        color: #bac8dc;
    }

    div[data-testid="stMetricValue"] {
        color: #f2f6ff;
    }

    div.stButton > button,
    div.stDownloadButton > button {
        border-radius: 10px;
        min-height: 42px;
        font-weight: 600;
    }

    .stCaption {
        color: #aab8cc;
    }

    hr {
        border-color: #30394a;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 3. Dashboard header
# --------------------------------------------------

st.markdown("""
<div class="hero">
    <h1>🖼️ Smart Image Enhancement System</h1>
    <p>
        Enhance image contrast, reduce noise, sharpen details,
        and explore grayscale intensity distributions.
    </p>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 4. Sidebar controls
# --------------------------------------------------

st.sidebar.title("⚙️ Processing Studio")
st.sidebar.caption("Upload an image and select an operation.")

uploaded_file = st.sidebar.file_uploader(
    "Upload your image",
    type=["jpg", "jpeg", "png", "bmp"]
)

operation = st.sidebar.selectbox(
    "Choose processing technique",
    [
        "Contrast Stretching",
        "Histogram Equalization",
        "CLAHE",
        "Mean Filter",
        "Gaussian Filter",
        "Median Filter",
        "Laplacian Sharpening",
        "Unsharp Masking"
    ]
)

operation_descriptions = {
    "Contrast Stretching":
        "Expands the intensity range to improve global contrast.",
    "Histogram Equalization":
        "Redistributes grayscale intensities to improve contrast.",
    "CLAHE":
        "Enhances local contrast while limiting excessive amplification.",
    "Mean Filter":
        "Smooths the image by averaging neighboring pixels.",
    "Gaussian Filter":
        "Reduces noise using a Gaussian-weighted neighborhood.",
    "Median Filter":
        "Removes isolated noise using neighborhood median values.",
    "Laplacian Sharpening":
        "Enhances fine details using second-order derivatives.",
    "Unsharp Masking":
        "Sharpens the image by adding high-frequency details."
}

st.sidebar.divider()
st.sidebar.markdown("### 📘 Selected technique")
st.sidebar.write(operation_descriptions[operation])

st.sidebar.divider()
st.sidebar.caption("Image Processing & Applications | Academic Project")


# --------------------------------------------------
# 5. Upload validation and image loading
# --------------------------------------------------

if uploaded_file is None:
    st.info("👈 Upload an image using the sidebar to begin.")

    st.markdown("### ✨ Available capabilities")

    col1, col2, col3, col4 = st.columns(4)

    col1.markdown("**🎚️ Contrast**")
    col1.caption("Stretching, equalization, CLAHE")

    col2.markdown("**🧹 Noise Reduction**")
    col2.caption("Mean, Gaussian, median filtering")

    col3.markdown("**🔍 Sharpening**")
    col3.caption("Laplacian and unsharp masking")

    col4.markdown("**📊 Analysis**")
    col4.caption("Intensity statistics and histograms")

    st.stop()

try:
    pil_image = Image.open(uploaded_file).convert("L")
    gray_image = np.array(pil_image, dtype=np.uint8)
except Exception:
    st.error("Unable to read this image. Please upload a valid image.")
    st.stop()


# --------------------------------------------------
# 6. Image processing controls and algorithms
# --------------------------------------------------

processed_image = gray_image.copy()

if operation == "Contrast Stretching":

    minimum = int(np.min(gray_image))
    maximum = int(np.max(gray_image))

    if maximum > minimum:
        processed_image = np.clip(
            (gray_image.astype(np.float32) - minimum)
            * 255.0 / (maximum - minimum),
            0,
            255
        ).astype(np.uint8)

elif operation == "Histogram Equalization":

    processed_image = cv2.equalizeHist(gray_image)

elif operation == "CLAHE":

    clip_limit = st.sidebar.slider(
        "Contrast limit",
        min_value=1.0,
        max_value=5.0,
        value=2.0,
        step=0.5
    )

    tile_size = st.sidebar.select_slider(
        "Tile grid size",
        options=[4, 8, 12, 16],
        value=8
    )

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(tile_size, tile_size)
    )

    processed_image = clahe.apply(gray_image)

elif operation == "Mean Filter":

    kernel_size = st.sidebar.select_slider(
        "Kernel size",
        options=[3, 5, 7, 9],
        value=5
    )

    processed_image = cv2.blur(
        gray_image,
        (kernel_size, kernel_size)
    )

elif operation == "Gaussian Filter":

    kernel_size = st.sidebar.select_slider(
        "Kernel size",
        options=[3, 5, 7, 9],
        value=5
    )

    sigma = st.sidebar.slider(
        "Sigma",
        min_value=0.1,
        max_value=3.0,
        value=1.0,
        step=0.1
    )

    processed_image = cv2.GaussianBlur(
        gray_image,
        (kernel_size, kernel_size),
        sigmaX=sigma
    )

elif operation == "Median Filter":

    kernel_size = st.sidebar.select_slider(
        "Kernel size",
        options=[3, 5, 7, 9],
        value=5
    )

    processed_image = cv2.medianBlur(
        gray_image,
        kernel_size
    )

elif operation == "Laplacian Sharpening":

    strength = st.sidebar.slider(
        "Sharpening strength",
        min_value=0.1,
        max_value=2.0,
        value=1.0,
        step=0.1
    )

    laplacian = cv2.Laplacian(
        gray_image,
        cv2.CV_32F,
        ksize=3
    )

    processed_image = np.clip(
        gray_image.astype(np.float32)
        - strength * laplacian,
        0,
        255
    ).astype(np.uint8)

elif operation == "Unsharp Masking":

    strength = st.sidebar.slider(
        "Sharpening strength",
        min_value=0.5,
        max_value=3.0,
        value=1.5,
        step=0.1
    )

    blurred = cv2.GaussianBlur(
        gray_image,
        (5, 5),
        sigmaX=1.0
    )

    detail_mask = (
        gray_image.astype(np.float32)
        - blurred.astype(np.float32)
    )

    processed_image = np.clip(
        gray_image.astype(np.float32)
        + strength * detail_mask,
        0,
        255
    ).astype(np.uint8)


# --------------------------------------------------
# 7. Image comparison
# --------------------------------------------------

st.markdown('<div class="section-title">🔍 Image Comparison</div>',
            unsafe_allow_html=True)

st.caption(
    f"Current operation: {operation} — "
    f"{operation_descriptions[operation]}"
)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("#### 🖼️ Original Image")
    st.image(
        gray_image,
        use_container_width=True,
        clamp=True
    )

with col2:
    st.markdown("#### ✨ Enhanced Image")
    st.image(
        processed_image,
        use_container_width=True,
        clamp=True
    )

st.divider()


# --------------------------------------------------
# 8. Image statistics
# --------------------------------------------------

st.markdown('<div class="section-title">📊 Image Statistics</div>',
            unsafe_allow_html=True)

original_mean = float(np.mean(gray_image))
processed_mean = float(np.mean(processed_image))

original_std = float(np.std(gray_image))
processed_std = float(np.std(processed_image))

original_min = int(np.min(gray_image))
original_max = int(np.max(gray_image))

processed_min = int(np.min(processed_image))
processed_max = int(np.max(processed_image))

metric1, metric2, metric3, metric4 = st.columns(4)

metric1.metric(
    "Mean Intensity",
    f"{processed_mean:.2f}",
    f"{processed_mean - original_mean:+.2f}"
)

metric2.metric(
    "Intensity Std. Dev.",
    f"{processed_std:.2f}",
    f"{processed_std - original_std:+.2f}"
)

metric3.metric(
    "Intensity Range",
    f"{processed_min}–{processed_max}",
    f"{(processed_max - processed_min) - (original_max - original_min):+d}"
)

metric4.metric(
    "Image Resolution",
    f"{gray_image.shape[1]} × {gray_image.shape[0]}"
)

st.caption(
    "Statistics describe grayscale pixel intensities. "
    "An increase in standard deviation does not always mean better image quality."
)

st.divider()


# --------------------------------------------------
# 9. Histogram comparison
# --------------------------------------------------

st.markdown('<div class="section-title">📈 Intensity Distribution</div>',
            unsafe_allow_html=True)

st.caption(
    "Compare how the selected operation changes the distribution "
    "of pixel intensity values from 0 (black) to 255 (white)."
)

fig, axes = plt.subplots(
    1, 2,
    figsize=(12, 4),
    facecolor="#171d2b"
)

for ax in axes:
    ax.set_facecolor("#202a3b")
    ax.tick_params(colors="#d5deed")
    ax.xaxis.label.set_color("#d5deed")
    ax.yaxis.label.set_color("#d5deed")
    ax.title.set_color("#f2f6ff")
    for spine in ax.spines.values():
        spine.set_color("#43516a")

axes[0].hist(
    gray_image.ravel(),
    bins=256,
    range=(0, 256),
    color="#8ea4c8"
)
axes[0].set_title("Original Histogram")
axes[0].set_xlabel("Pixel Intensity")
axes[0].set_ylabel("Pixel Count")
axes[0].set_xlim(0, 255)

axes[1].hist(
    processed_image.ravel(),
    bins=256,
    range=(0, 256),
    color="#70c1b3"
)
axes[1].set_title("Processed Histogram")
axes[1].set_xlabel("Pixel Intensity")
axes[1].set_ylabel("Pixel Count")
axes[1].set_xlim(0, 255)

fig.tight_layout()
st.pyplot(fig)
plt.close(fig)


# --------------------------------------------------
# 10. Download processed image
# --------------------------------------------------

st.divider()

st.markdown('<div class="section-title">💾 Export Result</div>',
            unsafe_allow_html=True)

success, encoded_image = cv2.imencode(
    ".png",
    processed_image
)

if success:
    st.download_button(
        label="⬇️ Download Processed Image (PNG)",
        data=encoded_image.tobytes(),
        file_name=f"{operation.lower().replace(' ', '_')}.png",
        mime="image/png",
        use_container_width=True
    )

st.caption(
    "The uploaded image is converted to grayscale before processing. "
    "The downloaded image is a PNG file."
)
