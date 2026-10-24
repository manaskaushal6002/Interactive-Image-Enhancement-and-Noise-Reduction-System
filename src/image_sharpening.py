
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons
from pathlib import Path

# --------------------------------------------------
# 1. Define project paths
# --------------------------------------------------

project_dir = Path(__file__).resolve().parent.parent
data_dir = project_dir / "data"
output_dir = project_dir / "outputs"

output_dir.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. Load the clean and blurred images
# --------------------------------------------------

original_path = data_dir / "sample.jpg"
blurred_path = data_dir / "blurred_image.jpg"

original_image = cv2.imread(
    str(original_path),
    cv2.IMREAD_GRAYSCALE
)

blurred_image = cv2.imread(
    str(blurred_path),
    cv2.IMREAD_GRAYSCALE
)

if original_image is None:
    raise FileNotFoundError(
        f"Could not load image: {original_path}"
    )

if blurred_image is None:
    raise FileNotFoundError(
        f"Could not load image: {blurred_path}"
    )

# --------------------------------------------------
# 3. Laplacian sharpening
# --------------------------------------------------

laplacian = cv2.Laplacian(
    original_image,
    cv2.CV_32F,
    ksize=3
)

# Subtract the Laplacian response to sharpen
laplacian_sharpened = np.clip(
    original_image.astype(np.float32) - laplacian,
    0,
    255
).astype(np.uint8)

# --------------------------------------------------
# 4. Unsharp masking
# --------------------------------------------------

blur_for_mask = cv2.GaussianBlur(
    original_image,
    (5, 5),
    sigmaX=1.0
)

detail_mask = (
    original_image.astype(np.float32)
    - blur_for_mask.astype(np.float32)
)

unsharp_masked = np.clip(
    original_image.astype(np.float32)
    + 1.5 * detail_mask,
    0,
    255
).astype(np.uint8)

# --------------------------------------------------
# 5. Sharpen the blurred image using unsharp masking
# --------------------------------------------------

blurred_smooth = cv2.GaussianBlur(
    blurred_image,
    (5, 5),
    sigmaX=1.0
)

blurred_detail = (
    blurred_image.astype(np.float32)
    - blurred_smooth.astype(np.float32)
)

restored_sharp = np.clip(
    blurred_image.astype(np.float32)
    + 2.0 * blurred_detail,
    0,
    255
).astype(np.uint8)

# --------------------------------------------------
# 6. Save output images
# --------------------------------------------------

outputs = {
    "laplacian_sharpened.jpg": laplacian_sharpened,
    "unsharp_masked.jpg": unsharp_masked,
    "blurred_unsharp.jpg": restored_sharp
}

for filename, image in outputs.items():
    cv2.imwrite(str(output_dir / filename), image)

# --------------------------------------------------
# 7. Interactive image viewer
# --------------------------------------------------

image_options = {
    "Original": original_image,
    "Blurred Input": blurred_image,
    "Laplacian Sharpening": laplacian_sharpened,
    "Unsharp Masking": unsharp_masked,
    "Blurred + Unsharp Mask": restored_sharp
}

fig = plt.figure(figsize=(12, 9))

image_axis = fig.add_axes([0.05, 0.08, 0.70, 0.84])
control_axis = fig.add_axes([0.79, 0.30, 0.19, 0.42])

current_image = image_axis.imshow(
    original_image,
    cmap="gray",
    vmin=0,
    vmax=255
)

image_axis.set_title("Original Image", fontsize=15)
image_axis.axis("off")

radio_buttons = RadioButtons(
    control_axis,
    list(image_options.keys()),
    active=0
)

control_axis.set_title("Select Image", fontsize=12)


def update_image(selected_name):
    current_image.set_data(image_options[selected_name])
    image_axis.set_title(selected_name, fontsize=15)
    fig.canvas.draw_idle()


radio_buttons.on_clicked(update_image)

fig.canvas.manager.set_window_title(
    "IPA Project - Image Sharpening Explorer"
)

plt.show()

print("Image sharpening completed!")
print("Output images saved in:", output_dir)
