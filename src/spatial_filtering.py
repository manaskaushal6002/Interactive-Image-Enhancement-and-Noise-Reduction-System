
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# --------------------------------------------------
# 1. Define project paths
# --------------------------------------------------

project_dir = Path(__file__).resolve().parent.parent
data_dir = project_dir / "data"
output_dir = project_dir / "outputs"

output_dir.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. Load the Gaussian-noisy image
# --------------------------------------------------

input_path = data_dir / "gaussian_noise.jpg"

noisy_image = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)

if noisy_image is None:
    raise FileNotFoundError(
        f"Could not load image: {input_path}"
    )

# --------------------------------------------------
# 3. Apply three spatial filters
# --------------------------------------------------

# Mean filtering: averages a 5x5 neighborhood
mean_filtered = cv2.blur(noisy_image, (5, 5))

# Gaussian filtering: weighted averaging
gaussian_filtered = cv2.GaussianBlur(
    noisy_image,
    (5, 5),
    sigmaX=1.0
)

# Median filtering: median of a 5x5 neighborhood
median_filtered = cv2.medianBlur(noisy_image, 5)

# --------------------------------------------------
# 4. Save filtered images
# --------------------------------------------------

cv2.imwrite(
    str(output_dir / "mean_filtered.jpg"),
    mean_filtered
)

cv2.imwrite(
    str(output_dir / "gaussian_filtered.jpg"),
    gaussian_filtered
)

cv2.imwrite(
    str(output_dir / "median_filtered.jpg"),
    median_filtered
)


# --------------------------------------------------
# 5. Interactive Image Viewer
# --------------------------------------------------

from matplotlib.widgets import RadioButtons

# Images available for inspection
image_options = {
    "Original Noisy": noisy_image,
    "Mean Filter": mean_filtered,
    "Gaussian Filter": gaussian_filtered,
    "Median Filter": median_filtered
}

# Create a large figure with space for the image and controls
fig = plt.figure(figsize=(12, 9))

image_axis = fig.add_axes([0.05, 0.08, 0.70, 0.84])
control_axis = fig.add_axes([0.79, 0.35, 0.18, 0.30])

# Display the original image first
current_image = image_axis.imshow(
    image_options["Original Noisy"],
    cmap="gray",
    vmin=0,
    vmax=255
)

image_axis.set_title(
    "Original Gaussian Noisy Image",
    fontsize=15,
    pad=12
)

image_axis.axis("off")

# Add selectable filter options
radio_buttons = RadioButtons(
    control_axis,
    list(image_options.keys()),
    active=0
)

control_axis.set_title("Select Image", fontsize=12, pad=12)


# Update the displayed image when an option is selected
def update_image(selected_name):
    current_image.set_data(image_options[selected_name])

    image_axis.set_title(
        selected_name,
        fontsize=15,
        pad=12
    )

    fig.canvas.draw_idle()


radio_buttons.on_clicked(update_image)

# Window title
fig.canvas.manager.set_window_title(
    "IPA Project - Spatial Filtering Explorer"
)

plt.show()
