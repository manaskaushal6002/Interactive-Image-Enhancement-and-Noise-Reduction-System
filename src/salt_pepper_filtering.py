
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons
from pathlib import Path

# --------------------------------------------------
# 1. Define paths
# --------------------------------------------------

project_dir = Path(__file__).resolve().parent.parent
data_dir = project_dir / "data"
output_dir = project_dir / "outputs"

output_dir.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. Load salt-and-pepper noisy image
# --------------------------------------------------

input_path = data_dir / "salt_pepper_noise.jpg"

noisy_image = cv2.imread(
    str(input_path),
    cv2.IMREAD_GRAYSCALE
)

if noisy_image is None:
    raise FileNotFoundError(
        f"Could not load image: {input_path}"
    )

# --------------------------------------------------
# 3. Apply spatial filters
# --------------------------------------------------

mean_filtered = cv2.blur(noisy_image, (5, 5))

gaussian_filtered = cv2.GaussianBlur(
    noisy_image,
    (5, 5),
    sigmaX=1.0
)

median_filtered = cv2.medianBlur(noisy_image, 5)

# --------------------------------------------------
# 4. Save results
# --------------------------------------------------

cv2.imwrite(
    str(output_dir / "salt_pepper_mean.jpg"),
    mean_filtered
)

cv2.imwrite(
    str(output_dir / "salt_pepper_gaussian.jpg"),
    gaussian_filtered
)

cv2.imwrite(
    str(output_dir / "salt_pepper_median.jpg"),
    median_filtered
)

# --------------------------------------------------
# 5. Create interactive image viewer
# --------------------------------------------------

image_options = {
    "Original Noisy": noisy_image,
    "Mean Filter": mean_filtered,
    "Gaussian Filter": gaussian_filtered,
    "Median Filter": median_filtered
}

fig = plt.figure(figsize=(12, 9))

image_axis = fig.add_axes([0.05, 0.08, 0.70, 0.84])
control_axis = fig.add_axes([0.79, 0.35, 0.18, 0.30])

current_image = image_axis.imshow(
    noisy_image,
    cmap="gray",
    vmin=0,
    vmax=255
)

image_axis.set_title(
    "Original Salt-and-Pepper Noise",
    fontsize=15
)

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
    "IPA Project - Salt-and-Pepper Filtering"
)

plt.show()

print("Salt-and-pepper filtering completed!")
print("Results saved in:", output_dir)
