
import cv2
import numpy as np
from pathlib import Path

# --------------------------------------------------
# 1. Load the clean reference image
# --------------------------------------------------

project_dir = Path(__file__).resolve().parent.parent
input_path = project_dir / "data" / "sample.jpg"
data_dir = project_dir / "data"

image = cv2.imread(str(input_path))

if image is None:
    raise FileNotFoundError(
        f"Could not load reference image: {input_path}"
    )

# Use grayscale images for our initial experiments
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Use a fixed seed so the random noise is reproducible
rng = np.random.default_rng(42)


# --------------------------------------------------
# 2. Generate Gaussian noise
# --------------------------------------------------

gaussian_noise = rng.normal(
    loc=0,
    scale=20,
    size=gray_image.shape
)

noisy_gaussian = np.clip(
    gray_image.astype(np.float32) + gaussian_noise,
    0,
    255
).astype(np.uint8)

cv2.imwrite(
    str(data_dir / "gaussian_noise.jpg"),
    noisy_gaussian
)


# --------------------------------------------------
# 3. Generate salt-and-pepper noise
# --------------------------------------------------

salt_pepper = gray_image.copy()

noise_mask = rng.random(gray_image.shape)

# Approximately 2% white pixels and 2% black pixels
salt_pepper[noise_mask < 0.02] = 0
salt_pepper[noise_mask > 0.98] = 255

cv2.imwrite(
    str(data_dir / "salt_pepper_noise.jpg"),
    salt_pepper
)


# --------------------------------------------------
# 4. Generate periodic noise
# --------------------------------------------------

height, width = gray_image.shape

x = np.arange(width)
y = np.arange(height)

X, Y = np.meshgrid(x, y)

# Sinusoidal pattern added to the original image
periodic_pattern = 25 * np.sin(2 * np.pi * X / 16)

periodic_noisy = np.clip(
    gray_image.astype(np.float32) + periodic_pattern,
    0,
    255
).astype(np.uint8)

cv2.imwrite(
    str(data_dir / "periodic_noise.jpg"),
    periodic_noisy
)


# --------------------------------------------------
# 5. Generate a blurred image
# --------------------------------------------------

blurred_image = cv2.GaussianBlur(
    gray_image,
    (9, 9),
    sigmaX=2
)

cv2.imwrite(
    str(data_dir / "blurred_image.jpg"),
    blurred_image
)


# --------------------------------------------------
# 6. Print the results
# --------------------------------------------------

print("Test images generated successfully!")

print("\nFiles created inside the data folder:")

for filename in [
    "gaussian_noise.jpg",
    "salt_pepper_noise.jpg",
    "periodic_noise.jpg",
    "blurred_image.jpg"
]:
    print(data_dir / filename)
