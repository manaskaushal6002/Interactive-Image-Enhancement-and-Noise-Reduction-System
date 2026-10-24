import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --------------------------------------------------
# 1. Load the image
# --------------------------------------------------

image_path = Path(__file__).resolve().parent.parent / "data" / "sample.jpg"

image = cv2.imread(str(image_path))

if image is None:
    raise FileNotFoundError(f"Could not load image: {image_path}")

# Convert BGR to RGB for Matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# --------------------------------------------------
# 2. Analyze image properties
# --------------------------------------------------

height, width, channels = image.shape

print("\n========== IMAGE INFORMATION ==========")
print("Image dimensions:", width, "x", height)
print("Number of color channels:", channels)
print("Data type:", image.dtype)

print("\n========== GRAYSCALE STATISTICS ==========")
print("Minimum intensity:", np.min(gray_image))
print("Maximum intensity:", np.max(gray_image))
print("Mean intensity:", np.mean(gray_image))
print("Standard deviation:", np.std(gray_image))


# --------------------------------------------------
# 3. Calculate the grayscale histogram
# --------------------------------------------------

histogram = cv2.calcHist(
    [gray_image],
    [0],
    None,
    [256],
    [0, 256]
)

print("\n========== HISTOGRAM INFORMATION ==========")
print("Number of intensity bins:", len(histogram))
print("Total number of pixels:", int(np.sum(histogram)))


# --------------------------------------------------
# 4. Calculate histograms for RGB channels
# --------------------------------------------------

red_histogram = cv2.calcHist([image_rgb], [0], None, [256], [0, 256])
green_histogram = cv2.calcHist([image_rgb], [1], None, [256], [0, 256])
blue_histogram = cv2.calcHist([image_rgb], [2], None, [256], [0, 256])


# --------------------------------------------------
# 5. Display the image and its grayscale histogram
# --------------------------------------------------

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.plot(histogram)
plt.title("Grayscale Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Number of Pixels")
plt.xlim([0, 256])
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# --------------------------------------------------
# 6. Display individual RGB histograms
# --------------------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(red_histogram, label="Red")
plt.plot(green_histogram, label="Green")
plt.plot(blue_histogram, label="Blue")

plt.title("RGB Channel Histograms")
plt.xlabel("Pixel Intensity")
plt.ylabel("Number of Pixels")
plt.xlim([0, 256])
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()