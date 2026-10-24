
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# --------------------------------------------------
# 1. Load the original image
# --------------------------------------------------

image_path = Path(__file__).resolve().parent.parent / "data" / "sample.jpg"

image = cv2.imread(str(image_path))

if image is None:
    raise FileNotFoundError(f"Could not load image: {image_path}")

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# --------------------------------------------------
# 2. Implement contrast stretching
# --------------------------------------------------

def contrast_stretching(input_image):

    minimum_intensity = np.min(input_image)
    maximum_intensity = np.max(input_image)

    # Handle the case where all pixels have the same intensity
    if maximum_intensity == minimum_intensity:
        return input_image.copy()

    # Apply the contrast stretching formula
    stretched_image = (
        (input_image.astype(np.float32) - minimum_intensity)
        / (maximum_intensity - minimum_intensity)
        * 255
    )

    # Convert the result to an 8-bit image
    stretched_image = np.clip(stretched_image, 0, 255).astype(np.uint8)

    return stretched_image


# --------------------------------------------------
# 3. Apply contrast stretching
# --------------------------------------------------

enhanced_image = contrast_stretching(gray_image)


# --------------------------------------------------
# 4. Print image statistics
# --------------------------------------------------

print("\n========== ORIGINAL IMAGE ==========")
print("Minimum intensity:", np.min(gray_image))
print("Maximum intensity:", np.max(gray_image))
print("Mean intensity:", np.mean(gray_image))
print("Standard deviation:", np.std(gray_image))

print("\n========== ENHANCED IMAGE ==========")
print("Minimum intensity:", np.min(enhanced_image))
print("Maximum intensity:", np.max(enhanced_image))
print("Mean intensity:", np.mean(enhanced_image))
print("Standard deviation:", np.std(enhanced_image))


# --------------------------------------------------
# 5. Calculate histograms
# --------------------------------------------------

original_histogram = cv2.calcHist(
    [gray_image], [0], None, [256], [0, 256]
)

enhanced_histogram = cv2.calcHist(
    [enhanced_image], [0], None, [256], [0, 256]
)


# --------------------------------------------------
# 6. Display original and enhanced images
# --------------------------------------------------

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(gray_image, cmap="gray", vmin=0, vmax=255)
plt.title("Original Grayscale Image")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(enhanced_image, cmap="gray", vmin=0, vmax=255)
plt.title("Contrast Stretched Image")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.plot(original_histogram)
plt.title("Original Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Number of Pixels")
plt.xlim([0, 256])
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 4)
plt.plot(enhanced_histogram)
plt.title("Enhanced Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Number of Pixels")
plt.xlim([0, 256])
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# --------------------------------------------------
# 7. Save the enhanced image
# --------------------------------------------------

output_path = (
    Path(__file__).resolve().parent.parent
    / "outputs"
    / "contrast_stretched.jpg"
)

cv2.imwrite(str(output_path), enhanced_image)

print("\nEnhanced image saved to:", output_path)

def histogram_equalization(input_image):
    """
    Improve global contrast using histogram equalization.

    Parameters:
        input_image: An 8-bit grayscale image.

    Returns:
        An equalized grayscale image.
    """

    equalized_image = cv2.equalizeHist(input_image)

    return equalized_image

# Apply histogram equalization
equalized_image = histogram_equalization(gray_image)

# Calculate the histogram of the equalized image
equalized_histogram = cv2.calcHist(
    [equalized_image], [0], None, [256], [0, 256]
)

# Display histogram equalization results
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(gray_image, cmap="gray", vmin=0, vmax=255)
plt.title("Original Grayscale Image")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(equalized_image, cmap="gray", vmin=0, vmax=255)
plt.title("Histogram Equalized Image")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.plot(cv2.calcHist(
    [gray_image], [0], None, [256], [0, 256]
))
plt.title("Original Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Number of Pixels")
plt.xlim([0, 256])
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 4)
plt.plot(equalized_histogram)
plt.title("Equalized Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Number of Pixels")
plt.xlim([0, 256])
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# --------------------------------------------------
# CLAHE: Local Contrast Enhancement
# --------------------------------------------------

# Create the CLAHE object
clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

# Apply CLAHE to the grayscale image
clahe_image = clahe.apply(gray_image)

# Display original, histogram equalization and CLAHE
equalized_image = cv2.equalizeHist(gray_image)

plt.figure(figsize=(12, 5))

plt.subplot(1, 3, 1)
plt.imshow(gray_image, cmap="gray", vmin=0, vmax=255)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(equalized_image, cmap="gray", vmin=0, vmax=255)
plt.title("Global Histogram Equalization")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(clahe_image, cmap="gray", vmin=0, vmax=255)
plt.title("CLAHE")
plt.axis("off")

plt.tight_layout()
plt.show()

# Save the result
output_path = (
    Path(__file__).resolve().parent.parent
    / "outputs"
    / "clahe_image.jpg"
)

cv2.imwrite(str(output_path), clahe_image)

print("CLAHE image saved to:", output_path)

