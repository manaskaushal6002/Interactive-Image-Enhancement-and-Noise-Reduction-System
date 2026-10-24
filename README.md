# 🖼️ Image Enhancement and Noise Reduction System

A Python-based image processing application built using **OpenCV, NumPy, Matplotlib, Pillow, and Streamlit**. This project allows users to upload grayscale images and apply different image enhancement, noise reduction, and sharpening techniques through an interactive web interface.

The project demonstrates fundamental concepts of **Digital Image Processing (DIP)**, including contrast enhancement, histogram processing, spatial filtering, and image sharpening.

## 📌 Project Overview

Images captured under poor lighting conditions may have low contrast, noise, or blurred details. The Smart Image Enhancement System provides an interactive platform to process images using classical image processing algorithms and visually compare the original image with the processed result.

Users can upload an image, select an enhancement technique, adjust relevant parameters, compare results, analyze intensity histograms, and download the processed image.

## ✨ Features

### 1. Image Enhancement
- **Contrast Stretching:** Improves image contrast by expanding the intensity range.
- **Histogram Equalization:** Enhances global contrast by redistributing intensity values.
- **CLAHE (Contrast Limited Adaptive Histogram Equalization):** Improves local contrast while limiting excessive amplification.

### 2. Noise Reduction and Spatial Filtering
- **Mean Filter:** Reduces noise by averaging neighboring pixel values.
- **Gaussian Filter:** Smooths images using a Gaussian-weighted kernel.
- **Median Filter:** Reduces impulse noise, including salt-and-pepper noise.

### 3. Image Sharpening
- **Laplacian Sharpening:** Enhances image details using the Laplacian operator.
- **Unsharp Masking:** Sharpens images by enhancing the difference between the original image and its blurred version.

### 4. Interactive Dashboard
- Upload images in JPG, JPEG, PNG, and BMP formats.
- Select image processing operations through the sidebar.
- Adjust algorithm parameters interactively.
- Compare original and processed images side by side.
- View grayscale intensity histograms.
- Analyze mean intensity, intensity standard deviation, and image dimensions.
- Download the processed image in PNG format.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| OpenCV | Image processing and enhancement algorithms |
| NumPy | Numerical operations and pixel manipulation |
| Matplotlib | Histogram visualization |
| Pillow (PIL) | Image loading and conversion |
| Streamlit | Interactive web application interface |

## 📂 Project Structure

```text
IPA_Project/
│
├── app.py
├── requirements.txt
├── README.md
│
├── src/
│   ├── image_analysis.py
│   ├── enhancement.py
│   ├── create_test_images.py
│   ├── spatial_filtering.py
│   ├── salt_pepper_filtering.py
│   └── image_sharpening.py
│
├── data/
│   ├── sample.jpg
│   ├── gaussian_noise.jpg
│   ├── salt_pepper_noise.jpg
│   ├── periodic_noise.jpg
│   └── blurred_image.jpg
│
├── outputs/
│
└── report/
```

## ⚙️ Installation and Setup

### Prerequisites

Make sure you have installed:

- Python 3.10 or a compatible version
- pip (Python package manager)
- Git

### Step 1: Clone the Repository

Replace `YOUR_USERNAME` with your GitHub username.

```bash
git clone https://github.com/YOUR_USERNAME/IPA_Project.git
cd IPA_Project
```

### Step 2: Create a Virtual Environment

**Windows:**

```bash
python -m venv .venv
```

Activate the environment:

```bash
.venv\Scripts\activate
```

### Step 3: Install Dependencies

Install the required libraries:

```bash
pip install -r requirements.txt
```

If the `requirements.txt` file is not available, install the dependencies manually:

```bash
pip install streamlit opencv-python numpy matplotlib pillow
```

### Step 4: Run the Application

Execute the following command from the project root directory:

```bash
streamlit run app.py
```

Streamlit will start a local web server. Open the URL displayed in the terminal, usually:

```text
http://localhost:8501
```

## 🚀 How to Use

1. Start the application using Streamlit.
2. Upload an image in a supported format.
3. Select an image processing operation from the sidebar.
4. Adjust the available parameters, such as kernel size, CLAHE clip limit, or sharpening strength.
5. Compare the original and processed images.
6. Examine the intensity histograms and image statistics.
7. Download the processed image as a PNG file.

## 🧠 Image Processing Concepts Demonstrated

This project applies several fundamental concepts from Digital Image Processing:

- Grayscale image representation and pixel intensity values
- Contrast enhancement and intensity transformations
- Histogram equalization and adaptive histogram equalization
- Spatial filtering and neighborhood operations
- Gaussian smoothing and median filtering
- Laplacian-based image sharpening
- Unsharp masking
- Image histogram analysis
- Quantitative comparison using mean intensity and standard deviation

## 📊 Results and Evaluation

The application allows visual and statistical comparison between original and processed images.

| Operation | Expected Effect |
|---|---|
| Contrast Stretching | Expands the intensity range |
| Histogram Equalization | Improves global contrast |
| CLAHE | Enhances local contrast |
| Mean Filtering | Smooths images and reduces some noise |
| Gaussian Filtering | Reduces high-frequency noise |
| Median Filtering | Removes many salt-and-pepper noise pixels |
| Laplacian Sharpening | Enhances fine details and edges |
| Unsharp Masking | Increases perceived sharpness |

The actual results depend on the input image and the selected algorithm parameters. Improved appearance does not necessarily mean that every image quality metric will improve.

## 🖼️ Screenshots

Add screenshots of your running Streamlit application here.

For example, create a `screenshots/` folder in the project root and save your screenshots inside it.

```markdown
![Application Dashboard](screenshots/dashboard.png)

![Original and Processed Image](screenshots/image_comparison.png)

![Histogram Comparison](screenshots/histogram_comparison.png)
```

Replace these example paths with the actual screenshot filenames in your repository.

## 🎓 Academic Context

This project was developed as part of the **Image Processing and Applications (IPA)** coursework. It demonstrates practical implementation of classical image processing techniques using Python libraries and an interactive interface.

