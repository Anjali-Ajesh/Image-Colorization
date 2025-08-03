# Image-Colorization
A deep learning project that uses a Convolutional Neural Network (CNN) autoencoder to colorize grayscale images. This script is built with Python and TensorFlow/Keras and is trained on the CIFAR-10 dataset.

## Features

-   **Deep Learning for Colorization:** Implements a CNN-based autoencoder, where the encoder learns features from a grayscale image and the decoder reconstructs a plausible color version.
-   **CIFAR-10 Dataset:** Uses the popular CIFAR-10 dataset, which contains 60,000 32x32 color images across 10 classes.
-   **Data Preprocessing:** The script automatically converts the color training images to grayscale to create the input (`X`) for the model, while using the original color images as the target output (`y`).
-   **Model Training & Evaluation:** Trains the autoencoder and then evaluates its performance by visually comparing the colorized outputs against the ground truth.
-   **Visualization:** Displays a set of test images showing the grayscale input, the model's colorized prediction, and the original color image side-by-side.

## Technology Stack

-   **Python**
-   **TensorFlow / Keras:** For building and training the deep learning model.
-   **NumPy:** For numerical operations and data manipulation.
-   **Matplotlib:** For visualizing the image results.
-   **scikit-image:** For converting images between RGB and LAB color spaces, which is often more effective for colorization tasks.

## Setup and Usage

A virtual environment is highly recommended for this project.

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/Anjali-Ajesh/image-colorization.git](https://github.com/Anjali-Ajesh/image-colorization.git)
    cd image-colorization
    ```

2.  **Install Dependencies:**
    ```bash
    # Create and activate a virtual environment
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

    # Install the required libraries
    pip install tensorflow numpy matplotlib scikit-image
    ```

3.  **Run the Script:**
    Execute the Python script from your terminal. TensorFlow will automatically download the CIFAR-10 dataset on the first run.
    ```bash
    python image_colorization.py
    ```
    The script will train the model (this can take some time, especially without a GPU) and then display a plot showing the colorization results on test images.
