# image_colorization.py

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import cifar10
from skimage.color import rgb2lab, lab2rgb

def run_image_colorization():
    """
    Loads CIFAR-10, builds and trains a CNN autoencoder for colorization,
    and visualizes the results.
    """
    # --- 1. Load and Preprocess Data ---
    print("Loading CIFAR-10 dataset...")
    (x_train, _), (x_test, _) = cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Convert RGB images to LAB color space.
    # The L channel is for lightness (grayscale), and a, b are for color.
    # We will use the L channel as input and predict the a, b channels.
    print("Converting images from RGB to LAB color space...")
    X_train_lab = rgb2lab(x_train)
    X_test_lab = rgb2lab(x_test)

    # Extract the L channel as input (grayscale)
    X_train_L = X_train_lab[:,:,:,0]
    X_test_L = X_test_lab[:,:,:,0]

    # Extract the a and b channels as the target output
    X_train_ab = X_train_lab[:,:,:,1:] / 128 # Normalize a,b channels to [-1, 1]
    X_test_ab = X_test_lab[:,:,:,1:] / 128

    # Reshape L channel to include a channel dimension for the CNN
    X_train_L = X_train_L.reshape(X_train_L.shape + (1,))
    X_test_L = X_test_L.reshape(X_test_L.shape + (1,))

    print(f"Training input shape (L channel): {X_train_L.shape}")
    print(f"Training output shape (a,b channels): {X_train_ab.shape}")

    # --- 2. Build the CNN Autoencoder Model ---
    print("Building the CNN autoencoder model...")
    model = Sequential([
        # Encoder
        Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 1)),
        MaxPooling2D((2, 2), padding='same'),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2), padding='same'),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2), padding='same'),

        # Decoder
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        UpSampling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        UpSampling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        UpSampling2D((2, 2)),
        Conv2D(2, (3, 3), activation='tanh', padding='same') # Output layer predicts 2 channels (a, b)
    ])

    model.compile(optimizer='adam', loss='mse')
    model.summary()

    # --- 3. Train the Model ---
    print("Training the model... (This will take a while)")
    # Note: Training for more epochs (e.g., 20-50) will yield better results.
    # We use 10 epochs here for a quicker demonstration.
    model.fit(X_train_L, X_train_ab, validation_split=0.1, epochs=10, batch_size=64)

    # --- 4. Evaluate and Visualize Results ---
    print("Making predictions on the test set...")
    predicted_ab = model.predict(X_test_L)
    predicted_ab *= 128 # De-normalize a,b channels

    # Combine the original L channel with the predicted a,b channels
    predicted_images = np.zeros(X_test_lab.shape)
    predicted_images[:,:,:,0] = X_test_L.squeeze()
    predicted_images[:,:,:,1:] = predicted_ab

    # Convert the predicted LAB images back to RGB
    predicted_rgb = np.array([lab2rgb(img) for img in predicted_images])

    # --- 5. Display the Results ---
    print("Displaying results...")
    num_images_to_show = 5
    plt.figure(figsize=(12, 6))
    for i in range(num_images_to_show):
        # Grayscale Input
        ax = plt.subplot(3, num_images_to_show, i + 1)
        plt.imshow(X_test_L[i].squeeze(), cmap='gray')
        plt.title("Input")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Predicted Color
        ax = plt.subplot(3, num_images_to_show, i + 1 + num_images_to_show)
        plt.imshow(predicted_rgb[i])
        plt.title("Predicted")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Original Color
        ax = plt.subplot(3, num_images_to_show, i + 1 + 2 * num_images_to_show)
        plt.imshow(x_test[i])
        plt.title("Original")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
    plt.tight_layout()
    plt.savefig('colorization_results.png')
    plt.show()


if __name__ == '__main__':
    run_image_colorization()
