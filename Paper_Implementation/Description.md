## PART -1: Autoencoder
### Autoencoder for MNIST Digit Compression and Reconstruction

This project implements a simple **Autoencoder** using **PyTorch** to encode and decode images from the **MNIST dataset**. The model compresses 28x28 grayscale images into a **2D latent space** and reconstructs them back, allowing visualization of the learned feature representations as a way to show the functioning of autoencoder used later in the project.

#### Autoencoders: 
- An autoencoder is a type of neural network used for **unsupervised learning**.
- It consists of two parts:
  - **Encoder**: Compresses the input image into a smaller latent representation.
  - **Decoder**: Reconstructs the image from the latent space.
- The model is trained to minimize reconstruction error, learning key features of the dataset.
- A 2D latent space enables visualization of digit clusters and smooth interpolation between different digits.
---

