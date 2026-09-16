# HW3: Image Generation for Handwritten Digits

This document provides instructions on how to set up the environment, train the model, generate the 10,000 images, visualize the diffusion process, and calculate the FID score.


## Environment Setup

1.  **Python Version**: Ensure your Python version is `3.10.12`.
2.  **Install Dependencies**: Install all required packages using the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```
3.  **Dataset**: Confirm that the mnist.zip dataset file is located in the project's root directory. The training script will handle the necessary data extraction.


## Train the DDPM Model

1.  **Run the command:**
    ```bash
    python3 train.py
    ```
2.  **Process**: This will create a `checkpoints` folder that saves the model weights after every 5 epochs.
3.  **Output**: After `train.py` is complete, the best model weights will be saved as `final_model.pth`.


## Generate the 10,000 Images

1.  **Run the command:**
    ```bash
    python3 generate.py
    ```
2.  **Process**: This command only loads the best model weights from `checkpoints/final_model.pth` and run the inference.
3.  **Output**: After the inference is complete, the 10,000 images will be generated in a folder named `generated_images`.


## Visualize the Diffusion Process

1.  **Run the command:**
    ```bash
    python3 visualize.py
    ```
2.  **Process**: This command records the diffusion results every 142 timesteps and processes 8 different samples, then arranges them into an 8x8 grid.
3.  **Output**: The grid will be saved as `diffusion_process.png`.


## Calculate the FID Score
1.  **Run the command:**
    ```bash
    python3 -m pytorch_fid generated_images mnist
    ```
2.  **Process**: This command will calculate the FID score between 10,000 generated images and the training data from the MNIST folder. 
3.  **Output**: The FID score will be displayed on the terminal.












