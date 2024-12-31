# Deep Generative Image Modeling: Final Project

In this project, we show the de-noising process on hand X-ray images from MedNIST dataset using DDPM. Only a few timesteps of DDPM are needed for de-noising a noisy image.

## Requirements
```
python==3.10.12
pytorch==2.1.1
torchvision==0.16.1
numpy==1.12.4
scikit-image==24.2
pillow=10.4.0
```

## Run De-noising

- The project only contains one code file: ```Term_Project_Code.ipynb```.
The easiest way to  re-produce the experiment is to run all the cell in the jupyter notebook.

- The list below shows the composition of the code.
    ```
    |--- Train DDPM
        |--- Setup Environment
        |--- Setup Imports
        |--- Setup Datasets
        |--- Model Training
        |--- Visualize Generated Images
    |--- Experiment on De-noising
        |--- Setup Helper Functions
        |--- Sampling for Whole Validation Set
        |--- Visualize the Results
    ```
