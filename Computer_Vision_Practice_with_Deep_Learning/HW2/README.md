# HW2: Long-Tailed Object Detection for Drone-Based Intelligent Counting

This document provides instructions on how to set up the environment, train the model, and run inference.


## Environment Setup

1.  **Python Version**: Ensure your Python version is `3.10.12`.
2.  **Install Dependencies**: Install all required packages using the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```
3.  **Dataset**: Confirm that both the `train` and `test` dataset folders are located in the project's root directory.


## Dataset Preprocessing

#### Phase 1: Create a folder containing text files with the corresponding ID numbers for the training and validation images.

1.  **Run the command:**
    ```bash
    python3 split.py
    ```
2.  **Process**: This will create a `splits` folder, splitting the original training data into 90% for training (855 images) and 10% for validation (95 images).
3.  **Output**: After `split.py` is complete, two txt files will be saved as `train_ids.txt` and `val_ids.txt`.

#### Phase 2: Create a folder containing the input format required by the YOLO model. 

1.  **Run the command:**
    ```bash
    python3 YOLO_Dataset.py
    ```
2.  **Process**: This will create a `YOLO_Dataset` folder. Using `train_ids.txt` and `val_ids.txt` as inputs, it will correctly copy the corresponding images and convert their label files from the `train` folder into the `YOLO_Dataset` folder based on the contents of the text files.
3.  **Output**: After `YOLO_Dataset.py` is complete, two intermediate folders named `images` and `labels` will be generated. Each of these contains subfolders named `train` and `val`, storing images and their corresponding labels respectively.

#### Phase 3: Confirm the Data Loading Path 
Open the `Dataset.yaml` file to verify that the loading paths for the `train`, `val`, and `test` folders are correct.


## How to Perform Training and Inference in a Single Pass 

1.  **Run the command:**
    ```bash
    python3 HW2.py --mode train
    ```
2.  **Process**: This command trains the model on the 90% training set (855 images) and validates on the 10% validation set (95 images) to record the mAP score per epoch. Finally, it will run predictions on the test dataset (550 images).
3.  **Output**: After the training is complete, the best model weights will be saved as `best.pt` in the `yolo11s/weights` folder, and the predictions will be generated in a file named `submission.csv`.


## How to Run Inference Only

1.  **Run the command:**
    ```bash
    python3 HW2.py --mode inference
    ```
2.  **Process**: This command only loads the best model weights from `yolo11s/weights/best.pt` and runs predictions on the test dataset (550 images).
3.  **Output**: The predictions will be generated in a file named `submission.csv`.











