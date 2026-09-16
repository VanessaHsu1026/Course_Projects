# HW1: Object Detection for Group-housed Swine

This document provides instructions on how to set up the environment, train the model, and run inference.

## Environment Setup

1.  **Python Version**: Ensure your Python version is `3.10.12`.
2.  **Install Dependencies**: Install all required packages using the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```
3.  **Dataset**: Confirm that both the `train` and `test` dataset folders are located in the project's root directory.

## How to Run Training

The training process is divided into two phases: development and final training.

#### Phase 1: Development Mode

This phase trains the model on a smaller subset of the data and uses a validation set to find the best model checkpoint.

1.  **Run the command:**
    ```bash
    python3 HW1.py --mode dev
    ```
2.  **Process**: This will create a `split_data` folder, splitting the original training data into 80% for training (1013 images) and 20% for validation (253 images).
3.  **Output**: After the training is complete, the best model checkpoint will be saved as `best_model.pth`.

#### Phase 2: Final Training Mode

This phase loads the best checkpoint from Phase 1 and continues training on the entire dataset to produce the final model.

1.  **Run the command:**
    ```bash
    python3 HW1.py --mode final
    ```
2.  **Process**: This command loads the weights from `best_model.pth` and retrains the model on the full training dataset (1266 images).
3.  **Output**: The final, fully trained model will be saved as `final_model.pth`.

## How to Run Inference

This final step uses the trained model to generate predictions for the test set.

1.  **Run the command:**
    ```bash
    python3 HW1.py --mode inference
    ```
2.  **Process**: This command loads the weights from `final_model.pth` and runs predictions on the test dataset (1864 images).
3.  **Output**: The predictions will be saved in a file named `submission.csv`.











