# Neural Network using PyTorch - Cancer Prediction

This repository contains code for a neural network model implemented using PyTorch for cancer prediction.

## Dataset
The dataset used for training and testing the neural network is not provided in this repository. However, you can use any cancer dataset of your choice, ensuring it is appropriately preprocessed and formatted for input into the neural network.

## Requirements
- Python 3.x
- PyTorch
- NumPy
- Pandas
- Matplotlib (optional, for visualization)

## Usage
1. **Data Preparation**: Prepare your dataset and preprocess it as required. Ensure the dataset is split into training and testing sets.

2. **Install Dependencies**: Install the necessary dependencies using `pip`:

    ```
    pip install torch numpy pandas matplotlib
    ```

3. **Model Training**: Run the `train.py` script to train the neural network model:

    ```
    python train.py
    ```

4. **Model Evaluation**: After training, you can evaluate the trained model using the `test.py` script:

    ```
    python test.py
    ```

5. **Prediction**: Once the model is trained, you can use it for making predictions on new data by calling the `predict` function defined in `predict.py`.

## Model Architecture
The neural network model architecture used for cancer prediction is defined in `model.py`. You can modify this file to experiment with different architectures, layers, and hyperparameters.

## Results
The results of training and testing the neural network model 96% and 97%.
