
# Fashion-MNIST Neural Network

This repository contains a simple implementation of a neural network using PyTorch to classify the **Fashion-MNIST** dataset. The model is trained and tested to classify images of 10 different fashion categories. The neural network consists of 3 fully connected layers, and it uses the **Adam** optimizer and **Cross-Entropy Loss** for training.

## Requirements

- Python 3.x
- PyTorch
- torchvision
- numpy

You can install the required libraries using pip:

```bash
pip install torch torchvision numpy
```

## Dataset

The **Fashion-MNIST** dataset is a collection of 60,000 28x28 grayscale images for training and 10,000 images for testing. It consists of 10 different fashion categories such as T-shirts, trousers, shoes, bags, etc. 

The dataset is downloaded automatically when the script runs for the first time.

## Model Overview

This repository implements a simple feedforward neural network with the following architecture:
- **Input Layer**: 784 nodes (28x28 images flattened)
- **Hidden Layer 1**: 300 nodes
- **Hidden Layer 2**: 100 nodes
- **Output Layer**: 10 nodes (representing 10 classes in Fashion-MNIST)

### Training Process:
1. **Forward Pass**: The input image is passed through the model to generate predictions.
2. **Loss Calculation**: The loss function (Cross-Entropy Loss) calculates how far the predictions are from the true labels.
3. **Backward Pass**: Gradients are computed using `loss.backward()`.
4. **Parameter Update**: The optimizer (Adam) updates the weights based on the gradients.

### Optimizer and Loss:
- Optimizer: **Adam**
- Loss Function: **Cross-Entropy Loss**

## How to Use

### Clone the Repository
To get started, clone the repository:

```bash
git clone https://github.com/Abdulraheem232/Fashion-Mnist-Neural-Network.git
cd Fashion-Mnist-Neural-Network
```

### Training the Model
To train the model, run the following script:

```bash
python train.py
```

The script will:
- Download the Fashion-MNIST dataset.
- Train the neural network on the training data for a specified number of epochs.
- Print the loss and accuracy during training.

### Testing the Model
The model is evaluated on the test set after training, and the loss and accuracy on the test data will be displayed.

## Results

During training, the model will output:
- **Loss**: The error between predictions and true labels.
- **Accuracy**: The percentage of correctly classified images.

After training is complete, the model will be tested on unseen data (test set), and loss and accuracy will be printed.

## License

This project is licensed under the MIT License.

## Acknowledgments

- The Fashion-MNIST dataset is provided by Zalando Research.
