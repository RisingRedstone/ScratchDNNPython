# Deep Neural Network from Scratch using NumPy

Welcome to the Deep Neural Network (DNN) implementation from scratch using NumPy! This project provides a simple yet flexible neural network that supports multiple activation functions and implements Xavier Weight Initialization to avoid issues like overflow. The neural network is implemented in the `DNNScratch.py` file.

## Mathematics of the Neural Network

### Forward Propagation

The forward propagation in a neural network is performed using the following steps:

1. **Linear Transformation:**
   \[ Z^{[l]} = W^{[l]} \cdot A^{[l-1]} + b^{[l]} \]
   
2. **Activation Function:**
   \[ A^{[l]} = g^{[l]}(Z^{[l]}) \]
   Where \( g^{[l]} \) is the activation function for layer \( l \).

3. **Repeat:**
   Repeat steps 1 and 2 for each layer until the final layer is reached.

### Backward Propagation

Backward propagation involves computing the gradients of the loss with respect to the parameters of the network. The gradients are computed using the chain rule of calculus.

1. **Compute Cost:**
   \[ J = -\frac{1}{m} \sum_{i=1}^{m} \left( Y^{(i)} \log(A^{[L]}) + (1 - Y^{(i)}) \log(1 - A^{[L]}) \right) \]

2. **Backward Pass:**
   - Compute the derivative of the cost with respect to the activation of the output layer.
   - Propagate the gradients backward through each layer.

3. **Update Parameters:**
   Update the parameters using gradient descent or another optimization algorithm.

## Usage

To use the Deep Neural Network, run the following command:

```bash
python3 DNNScratch.py
```

Feel free to explore and modify the code to suit your needs. If you have any questions or suggestions, don't hesitate to reach out!

## Contributing

If you'd like to contribute, please fork the repository and create a pull request. Feel free to open issues for feature requests or bug reports.

Happy deep learning with your scratch-built neural network!
