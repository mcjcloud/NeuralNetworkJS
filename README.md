# NeuralNetworkJS
NeuralNetworkJS is a simple library to demonstrate supervised learning with an Artificial Neural Network. The library allows you to create a NeuralNetwork object which represents a fully connected network of an arbitrary size.

## How to Install
1. First clone the repository into the node_modules folder of your npm project:

`git clone https://github.com/mcjcloud/NeuralNetworkJS.git`

2. Install the dependencies: `cd NeuralNetworkJS && npm install`

## How to Use
From your code:
``` javascript
const NeuralNetwork = require('NeuralNetworkJS');

var nn = new NeuralNetwork([3, 3, 2]);
```
The input for this constructor is an array of layer sizes. The size of the array is the number of layers (including the input and output layer), and each number in the array is the number of nodes in that layer.

Feedfoward data
``` javascript
// input is an array of numbers which is fed into the input layer
nn.feedforward([0, 1, 1]);
```

Train using back-propogation
``` javascript
// the first array is the input data and the second is the expected output data
nn.train([0, 1, 1], [0.5, 0.5]);
```

