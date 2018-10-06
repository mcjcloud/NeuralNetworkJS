/*
 * NeuralNetwork.js - an object that handles the creation and training of a neural network
 */

// init the neural network, structure should be an array where structure.length is the number of layers (including input/output)
// and structure[i] is the number of nodes in layer i. The network will be fully connected.
function NeuralNetwork(structure) {
    this.lr = 0.1;
    this.weights = [];  // this is an array of the matrices for each layer in the network
    this.biases = [];   // an array of bias matrices

    if (structure instanceof Array) {
        // build the matrices
        for (let layer = 1; layer < structure.length; layer++) {
            // create a matrix for each layer
            const numRows = structure[layer];
            const numCols = structure[layer - 1];
            let wts = [];   // temp weight 2x2 array
            let bis = [];   // temp bias array
            for (let i = 0; i < numRows; i++) {
                wts[i] = [];   // add a row
                for (let j = 0; j < numCols; j++) {
                    // add a column
                    wts[i][j] = Math.random();
                }
                // create the bias
                bis.push(Math.random());
            }

            // add the weight and bias matrix to matrices and biases
            
            let bias = math.matrix([bis]);
            bias = math.transpose(bias);
            this.weights.push( math.matrix(wts) );
            this.biases.push( math.matrix(bias) );
        }
    }
    // if a predefined network was passed in
    else {
        for (let weight of structure.weights) {
            this.weights.push( math.matrix(weight) );
        }
        for (let bias of structure.biases) {
            this.biases.push( math.matrix(bias) );
        }
    }
}

NeuralNetwork.prototype.feedforward = function(inputs) {
    // convert the inputs to a matrix
    if (inputs instanceof Array) {
        inputs = math.matrix([inputs]);
        inputs = math.transpose(inputs);
    }
    else {
        console.error('feedforward must take argument as array.');
        return;
    }

    // calculate the output of the first layer
    let output = math.multiply(this.weights[0], inputs);
    output = math.add(this.biases[0], output);
    // apply activation function
    output = math.map(output, sigmoid);

    // loop through each remaining layer of the network and calculate its output
    for (let i = 1; i < this.weights.length; i++) {
        // use the output as the input to the next layer
        output = math.multiply(this.weights[i], output);
        output = math.add(this.biases[i], output);
        // activation function
        output = math.map(output, sigmoid);
    }
    return output._data;
}

NeuralNetwork.prototype.train = function(inputs, targets) {
    // convert the inputs to a matrix
    if (inputs instanceof Array && targets instanceof Array) {
        inputs = math.matrix([inputs]);
        inputs = math.transpose(inputs);    // convert to column
        targets = math.matrix([targets]);
        targets = math.transpose(targets);  // convert to column
    }
    else {
        console.error('train must take arguments as array.');
        return;
    }

    // calculate the output of the first layer
    let outputs = [math.multiply(this.weights[0], inputs)];
    outputs[0] = math.add(this.biases[0], outputs[0]);
    // apply activation function
    outputs[0] = math.map(outputs[0], sigmoid);
    

    // loop through each remaining layer of the network and calculate its output
    for (let i = 1; i < this.weights.length; i++) {
        // use the output as the input to the next layer
        outputs.push(math.multiply(this.weights[i], outputs[i - 1]));
        outputs[i] = math.add(this.biases[i], outputs[i]);
        // activation function
        outputs[i] = math.map(outputs[i], sigmoid);
    }

    // get a matrix of the error of the entire network
    let outputError = math.subtract(targets, outputs[outputs.length - 1]);
    // TRAIN - start at the last layer, this will be the output of the last hidden layer, end before the first layer
    // calculate the gradient
    let gradients = math.map(outputs[outputs.length - 1], dSigmoid);
    // multiply by output errors and learning rate
    gradients = math.dotMultiply(outputError, gradients);
    gradients = math.dotMultiply(gradients, this.lr);
    for (let i = outputs.length - 1; i > 0; i--) {
        // calculate how much the weights should change by
        let transposedPrevOuput = math.transpose( math.matrix(outputs[i - 1]) );
        let deltas = math.multiply(gradients, transposedPrevOuput);

        // adjust weights at i + 1, since the output array is offset from the weight array
        this.weights[i] = math.add(this.weights[i], deltas);
        this.biases[i] = math.add(this.biases[i], gradients);

        // calculate next error (for prev layer)
        let transposedOutput = math.transpose(this.weights[i]);
        outputError = math.multiply(transposedOutput, outputError);

        // calculate the gradient for the next loop
        gradients = math.map(outputs[i - 1], dSigmoid);
        
        gradients = math.dotMultiply(gradients, outputError);
        gradients = math.dotMultiply(gradients, this.lr);
    }

    // do last layer using input
    let transposedInputs = math.transpose(inputs);
    let deltas = math.multiply(gradients, transposedInputs);

    // adjust input weights and biases
    this.weights[0] = math.add(this.weights[0], deltas);
    this.biases[0] = math.add(this.biases[0], gradients);

    // TODO: figure out what to return
    // return outputs[outputs.length - 1]._data;
}

NeuralNetwork.prototype.mutate = function(prob) {
    // loop through weights and randomly mutate
    for (let i = 0; i < this.weights.length; i++) {
        this.weights[i] = math.map(this.weights[i], (weight) => {
            const rand = Math.random();
            if (prob <= rand) {
                return Math.random();
            }
            else {
                return weight;
            }
        });
    }

    // loop through biases and randomly mutate
    for (let i = 0; i < this.biases.length; i++) {
        this.biases[i] = math.map(this.biases[i], (bias) => {
            const rand = Math.random();
            if (prob <= rand) {
                return Math.random();
            }
            else {
                return bias;
            }
        });
    }
}

function sigmoid(x) {
    return Math.exp(x) / (Math.exp(x) + 1);
}
// the d/s(x). (with the original sigmoids removed)
function dSigmoid(x) {
    return x * (1 - x);
}

module.exports = NeuralNetwork;
