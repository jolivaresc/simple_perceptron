# Simple Perceptron

This code is the implementation of a perceptron that simulates a NAND gate. In the ``run()`` function, the list ``target`` can be modified to represent another logical gate (has to be Linearly Separable, hence, doesn't works to simulate XOR gate).

![img1](http://www.ece.utep.edu/research/webfuzzy/docs/kk-thesis/kk-thesis-html/img40.gif "Linearly Separable example")
## 
**Note** In the list ``target`` to represent a 0 a -1 should be used.

## Dependencies
* python3
* numpy
* random
* matplotlib

## Usage

Run ``python3 perceptron.py`` to see the results:

        ```
        initial random weights: [ 0.72451334  0.12365737  0.72032936]
        weights updated: [-0.07548666 -0.27634263  0.32032936]
        Predicting [0, 0] -> 1
        Predicting [0, 1] -> 1
        Predicting [1, 0] -> 1
        Predicting [1, 1] -> -1
        ```
