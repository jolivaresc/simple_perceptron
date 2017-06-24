# Simple Perceptron

This code is the implementation of a perceptron that simulates a NAND gate. In the ``run()`` function, the list ``target`` can be modified to represent another logical gate (has to be Linearly Separable, hence, doesn't works to simulate XOR gate).
**Note** In the list ``target`` 0 is represented by -1.

## Example of Linearly Separable Pattern
![img1](http://www.ece.utep.edu/research/webfuzzy/docs/kk-thesis/kk-thesis-html/img40.gif "Linearly Separable example")

## Dependencies
* python 3.
* numpy.
* random.
* matplotlib. **(optional)**

## Usage

Run ``python3 perceptron.py`` to see the results:

        ```
        random weights: [ 0.94734709  0.77761066  0.92809446]
        weights updated: [-0.25265291 -0.22238934  0.32809446]
        Predicting	Aproximation	Result
        [0, 0]		0.32809		1
        [0, 1]		0.10571		1
        [1, 0]		0.07544		1
        [1, 1]		-0.14695	-1
        ```
