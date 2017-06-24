#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.


    Copyright (C)  @author: jose 
    Simple Perceptron for logical gates
    FI UNAM
    Created on Thu Jun 22 20:24:46 2017
"""

from random import choice,random
import numpy as np
import matplotlib.pyplot as plt

def set_data(target):
	x = [np.array([0,0]),
		 np.array([0,1]),
	     np.array([1,0]),
		 np.array([1,1])]
	bias = np.array([1 for _ in range(4)])
	# inputs: [x1 | x0 | bias]
	inputs = np.column_stack((x,bias))
	# return data: [x1 | x0 | bias | target]
	return [(np.array(i),j) for i,j in zip(inputs,target)]

heaviside = lambda x: 1 if x >= 0 else -1

def train(target,w,eta=0.1,epochs=40):
	# Activation function
	errors = []
	#w_tmp = []
	# Updating weights
	for _ in range(epochs):
		x,expected = choice(target)
		result = np.dot(w,x)
		error = expected - heaviside(result)
		errors.append(error)
		w += eta*error*x
	return [w,error]


def predict(inputs,w):
	# inputs: X + bias
	return 1 if np.dot(inputs,w) >= 0 else -1

def run():
	# output for a nand gate
	target = np.array([1,1,1,-1])
	# Random weights
	w = np.array([random() for _ in range(3)])
	print("random weights: {0}".format(w))
	nand = set_data(target)
	w,error = train(nand,w,eta=0.1,epochs=65)
	print("weights updated: {0}".format(w))
	print("Predicting\tAproximation\tResult")
	print("{0}\t\t{1:.5f}\t\t{2}".format([0,0],np.dot([0,0,1],w),predict([0,0,1],w)))
	print("{0}\t\t{1:.5f}\t\t{2}".format([0,1],np.dot([0,1,1],w),predict([0,1,1],w)))
	print("{0}\t\t{1:.5f}\t\t{2}".format([1,0],np.dot([1,0,1],w),predict([1,0,1],w)))
	print("{0}\t\t{1:.5f}\t{2}".format([1,1],np.dot([1,1,1],w),predict([1,1,1],w)))



if __name__ == '__main__':
	run()




