# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 16:40:14 2023

@author: prath
"""

import numpy as np
from keras.datasets import mnist
np.random.seed(0)

def ActivationFunctions(name):
    if(name == 'ReLu'):
        return lambda x: np.maximum(0, x)
    if(name == 'Sigmoid'):
        return lambda x: 1/(1+np.exp(-x))
    print("Activatio Function Not recognised")
    assert False

def ActivationFunctionDerivative(name):
    if(name=='ReLu'):
        return lambda x: 1-(x==0)
    if(name=='Sigmoid'):
        return lambda x: x*(1-x)
    return lambda x: np.ones(np.shape(x))

def LossFunctions(name):
    if(name == 'MSE'):
        return lambda x,y: np.trace(np.matmul(np.transpose((x-y)), (x-y)))
    return lambda x,y: x-y

class NeuralNet:
    def __init__(self, Layers, Activations, LossFunction, learningRate, epochSize):
        assert len(Layers) == len(Activations)+1
        assert len(Layers) > 1
        self.learningRate = learningRate
        self.epochSize = epochSize
        self.layers = Layers
        self.loss = LossFunctions(LossFunction)
        self.Activations = []
        self.ActivationsDer = []
        for act in Activations:
            self.Activations.append(ActivationFunctions(act))
            self.ActivationsDer.append(ActivationFunctionDerivative(act))
        
        self.M = []
        for i in range(len(Layers)-1):
            self.M.append(2 * (np.random.random((Layers[i+1], Layers[i])) - 0.5))
    
    def ForwardProp(self, Inputs):
        if(np.shape(Inputs)[0] != np.shape(self.M[0])[1]):
            print("Input Shape does not match the network")
            assert False
        OUTS = []
        OUTS.append(Inputs)
        for i in range(len(self.M)):
            Inputs = self.Activations[i](np.matmul(self.M[i], Inputs))
            OUTS.append(Inputs)
        return OUTS
    
    def Train(self, Inputs, Test):
        Comp = self.ForwardProp(Inputs)
        MChanges = [[] for x in range(len(self.M))]
        V = np.multiply( (Test - Comp[-1]), self.ActivationsDer[-1](Comp[-1]) )
        MChanges[-1] = -self.learningRate * np.matmul(V, Comp[-2].transpose())
        V = V.transpose()
        
        for i in range(2, len(self.M)+1):
            V = np.multiply( np.matmul(V, self.M[1-i]), self.ActivationsDer[-i](Comp[-i]).transpose() )
            MChanges[-i] = -self.learningRate * np.matmul(Comp[-1-i], V).transpose()
        
        for i in range(len(self.M)):
            self.M[i] -= MChanges[i]
        
    
    def Test(self, Inputs, Test):
        Ans = self.ForwardProp(Inputs)[-1]
        loss = self.loss(Ans, Test)
        Ans = np.argmax(Ans.transpose(), axis = 1)
        Test = np.argmax(Test.transpose(), axis = 1)
        Ans -= Test
        Ans = np.sum(Ans!=0)
        acc = (len(Test.transpose()) - Ans) / len(Test.transpose())
        return (acc, loss)
    
    def TrainEpoch(self, Inputs, Test):
        Inputs = Inputs.transpose()
        Test = Test.transpose()
        Inputs = Inputs[:((len(Inputs)//self.epochSize) * self.epochSize)]
        Inputs = np.split(Inputs, self.epochSize)
        Test = Test[:((len(Test)//self.epochSize) * self.epochSize)]
        Test = np.split(Test, self.epochSize)
        assert len(Inputs) == len(Test)
        
        for i in range(len(Inputs)):
            self.Train(Inputs[i].transpose(), Test[i].transpose())
        
        
        
        
        
        

LEARNING_RATE = 10e-4
EPOCH_SIZE = 2000

if __name__ == "__main__":
    A = NeuralNet((28*28, 128, 32, 16, 10), ('Sigmoid', 'Sigmoid', 'Sigmoid', 'Sigmoid'), 'MSE', LEARNING_RATE, EPOCH_SIZE)
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    train_X = np.reshape(train_X, (len(train_X), 28*28)).transpose().astype('float64')
    test_X = np.reshape(test_X, (len(test_X), 28*28)).transpose().astype('float64')
    
    trainH_Y = np.zeros((len(train_y), 10))
    trainH_Y[np.arange(len(train_y)), train_y] = 1
    testH_Y = np.zeros((len(test_y), 10))
    testH_Y[np.arange(len(test_y)), test_y] = 1
    
    train_X /= 255.0
    test_X /= 255.0
    
    for i in range(200):
        A.TrainEpoch(train_X, trainH_Y.transpose())
        tester = A.Test(test_X, testH_Y.transpose())
        print("LOSS: {0}\tACCURACY: {1}%".format(tester[1], tester[0]*100))
    
    A.ForwardProp(train_X.transpose()[0].transpose())
        
    
    