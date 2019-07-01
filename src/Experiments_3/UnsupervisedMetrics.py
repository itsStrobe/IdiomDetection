"""
    File:   Experiment
    Author: Jose Juan Zavala Iglesias
    Date:   23/06/2019

    Unsupervised Metrics for Idiomatic Detection Experiments.
"""
import operator
import numpy as np

def CosineSimilarity(vecsA, vecsB):
    if(vecsA.shape != vecsB.shape):
        print("Sample vectors have different shapes:", vecsA.shape, vecsB.shape)
        return None

    results = np.zeros(vecsA.shape[0])

    for idx in range(len(results)):
        results[idx] = np.dot(vecsA[idx], vecsB[idx]) / (np.linalg.norm(vecsA[idx]) * np.linalg.norm(vecsB[idx]))

    return results

def UnnamedMetric(cosSims, ovaFixs, beta=0.5):
    if(cosSims.shape != ovaFixs.shape):
        print("Sample vectors have different shapes:", cosSims.shape, ovaFixs.shape)
        return None

    if(beta > 1 or beta < 0):
        print("Parameter beta must be in range 0 <= beta <= 1. Received value:", beta)
        return None

    results = np.zeros(cosSims.shape[0])

    for idx in range(len(results)):
        results[idx] = beta * (1 - cosSims[idx]) + (1 - beta) * ovaFixs[idx]

    return results


def ThresholdClassifier(values, T=0.5, Op='>'):
    classOp = None
    results = np.zeros(values.shape[0], dtype=bool)

    if(Op == '>'):
        classOp = operator.gt
    elif(Op == '>='):
        classOp = operator.ge
    elif(Op == '<'):
        classOp = operator.lt
    elif(Op == '<='):
        classOp = operator.le
    elif(Op == '=='):
        classOp = operator.eq
    elif(Op == '!='):
        classOp = operator.ne
    else:
        print("Invalid Operator:", Op)
        return None

    for idx in range(values.shape[0]):
        results[idx] = classOp(values[idx], T)

    return results
