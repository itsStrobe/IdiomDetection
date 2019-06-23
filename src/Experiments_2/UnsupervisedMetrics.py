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
