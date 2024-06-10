import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report

from sklearn.utils.multiclass import type_of_target


def RSE(pred, true):
    return np.sqrt(np.sum((true-pred)**2)) / np.sqrt(np.sum((true-true.mean())**2))

def CORR(pred, true):
    u = ((true-true.mean(0))*(pred-pred.mean(0))).sum(0) 
    d = np.sqrt(((true-true.mean(0))**2*(pred-pred.mean(0))**2).sum(0))
    return (u/d).mean(-1)

def MAE(pred, true):
    return np.mean(np.abs(pred-true))

def MSE(pred, true):
    return np.mean((pred-true)**2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))

def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def REPORT(pred, true):

    probs = torch.nn.functional.softmax(torch.tensor(pred))  # (total_samples, num_classes) est. prob. for each class and sample
    pred = torch.argmax(probs, dim=2).cpu().numpy()  # (total_samples,) int class index for each sample
    return classification_report(true.ravel(), pred.ravel(), digits=5)

def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    report = REPORT(pred, true)
    
    return mae,mse,rmse,mape,mspe,report