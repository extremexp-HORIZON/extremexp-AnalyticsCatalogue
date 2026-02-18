import numpy as np
from scipy.linalg import orth
from scipy.linalg import svd
import torch

def genRandHorizontalMatrix(numRows, numCols):
    # generate random orthonormal matrix to initialize as random weights
    randMat = np.random.normal(0, 1, size=(numRows, numCols))
    # print("generated shape: ", randMat.shape)
    U, s, V = torch.svd(torch.tensor(randMat.reshape(numRows, numCols)))
    # print(" U shape: ", U.shape)
    # print("V shape : ", V.shape)
    return V

def genRandVerticalMatrix(numRows, numCols):
    # generate random orthonormal matrix to initialize as random weights
    randMat = np.random.normal(0, 1, size=(numRows, numCols))
    # print("generated shape: ", randMat.shape)
    U, s, V = torch.svd(torch.tensor(randMat.reshape(numRows, numCols)))
    # print(" U shape: ", U.shape)
    # print("V shape : ", V.shape)
    return U

def genPCAMat(samples):
    """
    Generate 
    """
    numRows, numCols = samples.shape
    U, s, V = svd(torch.tensor(samples.reshape(numRows, numCols)))
    return U

def genScaledRandMat(numRows, numCols, norm=None):
    if norm is None:
        norm = numRows + numCols
    # print("input dimensions: ", numRows, numCols)
    # norm_dim = min(numCols, numRows)
    if numRows <= numCols:
        orthMat = genRandHorizontalMatrix(numRows, numCols)
        coefs = torch.rand(size=(numRows, 1))
    else:
        orthMat = genRandVerticalMatrix(numRows, numCols)
        coefs = torch.rand(size=(1, numCols))

    # print("shape of the orthogonal matrix: ", orthMat.shape)
        
    coefs_sum = sum(coefs)
    # print("coefficients: ", coefs_sum.shape)
    normed_coefs = coefs.T * (norm/coefs_sum[0])
    # print("normed_coefs : ", normed_coefs.shape)

    # print("coeefs : ", normed_coefs.shape)
    if numRows < numCols:
        return normed_coefs * orthMat
    else:
    # temp = orthMat.T * normed_coefs
    # print("temp res ", temp.shape)
        return (normed_coefs * orthMat.T).T