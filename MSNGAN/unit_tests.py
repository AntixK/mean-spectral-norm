import numpy as np
import torch
from torch.autograd import Variable
from spectral_norm import spectral_norm
from time import time
import pylab as plt
import torch.nn as nn

def _test_Spectral_Norm():
    torch.manual_seed(354)
    W_t = torch.randn(10,10)
    W_n = W_t.numpy()
    W_t = Variable(W_t)
    start = time()
    U, S,V = np.linalg.svd(W_n)
    numpy_time = time() - start
    start = time()
    sigma, _ = spectral_norm(W_t, Num_iter = 100)
    print(sigma)
    torch_time = time() - start
    print("Max Singular Value (Numpy): {}, Time: {}".format(max(S), numpy_time))
    print("Max Singular Value (Torch): {}, Time: {}".format(sigma,torch_time))

    ## Variation of Spectral Norm with some divided coefficient
    # W_t /= sigma
    # sigma, _ = spectral_norm(W_t, Num_iter=100)
    # print(sigma)
    #
    # s = []
    # x = np.linspace(0.01,10,100)
    # for i in x:
    #     sigma, _ = spectral_norm(W_t/i, Num_iter=100)
    #     s.append(sigma)
    #
    # plt.figure()
    # plt.plot(x,s)
    # plt.show()

if __name__ == '__main__':

    #_test_Spectral_Norm()
    print(torch.ones(1,1))

    p = nn.Parameter(torch.ones(1))
    q = torch.autograd.Variable(torch.ones(1)*3)

    print(p*q)
