import torch
import torch.nn.functional as F

def _L2Norm(v, eps=1e-12):
    return v/(torch.norm(v) + eps)

def spectral_norm(W, u=None, Num_iter=100):
    '''
    Spectral Norm of a Matrix is its maximum singular value.
    This function employs the Power iteration procedure to
    compute the maximum singular value.

    :param W: Input(weight) matrix - autograd.variable
    :param u: Some initial random vector - FloatTensor
    :param Num_iter: Number of Power Iterations
    :return: Spectral Norm of W, orthogonal vector _u
    '''
    if not Num_iter >= 1:
        raise ValueError("Power iteration must be a positive integer")
    if u is None:
        u = torch.FloatTensor(1, W.size(0)).normal_(0,1).cuda()
    _u = u
    for _ in range(Num_iter):
        _v = _L2Norm(torch.matmul(_u, W.data))
        _u = _L2Norm(torch.matmul(_v, torch.transpose(W.data,0, 1)))
    sigma = torch.sum(F.linear(_u, torch.transpose(W.data, 0,1)) * _v)
    return sigma, _u
