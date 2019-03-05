import torch
from torch.nn.modules import conv, Linear
from torch.nn.modules.utils import _pair
import torch.nn.functional as F
import torch.nn as nn

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

class SNConv2d(conv._ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(SNConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)
        self.u = None
        self.renorm = nn.Parameter(torch.ones(1,1).cuda())

    def forward(self, input):
        #print("renorm:",self.renorm.data)
        w_mat = self.weight.view(self.weight.size(0), -1)
        sigma, _u = spectral_norm(w_mat, self.u)
        self.u = _u
        self.weight.data = self.renorm.data * self.weight.data / sigma
        return F.conv2d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class SNLinear(Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(SNLinear, self).__init__(in_features, out_features, bias)
        self.u = None
        self.renorm = nn.Parameter(torch.ones(1,1).cuda())
    def forward(self, input):
        w_mat = self.weight
        sigma, _u = spectral_norm(w_mat, self.u)
        self.u = _u
        self.weight.data = (self.weight.data / sigma) * self.renorm.data
        return F.linear(input, self.weight, self.bias)


class MeanSpectralNorm(nn.BatchNorm2d):
    def forward(self, x):
        self._check_input_dim(x)
        y = x.transpose(0,1)
        return_shape = y.shape
        y = y.contiguous().view(x.size(1), -1)
        mu = y.mean(dim=1)
        #sigma2 = y.var(dim=1)
        if self.training is not True:
            y = y - self.running_mean.view(-1, 1)
            #y = y / (self.running_var.view(-1, 1)**.5 + self.eps)
        else:
            if self.track_running_stats is True:
                with torch.no_grad():
                    self.running_mean = (1-self.momentum)*self.running_mean + self.momentum*mu
                    #self.running_var = (1-self.momentum)*self.running_var + self.momentum*sigma2
            y = y - mu.view(-1,1)
            #y = y / (sigma2.view(-1,1)**.5 + self.eps)

        y = self.weight.view(-1, 1) * y + self.bias.view(-1, 1)
        return y.view(return_shape).transpose(0,1)

# Alternate Definition of MSN (Doen't seem to work properly)
# class MeanSpectralNorm(nn.BatchNorm2d):
#     def forward(self, input):
#         self._check_input_dim(input)

#         exponential_average_factor = 0.0
#         #self.register_parameter('weight', None)
#         self.running_var = self.running_var*1.0

#         if self.training and self.track_running_stats:
#             # TODO: if statement only here to tell the jit to skip emitting this when it is None
#             if self.num_batches_tracked is not None:
#                 self.num_batches_tracked += 1
#                 if self.momentum is None:  # use cumulative moving average
#                     exponential_average_factor = 1.0 / float(self.num_batches_tracked)
#                 else:  # use exponential moving average
#                     exponential_average_factor = self.momentum

#         return F.batch_norm(
#             input, self.running_mean, self.running_var, self.weight*0.0, self.bias,
#             self.training or not self.track_running_stats,
#             exponential_average_factor, self.eps)