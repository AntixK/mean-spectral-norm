## Mean Spectral Normalization
----------------
This repo contains the code to reproduce the plots for the paper "Mean Spectral Normalization" (Under Review) by Anand Krishnamoorthy Subramanian and Nak-Young Chong, 2019.

#### Main Results
![MSN_sparse](https://github.com/AntixK/mean-spectral-norm/blob/master/plots/Gradient_Sparsity_CIFAR10.png)

*Spectral Normalization essentially induces a high level gradient sparsity in the network which is advatageous for deep neural nets, but dimishes the performance of medium and small neural nets. Our Mean Spectral Normalization (MSN) improves upon Spectral Normalization maintaining the gradient sparsity as improving the performance as close to Batch Norm.*

![MSN Test Accuracy](https://github.com/AntixK/mean-spectral-norm/blob/master/plots/Test_accuracy_comp.png)

*MSN works well for small, medium and alrge networks, comparable to that of Batch Norm, but with lesser number of trainable parameters.*

![MSNGAN on CIFAR10](https://github.com/AntixK/mean-spectral-norm/blob/master/MSNGAN/log/MSNGAN/MSNGAN_results.gif)

*Unsupervised Image Generation using WGAN on CIFAR10*

---------------------------
#### Requirements
The following are required to successfully run the code -
- Python 3.6.8
- Pytorch 1.0.0
- CUDA 9.0.176
- Numpy 1.15.4
- Matplotlib 3.0.2
- Seaborn (optional) 0.9.0
- A minimum of 1 CUDA_capable GPU

**Note:** The above experiments were run on NVIDIA 1080 ti GPU.

--------------------
%#### Citation Information
