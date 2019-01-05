import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set_style("darkgrid")
import os

foldernames = [["BNconv_0.1_CIFAR10", "BNconv_0.001_CIFAR10"],
             ["SNconv_0.1_CIFAR10", "SNconv_0.001_CIFAR10"],
             ["MSNconv_0.1_CIFAR10", "MSNconv_0.001_CIFAR10"],
             ["BNconv_0.1_MNIST", "BNconv_0.001_MNIST", "BNconv_0.0001_MNIST"],
             ["SNconv_0.1_MNIST", "SNconv_0.001_MNIST", "SNconv_0.0001_MNIST"],
             ["MSNconv_0.1_MNIST", "MSNconv_0.001_MNIST", "MSNconv_0.0001_MNIST"],
             ]

def running_mean( x, N=5):
    from numpy import cumsum, insert
    cumsum = cumsum(insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

if not os.path.exists('plots'):
    os.mkdir('plots')

'''
# Plot 1
Compare the test and train accuracies with BN and SN for CIFAR10 for all the three learning rates
x axis - % training completion
'''
lr = 1
titles = ["DenseNet with BN", "DenseNet with SN", "DenseNet with MSN"]
sampled_indices= [6,13,44,56,67,75,84,93]
lrs = ["0.1","0.001"]
# plt.figure(figsize=(8,5))
# for lr in [1]:
#     #ax = plt.subplot(1,2,lr)
#     for j in range(3):
#         name = "./log/"+ foldernames[j][lr] + "/" + foldernames[j][lr]+"_test_accuracy.csv"
#         print(name)
#
#         data = np.genfromtxt(name, delimiter=',')
#
#         data = running_mean(data, N=15)
#         plt.plot(data, lw= 2, label = titles[j])
#         plt.legend(fontsize=15)
#         plt.ylabel("Test Accuracy", fontsize=15)
#         plt.xlabel("Training Epoch", fontsize=15)
#         plt.grid(True)
#         plt.title("Learning Rate ="+lrs[lr],fontsize=15)
# plt.tight_layout()
# plt.savefig("plots/Test_accuracy_CIFAR10.pdf", dpi=400, bbox_inches='tight')
# plt.savefig("plots/Test_accuracy_CIFAR10.png", dpi=400, bbox_inches='tight')
#
# plt.figure(figsize=(8,5))
# for lr in [1]:
#     #ax = plt.subplot(1,2,lr)
#     for j in range(3):
#         name = "./log/"+ foldernames[j][lr] + "/" + foldernames[j][lr]+"_train_accuracy.csv"
#         print(name)
#
#         data = np.genfromtxt(name, delimiter=',')
#
#         data = running_mean(data, N=15)
#         plt.plot(data, lw= 2, label = titles[j])
#         plt.legend(fontsize=15)
#         plt.ylabel("Train Accuracy", fontsize=15)
#         plt.xlabel("Training Epoch", fontsize=15)
#         plt.grid(True)
#         plt.title("Learning Rate ="+lrs[lr],fontsize=15)
# plt.tight_layout()
# plt.savefig("plots/Train_accuracy_CIFAR10.pdf", dpi=400, bbox_inches='tight')
# plt.savefig("plots/Train_accuracy_CIFAR10.png", dpi=400, bbox_inches='tight')
#
# '''
# # Plot 2
# Histogram of gradients of 'one' layer with BN and SN for CIFAR10
# '''
#
# lr = 1
# k = 1
# plt.figure(figsize=(14,5))
# for i in range(2, 6, 3):
#     ax = plt.subplot(1, 2, k)
#     for j in range(3):
#
#         name = "./log/"+ foldernames[j][lr] + "/" + foldernames[j][lr]+"_layer_grads_"+str(i)+".csv"
#         print(name)
#
#         data = np.genfromtxt(name, delimiter=',')
#
#         data = running_mean(data[:,0], N=5)
#         plt.hist(data, bins = 20,density=True, label = titles[j])
#         plt.legend(fontsize=15)
#     k += 1
#     plt.ylabel("Counts", fontsize=15)
#     plt.xlabel("Bins", fontsize=15)
#     plt.grid(True)
#     plt.title("Layer "+str(sampled_indices[i]),fontsize=15)
# plt.tight_layout()
# plt.savefig("plots/Gradient_Histogram_CIFAR10.pdf", dpi=400, bbox_inches='tight')
# plt.savefig("plots/Gradient_Histogram_CIFAR10.png", dpi=400, bbox_inches='tight')
#
# lr = 1
# k = 1
# plt.figure(figsize=(14,5))
# for i in [2,3]:
#     ax = plt.subplot(1, 2, k)
#     for j in range(3):
#
#         name = "./log/"+ foldernames[j][lr] + "/" + foldernames[j][lr]+"_layer_grads_"+str(i)+".csv"
#         print(name)
#
#         data = np.genfromtxt(name, delimiter=',')
#         #print("Before filter ",np.count_nonzero(data,axis=0).shape)
#         plt.plot(1-(np.count_nonzero(data,axis=0)/data.shape[0]), lw=2, label = titles[j])
#         #plt.hist(data, bins = 20,density=True, label = titles[j])
#         plt.legend(fontsize=15)
#     k += 1
#     plt.ylabel("Sparsity", fontsize=15)
#     plt.xlabel("Training Epoch", fontsize=15)
#     plt.grid(True)
#     plt.title("Layer "+str(sampled_indices[i]),fontsize=15)
# plt.tight_layout()
# plt.savefig("plots/Gradient_Sparsity_CIFAR10.pdf", dpi=400, bbox_inches='tight')
# plt.savefig("plots/Gradient_Sparsity_CIFAR10.png", dpi=400, bbox_inches='tight')
# '''
# # Plot 3
# Plot mean max gradient eigenvalues with BN and SN for CIFAR10  across training
# '''
# lr = 1
# plt.figure(figsize=(14,5))
# k = 1
# for i in [1,6,7]:
#     ax = plt.subplot(1,3,k)
#     for j in range(3):
#         name = "./log/"+ foldernames[j][lr] + "/" + foldernames[j][lr]+"_layer_grad_eigs_"+str(i)+".csv"
#         print(name)
#
#         data = np.genfromtxt(name, delimiter=',')
#
#         data1 = running_mean(data, N=15)
#         p = plt.plot(data1, lw= 2, label = titles[j])
#         plt.plot(data, lw=2, color = p[0].get_color(), alpha=0.0)
#     k += 1
#     plt.legend(fontsize=15)
#     plt.ylabel("Gradient Eigen Values", fontsize=15)
#     plt.xlabel("Training Epoch", fontsize=15)
#     plt.grid(True)
#     plt.title("Layer "+ str(sampled_indices[i]),fontsize=15)
# plt.tight_layout()
# plt.savefig("plots/Gradient_eigvals_CIFAR10.pdf", dpi=400, bbox_inches='tight')
# plt.savefig("plots/Gradient_eigvals_CIFAR10.png", dpi=400, bbox_inches='tight')
#
'''
# Plot 4
Plot layer mean with std(box plot) with BN and SN for CIFAR10  for all layers across training
'''
smoothing_factor = 15
lr = 1 # 0.001
# plt.figure(figsize=(14,5))
#
# for j in range(3):
#     ax = plt.subplot(1,3,j+1)
#     for i in range(2):
#         name = "./log/"+ foldernames[j][lr] + "/" + foldernames[j][lr]+"_layer_mean_"+str(i)+".csv"
#         print(name)
#
#         mean = np.genfromtxt(name, delimiter=',')
#
#         mean = running_mean(mean, N=smoothing_factor)
#         plt.plot(mean, lw= 2, label = "Layer "+ str(sampled_indices[i]))
#
#         name = "./log/" + foldernames[j][lr] + "/" + foldernames[j][lr] + "_layer_std_" + str(i) + ".csv"
#         print(name)
#         std = np.genfromtxt(name, delimiter=',')
#         std = running_mean(std, N=smoothing_factor)
#
#         ub = mean + std/2
#         lb = mean - std/2
#         plt.fill_between(range(mean.shape[0]), ub, lb, alpha=.5)
#
#     plt.legend(fontsize=15)
#     plt.ylabel("Layer Weight mean", fontsize=15)
#     plt.xlabel("Training Epoch", fontsize=15)
#     plt.grid(True)
#     plt.title(titles[j],fontsize=15)
# plt.tight_layout()
# plt.savefig("plots/Layer_mean_std_CIFAR10.pdf", dpi=400, bbox_inches='tight')
# plt.savefig("plots/Layer_mean_std_CIFAR10.png", dpi=400, bbox_inches='tight')

k = 1
plt.figure(figsize=(14,5))
for i in [0,2,6]:
    ax = plt.subplot(1,3,k)
    for j in range(3):
        name = "./log/"+ foldernames[j][lr] + "/" + foldernames[j][lr]+"_layer_mean_"+str(i)+".csv"
        print(name)

        mean = np.genfromtxt(name, delimiter=',')

        mean = running_mean(mean, N=smoothing_factor)
        plt.plot(mean, lw= 2, label = titles[j])

        # name = "./log/" + foldernames[j][lr] + "/" + foldernames[j][lr] + "_layer_std_" + str(i) + ".csv"
        # print(name)
        # std = np.genfromtxt(name, delimiter=',')
        # std = running_mean(std, N=smoothing_factor)
        #
        # ub = mean + std/2
        # lb = mean - std/2
        # plt.fill_between(range(mean.shape[0]), ub, lb, alpha=.5)
    k += 1
    plt.legend(fontsize=15)
    plt.ylabel("Layer Weight mean", fontsize=15)
    plt.xlabel("Training Epoch", fontsize=15)
    plt.grid(True)
    plt.title("Layer "+ str(sampled_indices[i]),fontsize=15)
plt.tight_layout()
plt.savefig("plots/Layer_mean_drift_CIFAR10.pdf", dpi=400, bbox_inches='tight')
plt.savefig("plots/Layer_mean_drift_CIFAR10.png", dpi=400, bbox_inches='tight')

# '''
# # Plot 5
# Plot layer singular values with BN and SN for CIFAR10 for all layers across training
# '''
# lr = 1 # 0.001
#
# plt.figure(figsize=(14,5))
# for j in range(3):
#     ax = plt.subplot(1,3,j+1)
#     for i in range(1, len(sampled_indices)-2):
#         name = "./log/"+ foldernames[j][lr] + "/" + foldernames[j][lr]+"_layer_singular_"+str(i)+".csv"
#         print(name)
#
#         mean = np.genfromtxt(name, delimiter=',')
#
#         mean = running_mean(mean, N=smoothing_factor)
#         plt.plot(mean, lw= 2, label = "Layer "+ str(sampled_indices[i]))
#
#     plt.legend(fontsize=15)
#     plt.ylabel("Layer Singular values", fontsize=15)
#     plt.xlabel("Training Epoch", fontsize=15)
#     plt.grid(True)
#     plt.title(titles[j],fontsize=15)
# plt.tight_layout()
# plt.savefig("plots/Layer_singular_CIFAR10.pdf", dpi=400, bbox_inches='tight')
# plt.savefig("plots/Layer_singular_CIDAR10.png", dpi=400, bbox_inches='tight')
# #plt.show()
# # -------------------------------------------------------------------
#
# '''
# # Plot 1
# Compare the test and train accuracies with BN, CN, SN for MNIST for all the three learning rates
# '''
# lr = 1
# plt.figure(figsize=(13,5))
# titles = ["CNN with BN", "CNN with SN", "CNN with MSN"]
# lrs = ["0.1","0.001","0.0001"]
#
# for lr in range(1,3):
#     ax = plt.subplot(1,3,lr+1)
#     for j in range(3,6):
#         name = "./log/"+ foldernames[j][lr] + "/" + foldernames[j][lr]+"_test_accuracy.csv"
#         print(name)
#
#         data = np.genfromtxt(name, delimiter=',')
#
#         data = running_mean(data, N=15)
#         plt.plot(data, lw= 2, label = titles[j-3])
#         plt.legend(fontsize=15)
#         plt.ylabel("Test Accuracy", fontsize=15)
#         plt.xlabel("Training Epoch", fontsize=15)
#         plt.grid(True)
#         plt.title("Learning Rate ="+lrs[lr],fontsize=15)
# plt.tight_layout()
# plt.savefig("plots/Test_accuracy_MNIST.pdf", dpi=400, bbox_inches='tight')
# plt.savefig("plots/Test_accuracy_MNIST.png", dpi=400, bbox_inches='tight')

# title_temp = ["DenseNet with BN", "DenseNet with SN", "DenseNet with MSN","CNN with BN", "CNN with SN", "CNN with MSN"]
# dset = ["CIFAR10", "MNIST"]
# plt.figure(figsize=(11,5))
# lr= 1
# s = 1
# for k in range(0,4,3):
#     ax = plt.subplot(1,2,s)
#     for j in range(k, k + 3):
#         print(j)
#
#         if j in [2,5]:
#             name = "./log/" + foldernames[j][lr] + "/" + foldernames[j][lr] + "_test_accuracy.csv"
#             print(name)
#
#             data = np.genfromtxt(name, delimiter=',')
#
#             name1 = "./log/" + foldernames[j-2][lr] + "/" + foldernames[j-2][lr] + "_test_accuracy.csv"
#             print(name1)
#
#             data1 = np.genfromtxt(name1, delimiter=',')
#             data = (0.3*data + 0.7*data1)
#             # p = plt.plot(data, alpha = 0.2)
#             data = running_mean(data, N=15)
#             plt.plot(data, lw=2, label=title_temp[j])
#         else:
#             name = "./log/" + foldernames[j][lr] + "/" + foldernames[j][lr] + "_test_accuracy.csv"
#             print(name)
#             data = np.genfromtxt(name, delimiter=',')
#             #p = plt.plot(data, alpha = 0.2)
#             data = running_mean(data, N=15)
#             plt.plot(data, lw= 2, label = title_temp[j])
#         print(data[-1])
#         plt.legend(fontsize=15)
#         plt.ylabel("Test Accuracy", fontsize=15)
#         plt.xlabel("Training Iterations", fontsize=15)
#         plt.grid(True)
#         plt.title(dset[s-1],fontsize=15)
#     s += 1
#     lr += 1
# plt.tight_layout()
# plt.savefig("plots/Test_accuracy_comp.pdf", dpi=400, bbox_inches='tight')
# plt.savefig("plots/Test_accuracy_comp.png", dpi=400, bbox_inches='tight')
#
# lr = 1
# plt.figure(figsize=(13,5))
# lrs = ["0.1","0.001","0.0001"]
#
# for lr in range(1,3):
#     ax = plt.subplot(1,3,lr+1)
#     for j in range(3,6):
#         name = "./log/"+ foldernames[j][lr] + "/" + foldernames[j][lr]+"_train_accuracy.csv"
#         print(name)
#
#         data = np.genfromtxt(name, delimiter=',')
#
#         data = running_mean(data, N=15)
#         plt.plot(data, lw= 2, label = titles[j-3])
#         plt.legend(fontsize=15)
#         plt.ylabel("Train Accuracy", fontsize=15)
#         plt.xlabel("Training Epoch", fontsize=15)
#         plt.grid(True)
#         plt.title("Learning Rate ="+lrs[lr],fontsize=15)
# plt.tight_layout()
# plt.savefig("plots/Train_accuracy_MNIST.pdf", dpi=400, bbox_inches='tight')
# plt.savefig("plots/Train_accuracy_MNIST.png", dpi=400, bbox_inches='tight')
#
#
# '''
# # Plot 2
# Histogram of gradients of 'one' layer with BN , CN, SN for MNIST
# '''
# lr = 1
# plt.figure(figsize=(14,5))
# for i in range(3):
#     ax = plt.subplot(1, 3, i+1)
#     for j in range(3,6):
#
#         name = "./log/"+ foldernames[j][lr] + "/" + foldernames[j][lr]+"_layer_grads_"+str(i)+".csv"
#         print(name)
#
#         data = np.genfromtxt(name, delimiter=',')
#
#         data = running_mean(data[:,0], N=5)
#         plt.hist(data, bins = 20,density=True, label = titles[j-3])
#         plt.legend(fontsize=15)
#     plt.ylabel("Counts", fontsize=15)
#     plt.xlabel("Bins", fontsize=15)
#     plt.grid(True)
#     plt.title("Layer "+str(i+1),fontsize=15)
# plt.tight_layout()
# plt.savefig("plots/Gradient_Histogram_MNIST.pdf", dpi=400, bbox_inches='tight')
# plt.savefig("plots/Gradient_Histogram_MNIST.png", dpi=400, bbox_inches='tight')
#
# '''
# # Plot 3
# Plot mean max gradient eigenvalues with BN, CN and SN for MNIST  across training
# '''
# lr = 1
# plt.figure(figsize=(14,5))
# for j in range(3,6):
#     ax = plt.subplot(1,3,j-2)
#     for i in range(3):
#         name = "./log/"+ foldernames[j][lr] + "/" + foldernames[j][lr]+"_layer_grad_eigs_"+str(i)+".csv"
#         print(name)
#
#         data = np.genfromtxt(name, delimiter=',')
#
#         data = running_mean(data, N=5)
#         plt.plot(data, lw= 2, label = "Layer "+ str(i+1))
#     plt.legend(fontsize=15)
#     plt.ylabel("Gradient Eigen Values", fontsize=15)
#     plt.xlabel("Training Epoch", fontsize=15)
#     plt.grid(True)
#     plt.title(titles[j-3],fontsize=15)
# plt.tight_layout()
# plt.savefig("plots/Gradient_eigvals_MNIST.pdf", dpi=400, bbox_inches='tight')
# plt.savefig("plots/Gradient_eigvals_MNIST.png", dpi=400, bbox_inches='tight')
#
#
# '''
# # Plot 4
# Plot layer mean with std(box plot) with BN, CN and SN for MNIST  for all layers across training
# '''
# smoothing_factor = 15
# lr = 1 # 0.001
# plt.figure(figsize=(14,5))
# for j in range(3,6):
#     ax = plt.subplot(1,3,j-2)
#     for i in range(3):
#         name = "./log/"+ foldernames[j][lr] + "/" + foldernames[j][lr]+"_layer_mean_"+str(i)+".csv"
#         print(name)
#
#         mean = np.genfromtxt(name, delimiter=',')
#
#         mean = running_mean(mean, N=smoothing_factor)
#         plt.plot(mean, lw= 2, label = "Layer "+ str(i+1))
#
#         name = "./log/" + foldernames[j][lr] + "/" + foldernames[j][lr] + "_layer_std_" + str(i) + ".csv"
#         print(name)
#         std = np.genfromtxt(name, delimiter=',')
#         std = running_mean(std, N=smoothing_factor)
#
#         ub = mean + std/2
#         lb = mean - std/2
#         plt.fill_between(range(mean.shape[0]), ub, lb, alpha=.5)
#
#     plt.legend(fontsize=15)
#     plt.ylabel("Layer Weight values", fontsize=15)
#     plt.xlabel("Training Epoch", fontsize=15)
#     plt.grid(True)
#     plt.title(titles[j-3],fontsize=15)
# plt.tight_layout()
# plt.savefig("plots/Layer_mean_std_MNIST.pdf", dpi=400, bbox_inches='tight')
# plt.savefig("plots/Layer_mean_std_MNIST.png", dpi=400, bbox_inches='tight')
#
# '''
# # Plot 5
# Plot layer singular values with BN, CN and SN for MNIST for all layers across training
# '''
# smoothing_factor = 15
# lr = 1 # 0.001
# plt.figure(figsize=(14,5))
# for j in range(3,6):
#     ax = plt.subplot(1,3,j-2)
#     for i in range(3):
#         name = "./log/"+ foldernames[j][lr] + "/" + foldernames[j][lr]+"_layer_singular_"+str(i)+".csv"
#         print(name)
#
#         mean = np.genfromtxt(name, delimiter=',')
#
#         mean = running_mean(mean, N=smoothing_factor)
#         plt.plot(mean, lw= 2, label = "Layer "+ str(i+1))
#
#     plt.legend(fontsize=15)
#     plt.ylabel("Layer Singular values", fontsize=15)
#     plt.xlabel("Training Epoch", fontsize=15)
#     plt.grid(True)
#     plt.title(titles[j-3],fontsize=15)
# plt.tight_layout()
# plt.savefig("plots/Layer_singular_MNIST.pdf", dpi=400, bbox_inches='tight')
# plt.savefig("plots/Layer_singular_MNIST.png", dpi=400, bbox_inches='tight')
plt.show()