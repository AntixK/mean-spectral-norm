from model import MNISTConvNet, MNISTSNConvNet,MNISTBNConvNet,MNISTMSNConvNet, DenseNet, SNDenseNet, MSNDenseNet, vgg16
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import pylab as plt
import seaborn as sns
# import torchvision.utils as vutils
# from torch.autograd import Variable
import torch.backends.cudnn as cudnn
# from torch.autograd import grad
import torch.utils.data
from time import time
import random
import os
import numpy as np


sns.set_style("darkgrid")
current_palette = sns.color_palette()
cudnn.benchmark = True

class RunExperiments(object):
    def __init__(self, batchsize=64, dataset='MNIST', model_type='SNconv', lr = 0.001):
        self.batchsize= batchsize
        self.lr = lr
        self.dataset = dataset
        self.model_type = model_type

        self.model_name = self.model_type + '_' + str(self.lr) + '_' + self.dataset
        print("#############################################")
        print("         Training "+self.model_name)
        print("#############################################")


        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4), # 28 for MNIST and FashionMNIST
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        self.img_size = 0

        if self.dataset == 'CIFAR10':
            normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
            aug_trans = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
            common_trans = [transforms.ToTensor(), normalize]
            train_compose = transforms.Compose(aug_trans + common_trans)
            test_compose = transforms.Compose(common_trans)

            d_func = datasets.CIFAR10
            train_set = datasets.CIFAR10('data', train=True, transform=train_compose, download=True)
            test_set = datasets.CIFAR10('data', train=False, transform=test_compose)
            self.trainloader = torch.utils.data.DataLoader(train_set, batch_size=self.batchsize, shuffle=True, num_workers=2)
            self.train_len = len(self.trainloader)

            self.testloader = torch.utils.data.DataLoader(test_set, batch_size=self.batchsize, shuffle=False, num_workers=2)
            self.test_len = len(self.testloader)
            num_classes = 10

            self.img_size = 32

        elif self.dataset == 'SVHN':
            normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
            common_trans = [transforms.ToTensor(), normalize]
            train_compose = transforms.Compose(common_trans)
            test_compose = transforms.Compose(common_trans)

            train_set = datasets.SVHN('data', split = 'train', transform=train_compose, download=True)
            test_set = datasets.SVHN('data', split = 'test', transform=test_compose, download=True)
            self.trainloader = torch.utils.data.DataLoader(train_set, batch_size=self.batchsize, shuffle=True, num_workers=4)
            self.train_len = len(self.trainloader)

            self.testloader = torch.utils.data.DataLoader(test_set, batch_size=self.batchsize, shuffle=False, num_workers=4)
            self.test_len = len(self.testloader)
            num_classes = 10

            self.img_size = 32

        elif dataset == 'FashionMNIST':
            trainset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_train)
            self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
            self.train_len = len(self.trainloader)

            testset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_test)
            self.testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
            self.test_len = len(self.testloader)
            self.img_size = 28
        elif dataset == 'MNIST':
            trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
            self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
            self.train_len = len(self.trainloader)

            testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
            self.testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
            self.test_len = len(self.testloader)
            self.img_size = 28

        manualSeed = 8088 # random.randint(1, 10000)
        print("Random Seed: ", manualSeed)
        random.seed(manualSeed)
        torch.manual_seed(manualSeed)
        torch.cuda.manual_seed_all(manualSeed)

        if self.dataset in ['MNIST','FashionMNIST']:
            if self.model_type == 'SNconv':
                self.net = MNISTSNConvNet().cuda()
                self.mult = 3
            elif self.model_type == 'BNconv':
                self.net = MNISTBNConvNet().cuda()
                self.mult = 4
            elif self.model_type == 'Conv':
                self.net = MNISTConvNet().cuda()
                self.mult = 2
            elif self.model_type == 'MSNconv':
                self.net = MNISTMSNConvNet().cuda()
                self.mult = 5

            # Create placeholder for layer-wise gradients
            self.layer_grads = [np.empty((400, 0)), np.empty((4608, 0)), np.empty((16384, 0)), np.empty((81920, 0))]
            self.layer_grad_eigs = [[], [], []]
            self.layer_mean = [[], [], [], []]
            self.layer_std = [[], [], [], []]
            self.layer_rank = [[], [], []]
            self.layer_singular = [[], [], []]
            self.num_logs = 3

        elif self.dataset == 'SVHN':
            if self.model_type == 'SNconv':
                self.net = vgg16(norm='SN').cuda()
                self.mult = 3
            elif self.model_type == 'BNconv':
                self.net = vgg16(norm='BN').cuda()
                self.mult = 4
            elif self.model_type == 'MSNconv':
                self.net = vgg16(norm='MSN').cuda()
                self.mult = 4

        elif self.dataset == 'CIFAR10':
            if self.model_type == 'SNconv':
                self.net = SNDenseNet().cuda()
                self.mult = 3
            elif self.model_type == 'BNconv':
                self.net = DenseNet().cuda()
                self.mult = 4
            elif self.model_type == 'MSNconv':
                self.net = MSNDenseNet().cuda()
                self.mult = 4

            self.layer_grads = []
            self.layer_grad_eigs = []
            self.layer_mean = []
            self.layer_std = []
            self.layer_rank = []
            self.layer_singular = []
            self.densenet_weight_sizes = [648, 1152, 5184, 1728, 5184, 2304, 5184, 2880, 5184,
                                     3456, 5184, 4032, 5184, 4608, 5184, 5184, 5184, 5760,
                                     5184, 6336, 5184, 6912, 5184, 7488, 5184, 8064, 5184,
                                     8640, 5184, 9216, 5184, 9792, 5184, 37152, 8256, 5184,
                                     8832, 5184, 9408, 5184, 9984, 5184, 10560, 5184, 11136,
                                     5184, 11712, 5184, 12288, 5184, 12864, 5184, 13440, 5184,
                                     14016, 5184, 14592, 5184, 15168, 5184, 15744, 5184, 16320,
                                     5184, 16896, 5184, 105924, 13968, 5184, 14544, 5184, 15120,
                                     5184, 15696, 5184, 16272, 5184, 16848, 5184, 17424, 5184,
                                     18000, 5184, 18576, 5184, 19152, 5184, 19728, 5184, 20304,
                                     5184, 20880, 5184, 21456, 5184, 22032, 5184, 22608, 5184]

            self.sampled_indices= [6,13,44,56,67,75,84,93]

            for i in self.sampled_indices:
                self.layer_grads.append(np.empty((self.densenet_weight_sizes[i], 0)))

            for i in range(8):
                #self.layer_grads.append(np.empty(,0))

                self.layer_grad_eigs.append([])
                self.layer_mean.append([])
                self.layer_std.append([])
                self.layer_rank.append([])
                self.layer_singular.append([])
            self.num_logs = 8
        else:
            raise ValueError('Invalid Dataset Given')
        print("Number of parameters in the %s: %d"%(self.model_name, sum(p.numel() for p in self.net.parameters())))

        # output = net(data)
        # print(output.size())
        #net = torch.nn.DataParallel(net)
        #print(list(self.net.parameters()))
        self.criterion = nn.CrossEntropyLoss()
        #self.optimizer = optim.SGD(self.net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self.best_acc = 0.0
        self.test_loss_log = []
        self.test_acc = []
        self.train_loss_log = []
        self.train_acc = []

        if not os.path.isdir('log/'):
            os.mkdir('log')

        if not os.path.exists('log/'+self.model_name+'/'):
            os.mkdir('log/'+self.model_name+'/')
        self.model_path = './log/'+self.model_name + '/'

    def log_layer_params(self, log_grads = False):
        params = list(self.net.parameters())
        print("params = ", [x.shape for x in params])
        print([x.size() for i, x in enumerate(params)])# if x.dim() == 4])
        #
        #print(idx)
        # print(params[0].grad.cpu().numpy().ravel().shape)
        # print(params[4].grad.cpu().numpy().ravel().shape)
        # print(params[8].grad.cpu().numpy().ravel().shape)
        # print(params[12].grad.cpu().numpy().ravel().shape)
        # plt.hist(params[0].grad.cpu().numpy().ravel(), bins = 'auto')

        # grads = params[0].grad.cpu().numpy().ravel().reshape(-1,1)
        # print(layer_grads.shape)
        # layer_grads = np.hstack((layer_grads, grads))
        # print(layer_grads.shape)

        if self.dataset == 'MNIST':
            # Log gradients
            for i in range(4):
                # Log mean and std of each layer weight
                weights = params[i * self.mult].data.cpu().numpy()
                #print(params[i * self.mult].size())
                #print(weights.shape)
                self.layer_mean[i].append(weights.mean())
                self.layer_std[i].append(weights.std())

                grads = params[i * self.mult].grad.cpu().numpy()
                if i < 3:
                    # Log mean rank and mean singular values of each layer
                    #print(weights.shape)
                    #print(np.linalg.matrix_rank(weights))
                    self.layer_rank[i].append(np.linalg.matrix_rank(weights).mean())
                    _, S, _ = np.linalg.svd(weights)
                    self.layer_singular[i].append(S.max(axis=2).mean())

                    # Log Mean Max Eigen value of the gradient matrix
                    self.layer_grad_eigs[i].append(np.abs(np.linalg.eigvals(grads)).mean(axis=2).max())

                if log_grads == True:
                    #print(i, self.layer_grads[i].shape, grads.shape)
                    self.layer_grads[i] = np.hstack((self.layer_grads[i], grads.ravel().reshape(-1, 1)))
                    # print(self.layer_grads[i].shape)
        else:
            idx = np.array([i for i, x in enumerate(params) if x.dim() == 4])
            i = 0
            self.dense_net_indices = idx[self.sampled_indices]
            # Log gradients
            for j in self.dense_net_indices:
                # Log mean and std of each layer weight
                weights = params[j].data.cpu().numpy()
                # print(params[i * self.mult].size())
                # print(weights.shape)
                self.layer_mean[i].append(weights.mean())
                self.layer_std[i].append(weights.std())

                grads = params[j].grad.cpu().numpy()

                # Log mean rank and mean singular values of each layer
                #print(i,j, self.dense_net_indices[j], weights.shape)
                #print(np.linalg.matrix_rank(weights))
                self.layer_rank[i].append(np.linalg.matrix_rank(weights).mean())
                _, S, _ = np.linalg.svd(weights)
                self.layer_singular[i].append(S.max(axis=2).mean())

                # Log Mean Max Eigen value of the gradient matrix
                self.layer_grad_eigs[i].append(np.abs(np.linalg.eigvals(grads)).mean(axis=2).max())
                if log_grads == True:
                    #print(i, j, self.layer_grads[i].shape, grads.shape)
                    self.layer_grads[i] = np.hstack((self.layer_grads[i], grads.ravel().reshape(-1, 1)))
                    # print(self.layer_grads[i].shape)
                i += 1

    def train(self, epoch):
        self.net.train()
        train_loss = 0
        correct = 0
        total = 0
        start_time = time()
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            self.optimizer.zero_grad()
            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            if batch_idx % 500 == 0:
                if batch_idx % 20000 == 0:
                    self.log_layer_params(True)
                else:
                    self.log_layer_params(False)

            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            self.train_loss_log.append(train_loss/(batch_idx+1))
            self.train_acc.append(100. * correct / total)

            if batch_idx % 100 == 0:
                end_time = time() - start_time
                print('Training Epoch:%3d [%3d/%3d] Loss: %.3f | Acc: %.3f%% (%5d/%5d) Time: %.3f'
                % (epoch, batch_idx, self.train_len, train_loss/(batch_idx+1), 100.*correct/total, correct, total, end_time))
                start_time = time()

    def test(self, epoch):
        self.net.eval()
        test_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            start_time = time()
            for batch_idx, (inputs, targets) in enumerate(self.testloader):

                inputs, targets = inputs.cuda(), targets.cuda()
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                self.test_loss_log.append(test_loss/(batch_idx+1))
                self.test_acc.append(100.*correct/total)
                if batch_idx % 100 == 0:
                    end_time = time() - start_time
                    print('Testing Epoch: %3d [%3d/%3d] Loss: %.3f | Acc: %.3f%% (%5d/%5d) Time: %.3f'
                      % (epoch, batch_idx, self.test_len, test_loss / (batch_idx + 1), 100. * correct / total,
                         correct, total, end_time))
                    start_time = time()
        # Save checkpoint.
        acc = 100.*correct/total
        if acc > self.best_acc:
            print('Saving..')
            state = {
                'net': self.net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/'+self.model_name + '.t7')
            self.best_acc = acc

    def running_mean(self, x, N=5):
        from numpy import cumsum, insert
        cumsum = cumsum(insert(x, 0, 0))
        return (cumsum[N:] - cumsum[:-N]) / float(N)

    def run_experiment(self, start_epoch=0, num_epochs = 1):
        for epoch in range(start_epoch, start_epoch+num_epochs):
            self.train(epoch)
            self.test(epoch)

        np.savetxt(self.model_path+self.model_name+'_test_loss.csv',self.test_loss_log, delimiter = ',')
        np.savetxt(self.model_path+self.model_name+'_train_loss.csv',self.train_loss_log, delimiter = ',')
        np.savetxt(self.model_path+self.model_name+'_test_accuracy.csv',self.test_acc, delimiter = ',')
        np.savetxt(self.model_path+self.model_name+'_train_accuracy.csv',self.train_acc, delimiter = ',')

        for i in range(self.num_logs):
            #print(i)
            #print("Layer grad size : ", self.layer_grads[i].shape)
            np.savetxt(self.model_path+self.model_name+'_layer_grads_'+str(i)+'.csv',
            self.layer_grads[i], delimiter=",")

            #print("Layer grad eigs size : ", len(self.layer_grad_eigs[i]))
            np.savetxt(self.model_path+self.model_name+'_layer_grad_eigs_'+str(i)+'.csv',
            self.layer_grad_eigs[i], delimiter=",")

            #print("Layer mean size : ", len(self.layer_mean[i]))
            np.savetxt(self.model_path+self.model_name+'_layer_mean_'+str(i)+'.csv',
            self.layer_mean[i], delimiter=",")

            #print("Layer std size : ", len(self.layer_std[i]))
            np.savetxt(self.model_path+self.model_name+'_layer_std_'+str(i)+'.csv',
            self.layer_std[i], delimiter=",")

            #print("Layer rank size : ", len(self.layer_rank[i]))
            np.savetxt(self.model_path+self.model_name+'_layer_rank_'+str(i)+'.csv',
            self.layer_rank[i], delimiter=",")

            #print("Layer singular size : ", len(self.layer_singular[i]))
            np.savetxt(self.model_path+self.model_name+'_layer_singular_'+str(i)+'.csv',
            self.layer_singular[i], delimiter=",")

        plt.figure(figsize = (10,4))
        ax = plt.subplot(121)
        ax.plot(self.train_loss_log, lw=2, alpha=0.3, color = current_palette[0])
        ax.plot(self.test_loss_log, lw=2, alpha = 0.3, color = current_palette[1])
        ax.plot(self.running_mean(self.train_loss_log, N=5), lw=2,
                color = current_palette[0] ,label='Train Loss')
        ax.plot(self.running_mean(self.test_loss_log, N=5), lw=2,
                color = current_palette[1], label='Test Loss')
        plt.legend(fontsize=17)
        plt.xlabel('Epoch', fontsize=17)
        plt.ylabel('Loss', fontsize=17)
        plt.grid(True)

        ax = plt.subplot(122)
        ax.plot(self.train_acc, lw=2, alpha=0.3, color = current_palette[0])
        ax.plot(self.test_acc, lw=2, alpha = 0.3, color = current_palette[1])
        ax.plot(self.running_mean(self.train_acc, N=5), lw=2,
                color = current_palette[0] ,label='Train Accuracy')
        ax.plot(self.running_mean(self.test_acc, N=5), lw=2,
                color = current_palette[1], label='Test Accuracy')
        plt.legend(fontsize=17)
        plt.xlabel('Epoch', fontsize=17)
        plt.ylabel('Accuracy', fontsize=17)
        plt.tight_layout()
        plt.grid(True)
        plt.savefig( self.model_path+self.model_name+'.png', dpi=400, bbox_inches='tight')
        #plt.show()

grid = [(64, 'MNIST', 'MSNconv', 0.1),
        (64, 'MNIST', 'SNconv', 0.1),
        (64, 'MNIST', 'BNconv', 0.1),
        (64, 'MNIST', 'Conv', 0.1),
        (64, 'MNIST', 'MSNconv', 0.001),
        (64, 'MNIST', 'SNconv', 0.001),
        (64, 'MNIST', 'BNconv', 0.001),
        (64, 'MNIST', 'Conv', 0.001),
        (64, 'MNIST', 'MSNconv', 0.0001),
        (64, 'MNIST', 'SNconv', 0.0001),
        (64, 'MNIST', 'BNconv', 0.0001),
        (64, 'MNIST', 'Conv', 0.0001),
        (64, 'CIFAR10', 'BNconv', 0.1),
        (64, 'CIFAR10', 'SNconv', 0.1),
        (64, 'CIFAR10', 'MSNconv', 0.1),
        (64, 'CIFAR10', 'BNconv', 0.001),
        (64, 'CIFAR10', 'SNconv', 0.001),
        (64, 'CIFAR10', 'MSNconv', 0.001),
        (64, 'SVHN', 'BNconv', 0.1),
        (64, 'SVHN', 'SNconv', 0.1),
        (64, 'SVHN', 'MSNconv', 0.1),
        (64, 'SVHN', 'BNconv', 0.001),
        (64, 'SVHN', 'SNconv', 0.001),
        (64, 'SVHN', 'MSNconv', 0.001)
        ]

#print(torch.has_cudnn)
batchsize = 64
for dataset in ['SVHN']:#['MNIST', 'CIFAR10']:
    for model_type in ['MSNconv']:
        for lr in [0.0001]:#[0.1, 0.001]:
            exp = RunExperiments(batchsize, dataset, model_type, lr)
            exp.run_experiment(num_epochs = 1)

# for i, (batchsize, dataset, model_type, lr) in grid:
#     exp = RunExperiments(batchsize, dataset, model_type, lr)
#     exp.run_experiment(num_epochs = 400)