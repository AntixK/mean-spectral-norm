import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.utils.data
import torch.backends.cudnn as cudnn
from torch.autograd import grad
from time import time
import random
import argparse
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from model import _netG, _netD, _MSNnetD
from SNlayers import SNConv2d

parser = argparse.ArgumentParser(description='train SNDCGAN model')
parser.add_argument('--cuda', default=True, action='store_true', help='enables cuda')
parser.add_argument('--gpu_ids', default=0, help='gpu ids: e.g. 0,1,2, 0,2.')
parser.add_argument('--manualSeed', type=int, help='manual seed',default=8088)
parser.add_argument('--n_dis', type=int, default=1, help='discriminator critic iters')
parser.add_argument('--nz', type=int, default=128, help='dimention of lantent noise')
parser.add_argument('--batchsize', type=int, default=64, help='training batch size')
opt = parser.parse_args()
print(opt)

dataset = datasets.CIFAR10(root='dataset', download=True,
                           transform=transforms.Compose([
                               transforms.Resize(32),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchsize,
                                         shuffle=True, num_workers=int(2))

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)
    #torch.cuda.set_device(opt.gpu_ids[2])

cudnn.benchmark = True

def weight_filler(m):
    classname = m.__class__.__name__
    if classname.find('Conv' or 'SNConv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def gradient_penalty(x, f):
    # interpolation
    shape = [x.size(0)] + [1] * (x.dim() - 1)
    alpha = torch.rand(shape).cuda()
    beta  = torch.rand(x.size()).cuda()
    y = x + 0.5 * x.std() * beta
    z = x + alpha * (y - x)

    # gradient penalty
    z = Variable(z, requires_grad=True).cuda()
    o = f(z)
    g = grad(o, z, grad_outputs=torch.ones(o.size()).cuda(), create_graph=True)[0].view(z.size(0), -1)
    M = 0.05 * torch.ones(z.size(0))
    M = Variable(M).cuda()
    zer =  Variable(torch.zeros(z.size(0))).cuda()
    gp = ((g.norm(p=2, dim=1) - 1)**2).mean() + torch.max(zer,(g.norm(p=2,dim=1) - M)).mean()

    return gp

n_dis = opt.n_dis
nz = opt.nz

def train_GAN(model = "SNGAN"):
    D_loss = []
    G_loss = []
    Dx_loss = []
    DGx_loss = []
    G = _netG(nz, 3, 64)

    if model == "SNGAN":
        SND = _netD(3, 64)
    elif model == "MSNGAN":
        SND = _MSNnetD(3, 64)
    else:
        raise ValueError("Invalid GAN type given")

    if not os.path.exists('log/' + model + '/'):
        os.mkdir('log/' + model + '/')

    print(G)
    print(SND)
    G.apply(weight_filler)
    SND.apply(weight_filler)

    input = torch.FloatTensor(opt.batchsize, 3, 32, 32)
    noise = torch.FloatTensor(opt.batchsize, nz, 1, 1)
    fixed_noise = torch.FloatTensor(opt.batchsize, nz, 1, 1).normal_(0, 1)
    label = torch.FloatTensor(opt.batchsize)
    real_label = 1
    fake_label = 0

    fixed_noise = Variable(fixed_noise)
    criterion = nn.BCELoss()

    if opt.cuda:
        G.cuda()
        SND.cuda()
        criterion.cuda()
        input, label = input.cuda(), label.cuda()
        noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

    optimizerG = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizerSND = optim.Adam(SND.parameters(), lr=0.0002, betas=(0.5, 0.999))
    print(len(list(SND.parameters())))

    num_iter = 400
    for epoch in range(num_iter):
        start_time = time()
        for i, data in enumerate(dataloader, 0):

            step = epoch * len(dataloader) + i
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            SND.zero_grad()
            real_cpu, _ = data
            batch_size = real_cpu.size(0)
            if opt.cuda:
               real_cpu = real_cpu.cuda()
            input.resize_(real_cpu.size()).copy_(real_cpu)
            label.resize_(batch_size).fill_(real_label)
            inputv = Variable(input)
            labelv = Variable(label)
            output = SND(inputv)

            errD_real = torch.mean(F.softplus(-output))
            #errD_real = criterion(output, labelv)
            #errD_real.backward()
            D_x = output.data.mean()
            # train with fake
            noise.resize_(batch_size, nz, 1, 1).normal_(0, 1)
            noisev = Variable(noise)
            fake = G(noisev)
            labelv = Variable(label.fill_(fake_label))
            output = SND(fake.detach())
            errD_fake = torch.mean(F.softplus(output))
            #errD_fake = criterion(output, labelv)

            D_G_z1 = output.data.mean()
            #grad_penal = gradient_penalty(inputv.data, SND)
            errD = errD_real + errD_fake #+ grad_penal*10.0
            #print(output)
            errD.backward()
            optimizerSND.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            if step % n_dis == 0:
                G.zero_grad()
                labelv = Variable(label.fill_(real_label))  # fake labels are real for generator cost
                output = SND(fake)
                errG = torch.mean(F.softplus(-output))
                #errG = criterion(output, labelv)
                errG.backward()
                D_G_z2 = output.data.mean()
                optimizerG.step()
            if i % 100 == 0:
                end_time = time() - start_time
                print('[%3d/%3d][%3d/%3d] Loss_D: %.4f Loss_G: %.4f D(x): %+.4f D(G(z)): %+.4f / %+.4f Time : %.3f'
                      % (epoch, num_iter, i, len(dataloader),
                         errD.data.item(), errG.data.item(), D_x, D_G_z1, D_G_z2, end_time))

                G_loss.append(errG.data.item())
                D_loss.append(errD.data.item())
                Dx_loss.append(D_x)
                DGx_loss.append(D_G_z1 / D_G_z2)

                start_time = time()
            if i % 100 == 0:
                vutils.save_image(real_cpu,
                        '%s/%s/real_samples.png' % ('log', model),
                        normalize=True)
                fake = G(fixed_noise)
                vutils.save_image(fake.data,
                        '%s/%s/fake_samples_epoch_%03d.png' % ('log', model, epoch),
                        normalize=True)
        # do checkpointing
    torch.save(G.state_dict(), '%s/%s/netG_epoch_%d.pth' % ('log', model, epoch))
    torch.save(SND.state_dict(), '%s/%s/netD_epoch_%d.pth' % ('log', model, epoch))

    np.savetxt('log/'+model+'/G_loss.csv', G_loss, delimiter=',')
    np.savetxt('log/'+model+'/D_loss.csv', D_loss, delimiter=',')
    np.savetxt('log/'+model+'/Dx_loss.csv', Dx_loss, delimiter=',')
    np.savetxt('log/'+model+'/DGx_loss.csv', DGx_loss, delimiter=',')

train_GAN(model = "MSNGAN")
train_GAN(model = "SNGAN")

