import matplotlib.pyplot as plt
import numpy as np
import torch
from model import _netG, _netD, _MSNnetD
import torchvision.utils as vutils
from torch.autograd import Variable
import os
import seaborn as sns
import pandas as pd
from inception_score import inception_score
from torch.utils import data
from skimage import io, transform
from torchvision import  transforms

sns.set_style('darkgrid')


def running_mean( x, N=5):
    from numpy import cumsum, insert
    cumsum = cumsum(insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def Generate_images(model, model_checkpoint, path, num_images = 1000):
    # Load the model
    model_path = path + '/' + model_checkpoint
    if os.path.isfile(model_path):
        print("Loading model checkpoint '{}'".format(model_path))
        check_point = torch.load(model_path)
        model.load_state_dict(check_point)
        #start_epoch = check_point['epoch']

        # Subfolder for separate images
        if not os.path.exists(path + '/'+'final_images'):
            os.mkdir(path + '/'+'final_images')

        img_path = path + '/'+'final_images'+'/'
        print("Saving Image to ", img_path)

        image_names = []
        # Generate New fake images
        for i in range(num_images):
            # Fixed noise
            fixed_noise = torch.FloatTensor(64, 128, 1, 1).normal_(0, 1)
            fixed_noise = Variable(fixed_noise)
            fake = model(fixed_noise)
            img_name = 'New_fake_samples_%03d.png'%i
            vutils.save_image(fake.data,
                              '%sNew_fake_samples_%03d.png' % (img_path, i),
                              normalize=True)
            image_names.append(img_name)

        # Create CSV file of all the image names
        np.savetxt(img_path + "img_list.csv", image_names, fmt = '%s', delimiter=',')
    else:
        raise ValueError("No Pytorch model available")

    return img_path, model

# Compute Inception Score
class Result_Img_Dataset(data.Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(root_dir + csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name).T
        #print("image size :",image.shape)
        # landmarks = self.landmarks_frame.iloc[idx, 1:].as_matrix()
        # landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = image

        if self.transform:
            sample = self.transform(sample)
        return sample


MSN_path = "./log/MSNGAN"
model_checkpoint = 'netG_epoch_399.pth'
model = _netG(128, 3, 64)
MSN_root_path, MSN_trained_model = Generate_images(model, model_checkpoint, MSN_path, 1000)

MSN_root_path = MSN_path + '/'+'final_images'+'/'

SN_path = "./log/SNGAN"
model_checkpoint = 'netG_epoch_399.pth'
model = _netG(128, 3, 64)
SN_root_path, SN_trained_model = Generate_images(model, model_checkpoint, SN_path, 1000)

SN_root_path = SN_path + '/'+'final_images'+'/'

trans = transforms.Compose([transforms.ToTensor(),
transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# MSN_imgs = Result_Img_Dataset(csv_file = 'img_list.csv', root_dir = MSN_root_path, transform=None)
# # imgs = path + '/fake_samples_epoch_399.png'
# MSN_mean, MSN_std = inception_score(MSN_imgs, cuda=True, batch_size=32, resize=True, splits=1)
# print("MSN Inception Score : {} +/- {}".format(MSN_mean, MSN_std))
#
# SN_imgs = Result_Img_Dataset(csv_file = 'img_list.csv', root_dir = SN_root_path, transform=None)
# # imgs = path + '/fake_samples_epoch_399.png'
# SN_mean, SN_std = inception_score(SN_imgs, cuda=True, batch_size=32, resize=True, splits=1)
# print("SN Inception Score : {} +/- {}".format(SN_mean, SN_std))

# Get plots
# filenames= ["/G_loss.csv", "/D_loss.csv", "/Dx_loss.csv", "/DGx_loss.csv"]
#
# plt.figure(figsize=(8,5))
#
# title = ["Generator Loss", "Discriminator Loss", "$D(x)$", "$D(G(x))$"]
# for i in range(2):
#     ax = plt.subplot(1,2,i+1)
#
#     msn_file = np.genfromtxt(MSN_path+filenames[i], delimiter=',')
#     sn_file = np.genfromtxt(SN_path+filenames[i], delimiter=',')
#
#
#     plt.plot(running_mean(msn_file, N=50), lw=2, label = 'MSN')
#     plt.plot(running_mean(sn_file, N=50), lw=2, label = "SN")
#
#     plt.legend(fontsize = 15)
#     plt.xlabel('Training Epochs', fontsize=15)
#     plt.xlabel('Loss', fontsize=15)
#     plt.title(title[i], fontsize=15)
#
# plt.figure(figsize=(8,5))
# for i in range(2,4):
#     ax = plt.subplot(1,2,i-1)
#     msn_file = np.genfromtxt(MSN_path+filenames[i], delimiter=',')
#     sn_file = np.genfromtxt(SN_path+filenames[i], delimiter=',')
#
#     plt.plot(running_mean(msn_file, N=15), lw=2, label = 'MSN')
#     plt.plot(running_mean(sn_file, N=15), lw=2, label = "SN")
#
#     plt.legend(fontsize = 15)
#     plt.xlabel('Training Epochs', fontsize=15)
#     plt.xlabel('Loss', fontsize=15)
#     plt.title(title[i], fontsize=15)
#
# plt.show()





