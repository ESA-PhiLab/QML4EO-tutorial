import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import glob
import os
import cv2

from config import *

class DatasetHandler:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.classes = glob.glob(os.path.join(dataset_path, '*'))
    
    def print_classes(self):
        print('Classes: ') 
        for i,c in enumerate(self.classes): 
            print('     Class ' + str(i) + ' ->', c)

    def load_paths_labels(self, root, classes):
        # Initialize images path and images label lists
        imgs_path = []
        imgs_label = []
        
        # For each class in the class list
        for c in classes:
            # List all the images in that class
            paths_in_c = glob.glob(os.path.join(root, c+os.path.sep, '*'))
            #print(os.path.join(root, c+os.path.sep, '*'))
            # For each image in that class
            for path in paths_in_c:
                # Append the path of the image in the images path list
                imgs_path.append(path)
                imgs_label.append(CLASS_DICT[c])

        # Shuffle paths and labels in the same way
        c = list(zip(imgs_path, imgs_label))
        random.shuffle(c)
        imgs_path, imgs_label = zip(*c)

        return np.array(imgs_path), np.array(imgs_label)
    
    # Split the dataset into training and validation dataset
    def train_validation_split(self, images, labels, split_factor = 0.2):
        val_size = int(len(images)*split_factor)
        train_size = int(len(images) - val_size)
        return images[0:train_size], labels[0:train_size, ...], images[train_size:train_size+val_size], labels[train_size:train_size+val_size, ...]
    
    # Data genertor: given images paths and images labels yield a batch of images and labels
    def data_loader(self, imgs_path, imgs_label, batch_size = 1, img_shape = (64, 64, 3), returnpath=False):
        # Initialize the vectors to be yield
        batch_in = np.zeros((batch_size, img_shape[2], img_shape[0], img_shape[1]))
        batch_out = np.zeros((batch_size))

        # Repeat until the generator will be stopped
        while True:
            # Load a batch of images and labels
            for i in range(batch_size):
                # Select a random image and labels from the dataset
                index = random.randint(0, len(imgs_path)-1)
                # Fill the vectors with images and labels
                img = plt.imread(imgs_path[index])
                img = cv2.resize(img, (img_shape[0], img_shape[1]))
                if img.max() >= 100: img = img/255.0
                batch_in[i, ...] = np.transpose(img)
                batch_out[i] = imgs_label[index]
            # Yield/Return the image and labeld vectors
            if returnpath:
              yield  torch.Tensor(batch_in),  torch.Tensor(batch_out).type(torch.LongTensor), imgs_path[index]
            else:
              yield  torch.Tensor(batch_in),  torch.Tensor(batch_out).type(torch.LongTensor)