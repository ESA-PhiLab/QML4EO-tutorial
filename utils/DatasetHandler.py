import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import glob
import os
import cv2

from config import *

class DatasetHandler:
    '''
        Class responsible to loading and processing datasets.
    '''

    def __init__(self, dataset_path):
        '''
            DatasetHandler constructor

            Parameters
            ----------
            - dataset_path: root path for the dataset

            !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            The dataset must be organized in this format:
            RootFolder
            │   README.md
            │   requirements.txt    
            │
            └───dataset_root
                └───class 1
                |   └───img0.jpg
                |   └───img1.jpg
                |   └─── ...
                |   └───img3000.jpg
                |
                ...
                └───class 10
                    └───img0.jpg
                    └───img1.jpg
                    └─── ...
                    └───img3000.jpg
                         
            The following image format are supported:
                - .png
                - .jpg
                - .jpeg
            !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            Returns
            -------
            Nothing, DatasetHandler object is created
        '''

        self.dataset_path = dataset_path
        # If you need the datahandler just for loading function you can set dataset_path to None
        if dataset_path is not None: 
            self.classes = glob.glob(os.path.join(dataset_path, '*'))
        else:
            self.classes = []
    
    def print_classes(self):
        '''
            Print dataset classes.

            Parameters
            ----------
            Nothing, it uses the self.classes variable

            Returs
            ------
            Nothing, it prints the dataset classes

        '''
        print('Classes: ') 
        for i, c in enumerate(self.classes): 
            print('     Class ' + str(i) + ' ->', c)

    def load_paths_labels(self, root, classes):
        '''
            It load the image paths and contructs corresponding lables.

            !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            The dataset must be organized in this format:
            RootFolder
            │   README.md
            │   requirements.txt    
            │
            └───dataset_root
                └───class 1
                |   └───img0.jpg
                |   └───img1.jpg
                |   └─── ...
                |   └───img3000.jpg
                |
                ...
                └───class 10
                    └───img0.jpg
                    └───img1.jpg
                    └─── ...
                    └───img3000.jpg
                         
            The following image format are supported:
                - .png
                - .jpg
                - .jpeg
            !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        
            Parameters
            ----------
            - root: dataset root folder
            - classes: dataset classes

            Returns
            -------
            - imgs_paths: np array containing paths for dataset images
            - imgs_label: np array containing label for each image in the dataset
        
        '''

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
    
    def train_validation_split(self, images, labels, split_factor = 0.2):
        '''
            It splits the dataset into training and validation set given the split factor

            Parameters
            ----------
            - images: list or np array containing dataset image paths
            - labels: list or np array containing dataset image labels
            - split_factor: float values in [0 1] defining the split percentage between training and validation set
            Returns
            -------
        '''
        val_size = int(len(images)*split_factor)
        train_size = int(len(images) - val_size)
        return images[0:train_size], labels[0:train_size, ...], images[train_size:train_size+val_size], labels[train_size:train_size+val_size, ...]

    def data_loader(self, imgs_path, imgs_label, batch_size = 1, img_shape = (64, 64, 3), returnpath=False):
        '''
            Keras-like data loader
           
            Parameters
            ----------
            - imgs_path: list or np array containing dataset image paths
            - imgs_label: list or np array containing dataset image labels
            - batch_size: size of the batch for training
            - img_shape: shape of each image
            - return_path: if true the function will return also the batch image paths

            Returns
            -------
            - batch_in: tensor containing images
            - batch_out: tensor containing labels
            - imgs_path: list containing batch paths
        '''

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