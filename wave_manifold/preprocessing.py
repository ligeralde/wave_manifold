import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
import torchvision.transforms as transforms 
import itertools
from xml.dom import NotFoundErr
from skimage import measure
from wave_manifold.layers import Retina

class RetinalImageDataset(object):
    def __init__(self, retina, raw_data, targets=None):
        """Base class that loads an image dataset so it can be projected onto a retina
        ----------
        """

        self.retina = retina
        self.raw_data = raw_data
        self.targets = targets
        self.retinal_imageset = None

    def project_data_onto_retina(self, 
                                transform_data=False,
                                classes=None, 
                                equal_samples_per_class=False, 
                                create_dict=True
                                ):
        """
        Projects data onto the retina
        Args:
        retina: (object instance of layers.Retina) retina onto which the dataset will be projected
        digit_classes: list of which digit classes to include (default is 0/1)
        equal_samples_per_class: 
        """
        if self.targets is not None:

            #if no classes are specified, get all classes
            if classes == None:
                classes = list(np.array(torch.unique(self.targets)))

            #sort dataset by classes
            self.select_classes(classes, equal_samples_per_class=equal_samples_per_class)
            class_idx_list = list(itertools.chain.from_iterable(list(self.class_idxs.values())))
            imageset = self.raw_data[class_idx_list,:,:]
            self.targets = self.targets[class_idx_list]

        #if there are no targets specified 
        else:
            imageset = self.raw_data

        #optionally transform the data to fit on the retinal grid 
        if transform_data:
            imageset = self.transform_data(imageset)
        
        n_neurons = self.retina.n_neurons

        #initialize retinal images as 0's 
        retinal_imageset = torch.zeros(n_neurons, len(imageset))
        
        #loop through images in dataset 
        for idx, image in enumerate(imageset): 
            #loop through cells in the retina 
            for cell, xpos, ypos in zip(range(n_neurons), self.retina.pos[:,1], self.retina.pos[:,0]):
                #get cell positions as integers
                raw_xpos = np.around(xpos).astype('int')
                raw_ypos = np.around(ypos).astype('int')

                #adjust cell positions so they don't exceed image borders
                if raw_xpos == self.retina.grid_length:
                    raw_xpos -= 1
                if raw_ypos == self.retina.grid_length:
                    raw_ypos -= 1

                #assign binary values to retinal cells based on pixel intensity above mean
                if image[raw_ypos, raw_xpos] > image.mean():
                    retinal_imageset[cell, idx] = 1
                
            if idx%1000 == 0:
                print('Processed image {} out of {}'.format(idx, len(imageset)))

        self.retinal_imageset = retinal_imageset

        #add a dictionary 
        if self.targets is not None and create_dict == True:
            self.create_class_dictionary()
    

    def create_class_dictionary(self):
        if self.targets == None or self.retinal_imageset == None:
            raise NotFoundErr
        
        class_length = lambda one_class : len(self.class_idxs[one_class])
        
        retinal_imageset_dict = dict()
        start_idx = 0
        for one_class in self.class_idxs.keys():
            retinal_imageset_dict[one_class] = self.retinal_imageset[:,start_idx:start_idx+class_length(one_class)]
            start_idx += class_length(one_class)
        
        self.retinal_imageset_dict = retinal_imageset_dict


    def transform_data(self):
        """ Dummy function to be implemented in child classes 
        """
        # for MNIST data, this involves padding
        # for retinal waves data, this involves reducing/mean pooling 
        raise NotImplementedError


    def select_classes(self, classes, equal_samples_per_class):
        """
        Helper function for selecting indices from the desired classes
        
        Args:
        classes: list of strings or ints containing all the unique object categories (ex. digits, animals)
        
        Returns: 
        dictionary of elements containing the indices of datapoints from each class
        """
        if self.targets == None:
            raise NotImplementedError

        #find the positions of each digit class in the dataset and add to dictionary
        class_idxs = dict()
        for one_class in classes:
            class_idxs[one_class] = torch.where(self.targets == one_class)[0]

        #exercise option to ensure all digits are equally represented in the dataset
        if equal_samples_per_class == True:
            n = np.min([len(class_idxs[one_class]) for one_class in class_idxs])
            for one_class in class_idxs:
                class_idxs[one_class] = class_idxs[one_class][torch.randperm(len(class_idxs[one_class]))[:n]]

        self.class_idxs = class_idxs
        


class RetinalMNIST(RetinalImageDataset):
    def __init__(self, retina, train=True):
        """Class that loads the MNIST dataset and projects it onto a model retina.
        ----------
        train : True to load training data, False to load testing data
        """

        #load MNIST data
        raw_mnist = torch.utils.data.DataLoader(MNIST('../data', train=train, download=True,
                                                        transform=transforms.Compose([
                                                            transforms.ToTensor(),
                                                            transforms.Normalize((0.1307,), (0.3081,))
                                                            ])))

        super().__init__(retina, 
                         raw_mnist.dataset.data, 
                         raw_mnist.dataset.targets 
                        )


    def project_data_onto_retina(self,
                                 transform_data=True,
                                 **kwargs
                                 ):                          
        super(RetinalMNIST, self).project_data_onto_retina(transform_data, **kwargs)


    def transform_data(self, imageset):
        """
        Pad MNIST digits with zero rows and columns to match size of the retina. 
        """
        #get padding rows/columns that increase the digit images to match the length of the retina. 
        row_padding = torch.zeros(len(imageset), (self.retina.grid_length - 28)//2, 28)
        col_padding = torch.zeros(len(imageset), self.retina.grid_length, (self.retina.grid_length - 28)//2)

        #add rows and columns to images
        big_digits = torch.cat((imageset, row_padding), axis=1)
        big_digits = torch.cat((row_padding, big_digits), axis=1)
        big_digits = torch.cat((big_digits, col_padding), axis=2) 
        big_digits = torch.cat((col_padding, big_digits), axis=2)

        return(big_digits)



class RetinalLargeImageDataset(RetinalImageDataset):
    def __init__(self, retina, raw_data):
        """Class that loads a dataset larger than the projects it onto a model retina
        by mean averaging. 
        ----------
        train : True to load training data, False to load testing data
        """

        #load MNIST data

        super().__init__(retina, 
                         raw_data
                        )


    def project_data_onto_retina(self,
                                 transform_data=True,
                                 **kwargs
                                 ):                          
        super(RetinalLargeImageDataset, self).project_data_onto_retina(transform_data, **kwargs)


    def transform_data(self, imageset, block_size=4):
        """
        Mean pool images to reduce it to the size of the retina 
        """
        pooled_images = torch.tensor(np.array([self.pool_pixel_intensities(image, block_size) for image in imageset]))

        return(pooled_images)


    def pool_pixel_intensities(self, image, block_size):
        """
         Mean pool one
        """
        return(measure.block_reduce(image, block_size, np.mean))