# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 11:48:55 2018

@author: gonthier
"""
import glob
import numpy as np
import os

from misvmio import parse_c45, bag_set


def ExtractSubsampledSIVAL():
    """
    The goal of this function is too extract a sample of the SIVAL dataset as 
    it is done in the survey Carbonneau 2016
    
    For each class the 60 positives bag are selected and 5 images/bag from each 
    of 24  other class are randomly selected
    """
    return(0)  

def ExtractSIVAL():
    """
    This return a dictionnary in which each element is a list of the exemple of 
    the class
    
    
    This function return a list of the 
    
    """
    number_of_class = 25
    number_of_bag = 1499
    bags = None # List of the features
    labels_instance = [] # List of the labels of the instance
    list_names = []
    labels_bags = []
    
    path_directory = 'SIVAL'
    data_to_read = os.path.join(path_directory,'*.data')
    allclasses=glob.glob(data_to_read)
    
    for classe_name in allclasses:
        classe = os.path.split(classe_name)[-1]
        elt_name = classe.split('.')[0]
        list_names += [elt_name]
        # Load list of C4.5 Examples
        classe_set = parse_c45(elt_name,rootdir=path_directory)
        bagset = bag_set(classe_set,bag_attr=0)
        
        # Convert bags to NumPy arrays
        # (The ...[:, 2:-1] removes first two columns and last column,
        #  which are the bag/instance ids and class label)
        if bags is None:
            bags = [np.array(b.to_float())[:, 2:-1] for b in bagset]
        labels_instance_c = [2*np.array(b.to_float())[:,-1]-1 for b in bagset]
        labels_instance += [labels_instance_c]
        labels = np.array([b.label for b in bagset], dtype=float)
        # Convert 0/1 labels to -1/1 labels
        labels = 2 * labels - 1
        
        labels_bags  += [labels]
    
    Dataset = list_names,bags,labels_bags,labels_instance
            
    return(Dataset)
    