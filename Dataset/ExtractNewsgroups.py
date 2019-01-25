# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 16:45:35 2019

In this file we extract the Newsgroups dataset 

@author: gonthier
"""

import os
import numpy as np
import glob

def list_tofloat(l):
    r = []
    for elt in l:
        r += [float(elt)]
    return(r)

def ExtractNewsgroups():
    """
    The goal of this function is to extract the Newsgoups dataset 
    from the Newsgoups files
    
    return a list of 4 elements : 
            Dataset = list_names,bags,labels_bags,labels_instance
            with
    - list_names list of the names of the 13 classes
    - bags : list of the bags : a list of array of instances
    - labels_bags : array of the labels of the classes between -1 and 1 
        per class : ie a list of number_of_class lists 
    - labels_instance : array of the labels of the classes between -1 and 1 
        per class
    """
    number_of_class = 20
    number_of_bag = 100
    number_of_features = 200
    bags = [] # List of the features
    labels_instance = [] # List of the labels of the instance
    list_names = []
    labels_bags = []
    
    path_directory = 'Newsgroups'
    data_to_read = os.path.join(path_directory,'*.txt')
    allclasses=glob.glob(data_to_read)
    
    FirstTimeBag = True
    labels_instance = [[[] for i in range(number_of_bag)] for j in range(number_of_class)] 
    labels_bags = [-np.empty((number_of_bag,)) for j in range(number_of_class)] 
    for c,classe_name in enumerate(allclasses):
        classe = os.path.split(classe_name)[-1]
        elt_name = classe.split('.')[0]
        list_names += [elt_name]
        
        with open(classe_name,'r') as f:
            content = f.readlines()[6:] # skip header
            bag_id_old = -1
            bag = None
            labels_instance_c = []
            labels = []
            for line in content:
                line_splitted=line.split(' ')
#                % Each line corresponds to an instance
#                % Column1: Bag ID
#                % Column2: Bag class label
#                % Column3: Instance ID
#                % Column4: Instance class label
#                % Column5->204: Attribute values of the instance (200 numeric attributes)
                bag_id = int(line_splitted[0])
                bag_label = int(line_splitted[1])
                bag_label = 2 * bag_label - 1
                labels_bags[c][bag_id-1] = bag_label
#                instance_id = int(line_splitted[2])
                instance_label = int(line_splitted[3])
                labels_instance[c][bag_id-1] += [2*instance_label-1] 

                features = np.array(list_tofloat(line_splitted[4:203]))
                if FirstTimeBag:
                    if bag_id_old==bag_id:
                        bag = np.vstack((bag,features))
                    else:
                        if not(bag_id_old==-1):
                            bags += [bag]
                        bag_id_old = bag_id
                        bag = features
        
        # End of the class
        if FirstTimeBag:
            bags += [bag]
            FirstTimeBag = False
#        labels_instance += [np.array(labels_instance_c)]
#        labels_bags  += [np.array(labels)]

    for j in range(number_of_class):
        for i in range(number_of_bag):
            labels_instance[j][i] = np.array(labels_instance[j][i])
            
    # Quick test
    assert(len(bags)==number_of_bag)
    for j in range(number_of_class):
        assert(len(labels_instance[j])==number_of_bag)
        assert(len(labels_instance[j])==number_of_bag)
        assert(len(labels_bags[j])==number_of_bag)
        assert(len(labels_bags[j])==number_of_bag)

    Dataset = list_names,bags,labels_bags,labels_instance
            
    return(Dataset)

