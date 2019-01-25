# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 23:34:19 2018

@author: gonthier
"""

import os
import numpy as np
import pandas as pd

def list_tofloat(l):
    r = []
    for elt in l:
        r += [float(elt)]
    return(r)

def ExtractBirds():
    """
    The goal of this function is to extract the Birds dataset from the Birds file
    
    return a list of 4 elements : 
            Dataset = list_names,bags,labels_bags,labels_instance
            with
    - list_names list of the names of the 13 classes
    - bags : list of the bags : a list of array of instances
    - labels_bags : array of the labels of the classes between -1 and 1 
        per class : ie a list of number_of_class lists
    - labels_instance : array of the labels of the classes between -1 and 1 
        per class
    This means that the instances and bags are always in the same order
    """
    
    
    path_dataset = 'Birds'
    name_file_bag_labels = os.path.join(path_dataset,'hja_birdsong_bag_labels.txt')
    name_file_features = os.path.join(path_dataset,'hja_birdsong_features.txt')
    name_file_instances_labels = os.path.join(path_dataset,'hja_birdsong_instance_labels.txt')
    name_file_names = os.path.join(path_dataset,'hja_birdsong_class_names.txt')
    
    number_of_class = 13
    number_of_bag = 548
    
    bags = [] # List of the features
    labels_instance = [] # List of the labels of the instance
    
    # Load the names of the class
    list_names = []
    with open(name_file_names) as input_file:
        for line in input_file:
            line = line.strip()
            list_names+= [line.split('-')[-1]]
       
    # Load the labels of the bags
    labels_bags = [-np.ones((number_of_bag,)) for j in range(number_of_class)] 
    # List of the 13 lists of labels per bag
    with open(name_file_bag_labels) as input_file:
        first_line = True
        for line in input_file:
            line = line.strip()
            if not(first_line):
                line_splitted = line.split(',')
                bag_id = int(line_splitted[0])-1
                for label in line_splitted[1:]:
                    labels_bags[int(label)-1][bag_id] = 1
            else:
                first_line = False
                
    # Load the features grouped by bag       
    with open(name_file_features) as input_file:
        # Each elt of bags is a bag : ie an array of where each line is an instance
        first_line = True
        current_bag_id = -1
        array_of_features = None
        for line in input_file:
            line = line.strip()
            if not(first_line):
                line_splitted = line.split(',')
                bag_id = int(line_splitted[0])
                if bag_id==current_bag_id:
                    array_of_features = np.vstack([array_of_features,np.array(list_tofloat(line_splitted[1:])).reshape(1,-1)])
                else:
                    if not(current_bag_id==-1):
                        bags += [array_of_features]
                    current_bag_id = bag_id
                    array_of_features = np.array(list_tofloat(line_splitted[1:])).reshape(1,-1)
            else:
                first_line = False
        # Last bag
        bags += [array_of_features]
    
    # Load the labels of the instances
    labels_instance = [[[] for i in range(number_of_bag)] for j in range(number_of_class)]      
    with open(name_file_instances_labels) as input_file:
        first_line = True
        current_bag_id = -1
        array_of_features = None
        for line in input_file:
            line = line.strip()
            if not(first_line):
                line_splitted = line.split(',')
                bag_id = int(line_splitted[0])
                c_id = int(line_splitted[1])
                for c in range(number_of_class):
                    if c==c_id:
                        elt = 1
                    else:
                        elt = -1
                    labels_instance[c-1][bag_id-1] += [elt] 
            else:
                first_line = False
    
    # Convert to numpy array
    for j in range(number_of_class):
        for i in range(number_of_bag):
            labels_instance[j][i] = np.array(labels_instance[j][i])
            
            
    # Quick test
    assert(len(bags)==number_of_bag)
    for j in range(number_of_class):
        assert(len(labels_instance[j])==number_of_bag)
        assert(len(labels_instance[j])==number_of_bag)
            
    Dataset = list_names,bags,labels_bags,labels_instance
            
    return(Dataset)
        
                
    
                
    
    