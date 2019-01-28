# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 13:09:15 2019

@author: gonthier
"""

def getMethodConfig(method,dataset,expType):
    """
    method is a string containing the name of the MIL methods to use
    dataset is a string containing the name of the data set
    expType is a string containing the ype of experiment to conduct
    OUTPUT:
    opt is an object containing the field used to configure the method
    specified for the dataset and experiment type.
    """
    opt = None
    
    # Option selon la methode switch method
    if method=='miSVM':
        opt_method = None
    
    
    return(opt)