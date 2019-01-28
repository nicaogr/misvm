# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 15:49:39 2018

@author: gonthier
"""
    
import numpy as np
import misvm
from ExtractBirds import ExtractBirds
from ExtractSIVAL import ExtractSIVAL,ExtractSubsampledSIVAL
from ExtractNewsgroups import ExtractNewsgroups
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score,auc,accuracy_score,recall_score
from getMethodConfig import getMethodConfig

def mainTestFunction(allMethods=None,dataset=None):
    """
    @param : allMethods list of the methods to test
    """
    ## PARAMETERS
    if isinstance(allMethods, str):
        allMethods=[allMethods]

    Dataset=getDataset(dataset)
    ## LOAD DATASET

    print('=============================================================')
    print('= DATA SET ACQUIRED: ')
    print('-------------------------------------------------------------')
    ## TEST METHODS ON THE DATA SET
    
    for i in range(1,len(allMethods)).reshape(-1): # Boucle sur les methodes
        method=allMethods[i]
        print('= Method: ',method)
        # get config
        #opt=getMethodConfig(method,dataset,'single')
        opt = None
#        if exist('DT','var'):
        if False:
            print("bouh")
            # perform normalization if necessary
#            D,DT=normalizeDataSet(D,DT,opt)
            #perf,perfB=performExperimentWithTestSet(D,DT,method,opt)
        else:
            # perform normalization if necessary
#            D=normalizeDataSet(D,[],opt)
            list_names,bags,labels_bags,labels_instance = Dataset
            for c_i,c in enumerate(list_names):
                # Loop on the different class, we will consider each group one after the other
                print("For class :",c)
                labels_bags_c = labels_bags[c_i]
                labels_instance_c = labels_instance[c_i]
                bags_k = bags
                D = bags,labels_bags_c,labels_instance_c
                perf,perfB=performExperimentWithCrossVal(D,method,opt,dataset)
                
        ## Results
        print('=============================================================')
        print('= ',method)
        print('-------------------------------------------------------------')
        print('- instances')
        print('AUC: ',(perf.AUC))
        print('UAR: ',(perf.UAR))
        print('- bags')
        print('AUC: ',(perfB.AUC))
        print('UAR: ',(perfB.UAR))
        print('-------------------------------------------------------------')
        # save results
        #fn=(['Results/',dataset,'-',method,'-',date])
        #save(fn,'perf','perfB')
    
    return 0
    
if __name__ == '__main__':
    pass
    
def getTest_and_Train_Sets(Data,indextrain,indextest):
    """
    Split the list of data in train and test set according to the index lists
    provided
    """
    DataTrain = [ Data[i] for i in indextrain]
    DataTest = [ Data[i] for i in indextest]
    return(DataTrain,DataTest)

def performExperimentWithCrossVal(D=None,method=None,opt=None,dataset=None):
    """
    Perform nRep times a nFolds cross validation of the method on a specific 
    database and return performance at instance level and at bag level
    @param : D the dataset of a couple of list [bags,labels_bag,labels_instance]
    """
    bags,labels_bags_c,labels_instance_c  = D
    
    nRep=10
    nFolds=10
    numMetric = 4 


    if dataset=='SIVAL':
        num_sample = 5
        perfObj=np.empty((num_sample,nRep,nFolds,numMetric))
        perfObjB=np.empty((num_sample,nRep,nFolds,numMetric))
        for k in range(num_sample):
            labels_bags_c_k = labels_bags_c[k]
            labels_instance_c_k = labels_instance_c[k]
            bags_k = bags[k]
            for r in range(nRep):
                # Creation of nFolds splits
                kf = KFold(n_splits=nFolds, shuffle=True, random_state=r)
                fold = 0
                for train_index, test_index in kf.split(labels_bags_c_k):
                    labels_bags_c_train, labels_bags_c_test = \
                        getTest_and_Train_Sets(labels_bags_c,train_index,test_index)
                    bags_train, bags_test = \
                        getTest_and_Train_Sets(bags_k,train_index,test_index)
                    _ , labels_instance_c_test = \
                        getTest_and_Train_Sets(labels_instance_c_k,train_index,test_index)
                    gt_instances_labels_stack = np.hstack(np.array(labels_instance_c_test))
                    classifier=trainMIL(bags_train, labels_bags_c_train,method,opt)
                    pred_bag_labels, pred_instance_labels = classifier.predict(bags_test, instancePrediction=True)
                    perfObj[k,r,fold,:]=getClassifierPerfomance(pred_instance_labels,gt_instances_labels_stack)
                    perfObjB[k,r,fold,:]=getClassifierPerfomance(pred_bag_labels,labels_bags_c_test)
                    fold += 1
    else:
        perfObj=np.empty((nRep,nFolds,numMetric))
        perfObjB=np.empty((nRep,nFolds,numMetric))
        for r in range(nRep):
            # Creation of nFolds splits
            kf = KFold(n_splits=nFolds, shuffle=True, random_state=r)
            fold = 0
            for train_index, test_index in kf.split(labels_bags_c):
                labels_bags_c_train, labels_bags_c_test = \
                    getTest_and_Train_Sets(labels_bags_c,train_index,test_index)
                bags_train, bags_test = \
                    getTest_and_Train_Sets(bags,train_index,test_index)
                _ , labels_instance_c_test = \
                    getTest_and_Train_Sets(labels_instance_c,train_index,test_index)
                gt_instances_labels_stack = np.hstack(np.array(labels_instance_c_test))
                classifier=trainMIL(bags_train, labels_bags_c_train,method,opt)
                pred_bag_labels, pred_instance_labels = classifier.predict(bags_test, instancePrediction=True)
                perfObj[r,fold,:]=getClassifierPerfomance(pred_instance_labels,gt_instances_labels_stack)
                perfObjB[r,fold,:]=getClassifierPerfomance(pred_bag_labels,labels_bags_c_test)
                fold += 1
    perf=getMeanPref(perfObj)
    perfB=getMeanPref(perfObjB)
    return(perf,perfB)
 
def getClassifierPerfomance(y_true, y_pred):
    """
    This function compute 4 differents metrics :
    metrics = [f1Score,UAR,aucScore,accuracyScore]
        
    """
    f1Score = f1_score(y_true, y_pred) 
    # F1-score is the harmonic mean between precision and recall.
    aucScore = auc(y_true, y_pred) 
    # Compute Area Under the Curve (AUC) using the trapezoidal rule
    UAR = recall_score(y_true, y_pred,average='macro') 
    # Calculate metrics for each label, and find their unweighted mean. 
    # This does not take label imbalance into account. :
    # ie UAR = unweighted average recall (of each class)
    accuracyScore = accuracy_score(y_true, y_pred) # Accuracy classification score
    metrics = [f1Score,UAR,aucScore,accuracyScore] 
    return(metrics)
    
def trainMIL(bags_train, labels_bags_c_train,method,opt):
    
    return(0)
       
def getMeanPref(perfO=None):
    if  len(perfO.shape)==4:
        mean = np.mean(perfO,axis=[0,1,2])
        std = np.std(perfO,axis=[0,1,2])
    if  len(perfO.shape)==3:
        mean = np.mean(perfO,axis=[0,1])
        std = np.std(perfO,axis=[0,1])
    return([mean,std])
    

def getDataset(dataset=None):

    if 'SIVALfull'==dataset:
        Dataset = ExtractSIVAL()
    if 'SIVAL'==dataset:
        Dataset = ExtractSubsampledSIVAL() # Subsampled version
    if 'Birds'==dataset:
        Dataset = ExtractBirds()
    if 'Newsgroups'==dataset:
        Dataset = ExtractNewsgroups()

    return Dataset 
    
if __name__ == '__main__':
    pass
    
    
#@function
#def normalizeDataSet(D=None,DT=None,opt=None,*args,**kwargs):
#    varargin = normalizeDataSet.varargin
#    nargin = normalizeDataSet.nargin
#
#    if isfield(opt,'dataNormalization'):
#        if cellarray(['std']) == opt.dataNormalization:
#            # X is a dataset with row entries
#            u=mean(([[D.X],[DT.X]]))
## mainTestFunction.m:164
#            s=std(([[D.X],[DT.X]])) + eps
## mainTestFunction.m:165
#            um=repmat(u,size(D.X,1),1)
## mainTestFunction.m:166
#            sm=repmat(s,size(D.X,1),1)
## mainTestFunction.m:167
#            D.X = copy((D.X - um) / sm)
## mainTestFunction.m:168
#            um=repmat(u,size(DT.X,1),1)
## mainTestFunction.m:169
#            sm=repmat(s,size(DT.X,1),1)
## mainTestFunction.m:170
#            DT.X = copy((DT.X - um) / sm)
## mainTestFunction.m:171
#        else:
#            if cellarray(['var','variance']) == opt.dataNormalization:
#                # X is a dataset with row entries
#                u=mean(([[D.X],[DT.X]]))
## mainTestFunction.m:175
#                s=var(([[D.X],[DT.X]])) + eps
## mainTestFunction.m:176
#                um=repmat(u,size(D.X,1),1)
## mainTestFunction.m:177
#                sm=repmat(s,size(D.X,1),1)
## mainTestFunction.m:178
#                D.X = copy((D.X - um) / sm)
## mainTestFunction.m:179
#                um=repmat(u,size(DT.X,1),1)
## mainTestFunction.m:180
#                sm=repmat(s,size(DT.X,1),1)
## mainTestFunction.m:181
#                DT.X = copy((DT.X - um) / sm)
## mainTestFunction.m:182
#            else:
#                if cellarray(['0-1']) == opt.dataNormalization:
#                    ma=max(([[D.X],[DT.X]]))
## mainTestFunction.m:185
#                    mi=min(([[D.X],[DT.X]]))
## mainTestFunction.m:186
#                    mam=repmat(ma,size(D.X,1),1)
## mainTestFunction.m:187
#                    mim=repmat(mi,size(D.X,1),1)
## mainTestFunction.m:188
#                    D.X = copy((D.X - mim) / (mam - mim))
## mainTestFunction.m:189
#                    mam=repmat(ma,size(DT.X,1),1)
## mainTestFunction.m:190
#                    mim=repmat(mi,size(DT.X,1),1)
## mainTestFunction.m:191
#                    DT.X = copy((DT.X - mim) / (mam - mim))
## mainTestFunction.m:192
#    
#    return D,DT
    
if __name__ == '__main__':
    pass
    