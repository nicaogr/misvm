# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 15:49:39 2018

@author: gonthier
"""
    
@function
def mainTestFunction(allMethods=None,dataset=None,*args,**kwargs):
    varargin = mainTestFunction.varargin
    nargin = mainTestFunction.nargin

    close_('all')
    clc
    ## LOAD PACKAGES
    addpath(genpath('CodePackage/CarbonneauToolBox'))
    addpath('CodePackage/LIBSVM/libsvm-3.20/matlab')
    run('CodePackage/vlfeat/toolbox/vl_setup')
    addpath(genpath('CodePackage/prtools'))
    addpath(genpath('CodePackage/milToolbox'))
    addpath(genpath('CodePackage/dd_tools'))
    addpath(genpath('CodePackage/EMD'))
    ## PARAMETERS
    
    if ischar(allMethods):
        allMethods=cellstr(allMethods)
# mainTestFunction.m:17
    
    dataset,fn=getParForDataset(dataset,nargout=2)
# mainTestFunction.m:19
    ## LOAD DATASET
    
    load(fn)
    disp('=============================================================')
    disp(concat(['= DATA SET ACQUIRED: ',fn]))
    disp('-------------------------------------------------------------')
    ## TEST METHODS ON THE DATA SET
    
    for i in arange(1,length(allMethods)).reshape(-1):
        tic
        method=allMethods[i]
# mainTestFunction.m:32
        disp(concat(['= Method: ',method]))
        # get config for test
        opt=getMethodConfig(method,dataset,'single')
# mainTestFunction.m:36
        if exist('DT','var'):
            # perform normalization if necessary
            D,DT=normalizeDataSet(D,DT,opt,nargout=2)
# mainTestFunction.m:40
            perf,perfB=performExperimentWithTestSet(D,DT,method,opt,nargout=2)
# mainTestFunction.m:41
        else:
            # perform normalization if necessary
            D=normalizeDataSet(D,[],opt)
# mainTestFunction.m:44
            perf,perfB=performExperimentWithCrossVal(D,method,opt,nargout=2)
# mainTestFunction.m:45
        ## Results
        disp('=============================================================')
        disp(concat(['= ',method]))
        disp('-------------------------------------------------------------')
        disp('- instances')
        disp(concat(['AUC: ',num2str(perf.AUC)]))
        disp(concat(['UAR: ',num2str(perf.UAR)]))
        disp('- bags')
        disp(concat(['AUC: ',num2str(perfB.AUC)]))
        disp(concat(['UAR: ',num2str(perfB.UAR)]))
        toc
        disp('-------------------------------------------------------------')
        # save results
        fn=concat(['Results/',dataset,'-',method,'-',date])
# mainTestFunction.m:65
        save(fn,'perf','perfB')
    
    return
    
if __name__ == '__main__':
    pass
    
    
@function
def performExperimentWithCrossVal(D=None,method=None,opt=None,*args,**kwargs):
    varargin = performExperimentWithCrossVal.varargin
    nargin = performExperimentWithCrossVal.nargin

    nRep=10
# mainTestFunction.m:76
    nFolds=10
# mainTestFunction.m:77
    perfObj=cell(nRep,nFolds)
# mainTestFunction.m:78
    perfObjB=cell(nRep,nFolds)
# mainTestFunction.m:79
    for r in arange(1,nRep).reshape(-1):
        BagPerFoldList=divideBagsInFolds(nFolds,D)
# mainTestFunction.m:82
        for fold in arange(1,nFolds).reshape(-1):
            disp(concat(['---- Performing Fold ',num2str(fold),' of rep ',num2str(r)]))
            tic
            # create training and test datasets
            TRD,TED=getTrainingAndTestDatasets(fold,nFolds,BagPerFoldList,D,nargout=2)
# mainTestFunction.m:87
            pred=trainAndTestMIL(TRD,TED,method,opt)
# mainTestFunction.m:89
            perfObj[r,fold]=getClassifierPerfomance(pred.PL,pred.TL,pred.SC)
# mainTestFunction.m:91
            perfObjB[r,fold]=getClassifierPerfomance(pred.PLB,pred.TLB,pred.SCB)
# mainTestFunction.m:92
            toc
    
    perf=getMeanPref(perfObj)
# mainTestFunction.m:96
    perfB=getMeanPref(perfObjB)
# mainTestFunction.m:97
    return perf,perfB
    
if __name__ == '__main__':
    pass
    
    
@function
def performExperimentWithTestSet(D=None,DT=None,method=None,opt=None,*args,**kwargs):
    varargin = performExperimentWithTestSet.varargin
    nargin = performExperimentWithTestSet.nargin

    pred=trainAndTestMIL(D,DT,method,opt)
# mainTestFunction.m:103
    ## Compute Performances
    perf=getClassifierPerfomance(pred.PL,pred.TL,pred.SC)
# mainTestFunction.m:106
    perfB=getClassifierPerfomance(pred.PLB,pred.TLB,pred.SCB)
# mainTestFunction.m:107
    return perf,perfB
    
if __name__ == '__main__':
    pass
    
    
@function
def getMeanPref(perfO=None,*args,**kwargs):
    varargin = getMeanPref.varargin
    nargin = getMeanPref.nargin

    fName=fieldnames(perfO[1,1])
# mainTestFunction.m:114
    for i in arange(1,length(fName)).reshape(-1):
        setattr(meanPerf,fName[i],concat([0,0]))
# mainTestFunction.m:117
    
    for i in arange(1,length(fName)).reshape(-1):
        table=zeros(size(perfO))
# mainTestFunction.m:121
        for j in arange(1,size(perfO,1)).reshape(-1):
            for k in arange(1,size(perfO,2)).reshape(-1):
                table[j,k]=getattr(perfO[j,k],(fName[i]))
# mainTestFunction.m:124
        m=mean(table,2)
# mainTestFunction.m:128
        setattr(meanPerf,fName[i],concat([mean(ravel(table)),std(m)]))
# mainTestFunction.m:129
    
    return meanPerf
    
if __name__ == '__main__':
    pass
    
    
@function
def getParForDataset(dataset=None,*args,**kwargs):
    varargin = getParForDataset.varargin
    nargin = getParForDataset.nargin

    if cellarray(['musk1']) == lower(dataset):
        fn=concat(['Datasets/Musk/Musk1'])
# mainTestFunction.m:140
    else:
        if cellarray(['musk2']) == lower(dataset):
            fn=concat(['Datasets/Musk/Musk2'])
# mainTestFunction.m:143
        else:
            if cellarray(['tiger','fox','elephant']) == lower(dataset):
                fn=concat(['Datasets/FoxElephantTiger/',dataset])
# mainTestFunction.m:146
            else:
                if cellarray(['test']) == lower(dataset):
                    fn=concat(['Datasets/Test/test'])
# mainTestFunction.m:149
                else:
                    if cellarray(['testcv']) == lower(dataset):
                        fn=concat(['Datasets/Test/testCV'])
# mainTestFunction.m:152
    
    return dataset,fn
    
if __name__ == '__main__':
    pass
    
    
@function
def normalizeDataSet(D=None,DT=None,opt=None,*args,**kwargs):
    varargin = normalizeDataSet.varargin
    nargin = normalizeDataSet.nargin

    if isfield(opt,'dataNormalization'):
        if cellarray(['std']) == opt.dataNormalization:
            # X is a dataset with row entries
            u=mean(concat([[D.X],[DT.X]]))
# mainTestFunction.m:164
            s=std(concat([[D.X],[DT.X]])) + eps
# mainTestFunction.m:165
            um=repmat(u,size(D.X,1),1)
# mainTestFunction.m:166
            sm=repmat(s,size(D.X,1),1)
# mainTestFunction.m:167
            D.X = copy((D.X - um) / sm)
# mainTestFunction.m:168
            um=repmat(u,size(DT.X,1),1)
# mainTestFunction.m:169
            sm=repmat(s,size(DT.X,1),1)
# mainTestFunction.m:170
            DT.X = copy((DT.X - um) / sm)
# mainTestFunction.m:171
        else:
            if cellarray(['var','variance']) == opt.dataNormalization:
                # X is a dataset with row entries
                u=mean(concat([[D.X],[DT.X]]))
# mainTestFunction.m:175
                s=var(concat([[D.X],[DT.X]])) + eps
# mainTestFunction.m:176
                um=repmat(u,size(D.X,1),1)
# mainTestFunction.m:177
                sm=repmat(s,size(D.X,1),1)
# mainTestFunction.m:178
                D.X = copy((D.X - um) / sm)
# mainTestFunction.m:179
                um=repmat(u,size(DT.X,1),1)
# mainTestFunction.m:180
                sm=repmat(s,size(DT.X,1),1)
# mainTestFunction.m:181
                DT.X = copy((DT.X - um) / sm)
# mainTestFunction.m:182
            else:
                if cellarray(['0-1']) == opt.dataNormalization:
                    ma=max(concat([[D.X],[DT.X]]))
# mainTestFunction.m:185
                    mi=min(concat([[D.X],[DT.X]]))
# mainTestFunction.m:186
                    mam=repmat(ma,size(D.X,1),1)
# mainTestFunction.m:187
                    mim=repmat(mi,size(D.X,1),1)
# mainTestFunction.m:188
                    D.X = copy((D.X - mim) / (mam - mim))
# mainTestFunction.m:189
                    mam=repmat(ma,size(DT.X,1),1)
# mainTestFunction.m:190
                    mim=repmat(mi,size(DT.X,1),1)
# mainTestFunction.m:191
                    DT.X = copy((DT.X - mim) / (mam - mim))
# mainTestFunction.m:192
    
    return D,DT
    
if __name__ == '__main__':
    pass
    