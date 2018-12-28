#!/usr/bin/env python


import numpy as np

from misvmio import parse_c45, bag_set
import misvm
from ExtractBirds import ExtractBirds
from sklearn.model_selection import KFold

def main():
    # Load list of C4.5 Examples
    example_set = parse_c45('musk1')

    # Group examples into bags
    bagset = bag_set(example_set)

    # Convert bags to NumPy arrays
    # (The ...[:, 2:-1] removes first two columns and last column,
    #  which are the bag/instance ids and class label)
    bags = [np.array(b.to_float())[:, 2:-1] for b in bagset]
    labels = np.array([b.label for b in bagset], dtype=float)
    # Convert 0/1 labels to -1/1 labels
    labels = 2 * labels - 1

    # Spilt dataset arbitrarily to train/test sets
    train_bags = bags[10:]
    train_labels = labels[10:]
    test_bags = bags[:10]
    test_labels = labels[:10]

    # Construct classifiers
    classifiers = {}
    classifiers['SIL'] = misvm.SIL(kernel='linear', C=1.0)
    classifiers['MISVM'] = misvm.MISVM(kernel='linear', C=1.0, max_iters=50)
    classifiers['miSVM'] = misvm.miSVM(kernel='linear', C=1.0, max_iters=50)
    classifiers['NSK'] = misvm.NSK()
    classifiers['STK'] = misvm.STK()
    classifiers['MissSVM'] = misvm.MissSVM(kernel='linear', C=1.0, max_iters=10)
    classifiers['sMIL'] = misvm.sMIL(kernel='linear', C=1.0)
    classifiers['stMIL'] = misvm.stMIL(kernel='linear', C=1.0)
    classifiers['sbMIL'] = misvm.sbMIL(kernel='linear', eta=0.1, C=1.0)
    classifiers['MICA'] = misvm.MICA()
    

    # Train/Evaluate classifiers
    accuracies = {}
    for algorithm, classifier in list(classifiers.items()):
        classifier.fit(train_bags, train_labels)
        predictions = classifier.predict(test_bags)
        accuracies[algorithm] = np.average(test_labels == np.sign(predictions))

    for algorithm, accuracy in list(accuracies.items()):
        print('\n%s Accuracy: %.1f%%' % (algorithm, 100 * accuracy))

def getTest_and_Train_Sets(Data,indextrain,indextest):
    """
    Split the list of data in train and test set according to the index lists
    provided
    """
    DataTrain = [ Data[i] for i in indextrain]
    DataTest = [ Data[i] for i in indextest]
    return(DataTrain,DataTest)

def testBirds():
    Dataset = ExtractBirds()
    list_names,bags,labels_bags,labels_instance = Dataset
    
    nFolds=2
    r = 0
    for c_i,c in enumerate(list_names):
        # Loop on the different class, we will consider each group one after the other
        print("For class :",c)
        labels_bags_c = labels_bags[c_i]
        labels_instance_c = labels_instance[c_i]
        
        kf = KFold(n_splits=nFolds, shuffle=True, random_state=r)
        algorithm = 'SIL'
        for train_index, test_index in kf.split(labels_bags_c):
            labels_bags_c_train, labels_bags_c_test = \
                getTest_and_Train_Sets(labels_bags_c,train_index,test_index)
            bags_train, bags_test = \
                getTest_and_Train_Sets(bags,train_index,test_index)
            _ , labels_instance_c_test = \
                getTest_and_Train_Sets(labels_instance_c,train_index,test_index)
            classifier = misvm.SIL(kernel='linear', C=1.0)
            classifier.fit(bags_train, labels_bags_c_train)
#            predictions = classifier.predict(bags_test)
            pred_bag_labels, pred_instance_labels = classifier.predict(bags_test, instancePrediction=True)
            bag_accuracy = np.average(labels_bags_c_test == np.sign(pred_bag_labels))
            gt_instances_labels_stack = np.hstack(np.array(labels_instance_c_test))
            instances_accuracy = np.average(gt_instances_labels_stack == np.sign(pred_instance_labels))
            print('\n%s Bag Accuracy: %.1f%%' % (algorithm, 100 * bag_accuracy))
            print('\n%s Instance Accuracy: %.1f%%' % (algorithm, 100 * instances_accuracy))
            
if __name__ == '__main__':
#    main()
    testBirds()
