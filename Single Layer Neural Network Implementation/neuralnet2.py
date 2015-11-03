'''
Created on Oct 27, 2015

@author: satyamdhar
'''
import sys
import arff
import math
import random

trainingData = arff.load(open(sys.argv[1], 'rb'))
eta = float(sys.argv[3])
n = float(sys.argv[2])
e = int(sys.argv[4])
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def output(w0,weights, attrValues):
    Sum = 0
    for index in range(len(weights)):
        Sum = Sum + weights[index]*attrValues[index]
    return sigmoid(Sum+w0)

class trainInst:
    fold=0
    def __init__(self, instance):
        self.attrValues=instance
      
    def setAttrVal(self,AttributeIndex):
        self.value=self.attrValues[AttributeIndex]
        
    def display(self):
        print self.attrValues


class Attribute:
    def __init__(self,attribute,attributeIndex):
        self.name=attribute[0]   
        self.index=attributeIndex

def divideByClass(trainingInstances):
    class0instances=[]
    class1instances=[]
    for instance in trainingInstances:
        if instance.attrValues[-1] == 0 :
            class0instances.append(instance)
        else :
            class1instances.append(instance)
    return (class0instances,class1instances)
# def updateWeights(w0,weights,setInstances):

def makeSubsets(class0instances, class1instances, numOfNegativesInEach, numOfPositivesInEach):
    subset=[]
    listOfSubsets=[]
    lenClass0 = len(class0instances)
    lenClass1 = len(class1instances)
    #numOfFolds = 0
    while len(listOfSubsets)+1<=n and lenClass0 >= numOfNegativesInEach and lenClass1 >= numOfPositivesInEach :
#         print "lengths", lenClass0, lenClass1
        elementsToRemoveIn0=[]
        elementsToRemoveIn1=[]
        subsetIndexList = random.sample(range(0,lenClass0),numOfNegativesInEach)
        for i in subsetIndexList :
            subset.append(class0instances[i])
            class0instances[i].fold=len(listOfSubsets)+1
            elementsToRemoveIn0.append(class0instances[i])
#             print i,
#             class0instances.remove(class0instances[i])
        subsetIndexList = random.sample(range(0,lenClass1),numOfPositivesInEach)
        for i in subsetIndexList :
            subset.append(class1instances[i])
            class1instances[i].fold=len(listOfSubsets)+1
            elementsToRemoveIn1.append(class1instances[i])
#             class1instances.remove(class1instances[i])
#         elementsToRemoveIn0.sort(key=None, reverse=True)
#         elementsToRemoveIn1.sort(key=None, reverse=True)
        
#         print "fold is",elementsToRemoveIn0[0].fold
        for instance in elementsToRemoveIn0:
            class0instances.remove(instance)
        for instance in elementsToRemoveIn1:
            class1instances.remove(instance)
        lenClass0 = len(class0instances)
        lenClass1 = len(class1instances)
#         print "lengths", lenClass0, lenClass1

        listOfSubsets.append(subset)
        #numOfFolds+=1
        subset=[]
    if lenClass0 > 0 :
        for i in range(len(class0instances)):
#             print listOfSubsets[i]
            listOfSubsets[i].append(class0instances[i])
            class0instances[i].fold=i+1
    if lenClass1 > 0 :
        for i in range(len(class1instances)) :
#             print i
            listOfSubsets[len(listOfSubsets)-1-i].append(class1instances[i])
            class1instances[i].fold=len(listOfSubsets)-i
    return listOfSubsets
    
def updateWeights(trainingInstances,w0,weights):
    for instance in trainingInstances:
        op = output(w0, weights, instance.attrValues)
        errorDelta = op * (1 - op) * (instance.attrValues[-1] - op)
        for index in range(len(weights)):
            weights[index] = weights[index] + eta*errorDelta*instance.attrValues[index]
        w0 = w0 + eta*errorDelta
    return (w0,weights)

def makePredictions(testInstances, w0 ,weights):
    predictions=[]
    for instance in testInstances :
        op = output(w0, weights, instance.attrValues)
        if op > 0.5 :
            predictions.append(1)
        else:
            predictions.append(0)
    return predictions

def main():
    trainingInstances=[]    #list of training instance objects
    for instance in trainingData['data']:
        objInstance=trainInst(instance)
        trainingInstances.append(objInstance)
    
    attributes=[]           #list of training instance objects
    for attribute in trainingData['attributes']:
        attributeIndex = trainingData['attributes'].index(attribute)
        attributeObj = Attribute(attribute,attributeIndex)
        attributes.append(attributeObj)
    
#     Y = []

    for instance in trainingInstances:
#         if isinstance(instance.attrValues[-1], String):
        if instance.attrValues[-1] == 'Mine':
            instance.attrValues[-1] = 1
        else :
            instance.attrValues[-1] = 0
    
    w0 = 0.1                                       #bias parameter initially set to 0.1
    weights = []
    for i in range(len(attributes)-1):
        weights.append(0.1)                        #weights initially set to 0.1
    
    
    
    (class0instances,class1instances) = divideByClass(trainingInstances)
#     print len(class0instances),len(class1instances) 
    numOfinstancesInSampleSubsetWithClass0 = int((len(class0instances)/n))
    numOfinstancesInSampleSubsetWithClass1 = int((len(class1instances)/n))
    #     numOfinstancesInSampleSubsetWithClass0 = numOfinstancesInSampleSubsetWithClass1 = (len(trainingInstances)/n)/2
    
#     print "numbers",numOfinstancesInSampleSubsetWithClass0,numOfinstancesInSampleSubsetWithClass1
    listOfSubsets=makeSubsets(class0instances,class1instances,numOfinstancesInSampleSubsetWithClass0,numOfinstancesInSampleSubsetWithClass1)
#     for instance in class0instances:
#         print "fold is",instance.fold
    
#     for subset in listOfSubsets:
#         print len(subset),
    testaccuracy = 0
    trainaccuracy =0
    for k in range(e):
        for curr in range(len(listOfSubsets)):
            for i in range(len(listOfSubsets)):
                if i == curr :
                    continue
                else :
                    (w0,weights)=updateWeights(listOfSubsets[i], w0, weights)
    
    listroc = []
    num_neg=0.0
    num_pos=0.0
    for instance in trainingInstances :
        if instance.attrValues[-1] == 1:
            num_pos+=1
        else:
            num_neg+=1
        op = output(w0, weights, instance.attrValues)
        rc = ROCpoint()
        rc.y = instance.attrValues[-1]
        rc.o = 1 if op > 0.5 else 0
        rc.c = op
        listroc.append(rc)
    points=[]
    listroc.sort(key=lambda x: x.c, reverse=True)
    for rc in listroc:
        rc.display()
    TP=0
    FP=0
    lastTP=0
    for i in range(1,len(listroc)):
        if listroc[i-1].y!=listroc[i].y and listroc[i].y == 0 and TP>lastTP:
            FPR=FP/num_neg
            TPR=TP/num_pos
            points.append([TPR,FPR])
            lastTP=TP
        if listroc[i].y == 1:
            TP+=1
        else:
            FP+=1
    FPR=FP/num_neg
    TPR=TP/num_pos
#     print [FPR,TPR]
    points.append([TPR,FPR])
    
    for point in points:
        print point[0]
    print 'hahahahah'
    for point in points:
        print point[1]
    

class ROCpoint:
    def __init__(self):
        self.y=0
        self.o=0
        self.c=0.0
    def display(self):
        print [self.y,self.o, self.c]
            
    
#     listroc[i]!=listroc[i-1] and 
# for curr in range(len(listOfSubsets)):
#         truePositives = 0
#         falsePositives = 0
#         actualPositives = 0
#         trueNegatives = 0
#         falseNegatives = 0
#         predictions = makePredictions(listOfSubsets[curr], w0, weights)
#         for instance in listOfSubsets[curr]:
# #             if instance.attrValues[-1] == 1:
#                 actualPositives += 1
#                 if instance.attrValues[-1] == predictions[listOfSubsets[curr].index(instance)]:
#                     truePositives += 1
#                 else:
#                     falseNegatives +=1
#             if instance.attrValues[-1] == 0:
#                 if instance.attrValues[-1] == predictions[listOfSubsets[curr].index(instance)]:
#                     trueNegatives += 1
#                 else:
#                     falsePositives += 1
#         print 'tpr =',truePositives/float(truePositives+falseNegatives)
#         print 'fpr =',falsePositives/float(trueNegatives+falsePositives)
        
#         print 'false positves =',falsePositives
#         print 'actual positives = ', actualPositives
        
            
    
#     allOthersList = []  
#     curr = 0              
#     for curr in range(len(listOfSubsets)):
#         for other in range(len(listOfSubsets)):
#             if other == curr:
#                 continue
#             else :
#                 for element in listOfSubsets[other]:
#                     allOthersList.append(element)
#         predictions = makePredictions(listOfSubsets[curr],w0,weights)
#         print predictions
#         correctOutputs=0
#         for ind in range(len(listOfSubsets[curr])) :
#             if listOfSubsets[curr][ind].attrValues[-1] == predictions[ind] :
# #                 print listOfSubsets[curr][ind].attrValues[-1]," ",
#                 correctOutputs += 1
# #         print "correct outputs =",correctOutputs,"out of",len(listOfSubsets[curr])
#         testaccuracy = testaccuracy + correctOutputs/float(len(listOfSubsets[curr]))
#         
#         predictions = makePredictions(allOthersList,w0,weights)
#         print predictions
#         correctOutputs=0
#         for ind in range(len(allOthersList)) :
#             if allOthersList[ind].attrValues[-1] == predictions[ind] :
# #                 print listOfSubsets[curr][ind].attrValues[-1]," ",
#                 correctOutputs += 1
# #         print "correct outputs =",correctOutputs,"out of",len(allOthersList)
#         trainaccuracy = trainaccuracy + correctOutputs/float(len(allOthersList))
#     trainaccuracy /= n
#     testaccuracy /= n
#     print 'test acc =',testaccuracy
#     print 'train acc =',trainaccuracy
#     
#        for curr in range(len(listOfSubsets)):
#         
# #         print predictions
#         correctOutputsTest = 0
#         correctOutputsTrain = 0
#         for set in range(len(listOfSubsets)):
#             if set == curr:
#                 predictions = makePredictions(listOfSubsets[curr],w0,weights)
#                 for ind in range(len(listOfSubsets[curr])) :
#                     if listOfSubsets[curr][ind].attrValues[-1] == predictions[ind] :
# #                         print listOfSubsets[curr][ind].attrValues[-1]," ",
#                         correctOutputsTest += 1
# #                 print "correct outputs =",correctOutputsTest,"out of",len(listOfSubsets[curr])
#                 testaccuracy = testaccuracy + correctOutputsTest/float(len(listOfSubsets[curr]))
# #         print totalCorrect
#             else :
#                 
#                 for ind in range(len(listOfSubsets[set]))  :
#                     if listOfSubsets[set][ind].attrValues[-1] == predictions[ind] :
# #                         print listOfSubsets[set][ind].attrValues[-1]," ",
#                         correctOutputsTrain += 1
# #                 print "correct outputs =",correctOutputsTrain,"out of",len(listOfSubsets[curr])
#                 trainaccuracy = trainaccuracy + correctOutputsTrain/float(len(listOfSubsets[curr]))
#         testaccuracy /= n
#         trainaccuracy /= n
#     print 'test set acc =',testaccuracy
#     print 'train set acc =',trainaccuracy
#     
    
#     predictions = makePredictions(trainingInstances,w0,weights)
#     correctOutputs = 0 
#     for instance in trainingInstances: 
#           
# #         print "Fold:",instance.fold,"Predicted Class:",'Mine' if predictions[trainingInstances.index(instance)] else 'Rock',"Actual Class:",'Mine' if instance.attrValues[-1] else 'Rock',"Confidence:",output(w0,weights,instance.attrValues)
#         if instance.attrValues[-1] == predictions[trainingInstances.index(instance)]:
#             correctOutputs+=1
#     print "Total correct =",correctOutputs,"out of",len(trainingInstances)
    
#     print '\n\n\n',w0,weights

main()

    
    