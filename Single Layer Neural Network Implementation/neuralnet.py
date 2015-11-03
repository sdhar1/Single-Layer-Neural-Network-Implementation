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
    for k in range(e):
        for curr in range(len(listOfSubsets)):
#             correctOutputs = 0
            for i in range(len(listOfSubsets)):
                if i == curr :
                    continue
                else :
                    (w0,weights)=updateWeights(listOfSubsets[i], w0, weights)
#             for ind in range(len(listOfSubsets[curr])) :
#                 if listOfSubsets[ind].attrValues[-1] == predictions[ind] :
#                     correctOutputs += 1

    predictions = makePredictions(trainingInstances,w0,weights)
    correctOutputs = 0 
    for instance in trainingInstances: 
          
        print "Fold:",instance.fold,"Predicted Class:",'Mine' if predictions[trainingInstances.index(instance)] else 'Rock',"Actual Class:",'Mine' if instance.attrValues[-1] else 'Rock',"Confidence:",output(w0,weights,instance.attrValues)
        if instance.attrValues[-1] == predictions[trainingInstances.index(instance)]:
            correctOutputs+=1
    print "Total correct =",correctOutputs,"out of",len(trainingInstances)
    
#     print '\n\n\n',w0,weights

main()

    
    