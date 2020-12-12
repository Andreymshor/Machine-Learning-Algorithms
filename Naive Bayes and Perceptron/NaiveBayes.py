import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import time
from collections import Counter
pd.options.mode.chained_assignment = None
def NBC(mergedTrainData, mergedTestData):
    attrDict = constructDict(mergedTrainData)
    
    priorAndMLEDicts = computeProbabilities(mergedTrainData, attrDict)
   
    priorDict = priorAndMLEDicts[0]
    MLEDict = priorAndMLEDicts[1]
    results = accuracy(priorAndMLEDicts, mergedTestData, attrDict)
    
    squaredLoss = results[1]
    zeroOneLoss = 1 - results[ 0]
    testAccuracy = results[0]
    

    print(f'ZERO-ONE LOSS={zeroOneLoss}')
    print(f'SQUARED LOSS={squaredLoss} Test Accuracy={testAccuracy}')
    return zeroOneLoss, squaredLoss, testAccuracy


def evaluation(mergedData):
    percentages = [1,10,50]
    averageZeroOneLossList = []
    averageSquaredLossList = []
    averageDefaultError = []
    for percent in percentages:
        zeroOneLossList = []
        squaredLossList = []
        defaultErrorList = []
        testAccuracyList = []
        trainingSizeList = []
        
        for i in range(10):
            data = mergedData.copy(deep=False)
            fraction = percent * 1.0 / 100
            trainData = data.sample(frac=fraction)
            trainingSizeList.append(len(trainData.index))
            rowIndexList = []
            
            rowIndexList = trainData.index.values.tolist()
            data = data.drop(rowIndexList)
            
            testdata = data
            defaultResults = defaultAccuracy(trainData, testdata)
            defaultErrorList.append(defaultResults[0])
            zeroOneLoss, squaredLoss, testAccuracy = NBC(trainData, testdata)
            
            zeroOneLossList.append(zeroOneLoss)
            squaredLossList.append(squaredLoss)
            testAccuracyList.append(testAccuracy)
        

        averageZeroOneLossList.append(sum(zeroOneLossList) * 1.0 / 10)
        averageSquaredLossList.append(sum(squaredLossList) * 1.0 / 10)
        averageDefaultError.append(sum(defaultErrorList) * 1.0 / 10)
        print(f"Percent of overall Data used to train Naive Bayes: {percent}%")
        print(f'Average Default Error: {sum(defaultErrorList) * 1.0/10}')
        print(f"Average Zero-One Loss: {sum(zeroOneLossList) * 1.0 / 10}")
        print(f"Average squared Loss: {sum(squaredLossList) * 1.0 / 10}")
        print(f"Average test Accuracy: {sum(testAccuracyList) * 1.0 / 10}")
        
        print()
    plt.plot(percentages, averageDefaultError, label = 'Default Error')
    plt.plot(percentages, averageZeroOneLossList, label = 'Zero One Loss')
    plt.title("Training Size(%) vs Average Zero One Loss")
    plt.ylabel(" Average Zero One Loss")
    plt.xlabel("Training Size(%)")
    plt.legend()
    plt.savefig("ZeroOneLoss")
    plt.clf()

    
    plt.plot(percentages, averageSquaredLossList)
    plt.title("Training Size vs Average Squared Loss")
    plt.ylabel("Average Squared Loss")
    plt.xlabel("Training Size(%)")
    
    plt.savefig("SquaredLoss")
    plt.clf()        
    

def accuracy(priorAndMLEDicts, mergedTestData, attrDict):
    numRows = len(mergedTestData.index)
    count = 0
    predictedProbabilities = []
    squaredLoss = 0
    for i in range(numRows):
        rowDict = buildRowDictionary(mergedTestData.iloc[i])
        fullDict = buildRowDictionary(mergedTestData.iloc[i])
        result = predict(priorAndMLEDicts, rowDict, attrDict)
        predictedLabel = result[0]
        
        
        if predictedLabel == fullDict['survived']:
            squaredLoss += (1 - result[1])**2
            count += 1
        else:
            squaredLoss += result[1] ** 2
    
    squaredLoss /= numRows
    return [count * 1.0 / numRows, squaredLoss]


def predict(priorAndMLEDicts, rowDict, attrDict):
    priorDict = priorAndMLEDicts[0]
    MLEDict = priorAndMLEDicts[1]
    rowDict.pop('survived', None)
    # Alive
    alive = priorDict["Alive"]
    for attr, val in rowDict.items():
        currAttr = attr
        currVal = val
        if currAttr != "Age" and currAttr != "Fare" and currAttr != "relatives":
            keyName = currAttr + " = " + str(currVal * 1.0) + " | " + "Alive"
            probability = MLEDict.get(keyName)
            if probability:
                
                alive *= MLEDict[keyName]
            else:
                alive *= 1 * 1.0 / (len(attrDict[currAttr]))
        else:
            potentialIntervals = attrDict[currAttr]
            for interval in potentialIntervals:
                if interval.left <= currVal < interval.right:
                    currVal = interval
                    break
            keyName = currAttr + " = " + str(currVal) + " | " + "Alive"
            probability = MLEDict.get(keyName)
            if probability:
                alive *= MLEDict[keyName]
            else:
                alive *= 1 * 1.0 / (len(attrDict[currAttr]))
    
    # Dead
    dead = priorDict["Dead"]
    for attr, val in rowDict.items():
        currAttr = attr
        currVal = val
        if currAttr != "Age" and currAttr != "Fare" and currAttr != "relatives":
            keyName = currAttr + " = " + str(currVal * 1.0) + " | " + "Dead"
            probability = MLEDict.get(keyName)
            if probability:
                dead *= MLEDict[keyName]
            else:
                dead *= 1 * 1.0 / (len(attrDict[currAttr]))
            
        else:
            potentialIntervals = attrDict[currAttr]
            for interval in potentialIntervals:
                if interval.left <= currVal < interval.right:
                    currVal = interval
                    break
            keyName = currAttr + " = " + str(currVal) + " | " + "Dead"
            probability = MLEDict.get(keyName)
            if probability:
                dead *= MLEDict[keyName]
            else:
                dead *= 1 * 1.0/ (len(attrDict[currAttr]))

    probAlive = alive / (alive + dead)
    probDead = dead / (alive + dead)
    
    if probAlive >= probDead:
        return [1, probAlive]
    return [0, probDead]


def computeProbabilities(mergedData, attrDict):
    
    priorDict = getPriorProbabilities(mergedData)
    MLEDict = getMLEstimates(mergedData, attrDict)

    return [priorDict, MLEDict]


def getPriorProbabilities(mergedData):
    numRows = len(mergedData.index)
    deadData = mergedData[mergedData['survived'] < .5]
    aliveData = mergedData[mergedData['survived'] > .5]

    probabilitesDict = dict()

    probabilitesDict['Alive'] = len(aliveData.index) * 1.0 / numRows
    probabilitesDict['Dead'] = len(deadData.index) * 1.0 / numRows
    return probabilitesDict


def getMLEstimates(mergedData, attrDict):
    MLEDict = dict()

    for attribute, classes in attrDict.items():
        currAttr = attribute
        currClasses = classes

        for value in currClasses:
            if currAttr != "Age" and currAttr != "Fare" and currAttr != "relatives":
                results = MLECatHelper(mergedData, currAttr, value, currClasses)
                
                keyName = currAttr + " = " + str(value) + " | " + "Dead"
                MLEDict[keyName] = results[0]
                
                keyName = currAttr + " = " + str(value) + " | " + "Alive"
                MLEDict[keyName] = results[1]
            else: 
                results = MLEConHelper(mergedData, currAttr, value.left, value.right, currClasses)

                keyName = currAttr + " = " + str(value) + " | " + "Dead"
                MLEDict[keyName] = results[0]

                keyName = currAttr + " = " + str(value) + " | " + "Alive"
                MLEDict[keyName] = results[1]    
                 
    return MLEDict
        
def MLEConHelper(mergedData, attribute, left, right, currClasses):
    # returns 2 values in list format, spot 0 = P(attr = val | dead), spot 1 = P(attr = val | alive)

    deadData = mergedData[mergedData['survived'] < .5]
    aliveData = mergedData[mergedData['survived'] > .5]
    numDead = len(deadData.index)
    numAlive = len(aliveData.index)

    deadAttr = deadData[attribute]
    aliveAttr = aliveData[attribute]

    count = 0 
    for val in deadAttr: 
        if left <= val < right:
            count += 1
    deadProb = (count + 1) * 1.0 / (numDead + len(currClasses))

    count = 0
    for val in aliveAttr:
        if left <= val < right:
            count += 1
    aliveProb = (count + 1) * 1.0 / (numAlive + len(currClasses))

    return [deadProb, aliveProb]


def MLECatHelper(mergedData, attribute, value, currClasses): 
    # returns 2 values in list format, spot 0 = P(attr = val | dead), spot 1 = P(attr = val | alive)
    deadData = mergedData[mergedData['survived'] < .5]
    aliveData = mergedData[mergedData['survived'] > .5]

    numDead = len(deadData.index)
    numAlive = len(aliveData.index)

    deadAttr = deadData[attribute]
    aliveAttr = aliveData[attribute]

    count = 0
    for val in deadAttr:
        if val == value:
            count += 1

    probDead = (count + 1) * 1.0 / (numDead + len(currClasses))

    count = 0
    for val in aliveAttr:
        if val == value:
            count += 1
    
    probAlive = (count + 1) * 1.0 / (numAlive + len(currClasses))

    return [probDead, probAlive]

def buildRowDictionary(data):
    rowDict = dict()
    rowDict["Pclass"] = data[0]
    rowDict["Sex"] = data[1]
    rowDict["Age"] = data[2]
    rowDict["Fare"] = data[3]
    rowDict["Embarked"] = data[4]
    rowDict["relatives"] = data[5]
    rowDict["IsAlone"] = data[6]
    rowDict["survived"] = data[7]

    return rowDict

def constructDict(mergedData):
    attrDict = dict()
    for attribute in mergedData:
        if attribute != "Age" and attribute != "Fare" and attribute != "relatives":
            attrDict[attribute] = np.unique(mergedData[attribute])
        else:
            attrDict[attribute] = binMaker(attribute, mergedData)
    attrDict.pop("survived", None)
    return attrDict

def binMaker(attribute, mergedData):
    numDp = len(attribute)
    numBins = math.ceil(numDp ** .5)
    return pd.cut(mergedData[attribute], numBins, right=False).value_counts().index.tolist()


def cleanData(data):
    columns = list(data)
    for col in columns:
        values = data[col].value_counts().index.tolist()
        # print(values[0])
        data[col] = data[col].replace(np.nan, values[0])
    return data


def mergeData(data, label):
   return pd.concat([data, label], axis=1) 

def mostCommonLabel(train_label):
    numZeros = 0
    numOnes = 0
    for i in train_label['survived']:
        if i == 1:
            numOnes += 1
        else:
            numZeros += 1
    
    mostCommon = 0
    if numOnes > numZeros:
        mostCommon = 1
    return mostCommon

def defaultAccuracy(mergedData, mergedTestData):
    count = 0
    pLabel = mostCommonLabel(mergedData)
    numRows = len(mergedTestData.index)

    for i in range(numRows):
        currRow = mergedTestData.iloc[i]
        if pLabel == currRow['survived']:
            count += 1
    
    accuracy = count * 1.0 / numRows
    zeroOneLoss = 1 - accuracy
    # print(f"ZERO ONE LOSS: {zeroOneLoss}, Accuracy: {accuracy}")
    return [zeroOneLoss, accuracy]

if __name__ == "__main__":
    import sys
    trainDataPath = sys.argv[1]
    trainLabelPath = sys.argv[2]
    train_data = pd.read_csv(trainDataPath)
    train_label = pd.read_csv(trainLabelPath)

    train_data = cleanData(train_data)
    mergedTrainData = mergeData(train_data, train_label)
    
    testDataPath = sys.argv[3]
    testLabelPath = sys.argv[4]
    test_data = pd.read_csv(testDataPath)
    test_label = pd.read_csv(testLabelPath)
    
    test_data = cleanData(test_data)
    mergedTestData = mergeData(test_data, test_label)
    

    # defaultAccuracy(mergedTrainData)
    # print()
    # evaluation(mergedTrainData)
    NBC(mergedTrainData, mergedTestData)
    
    