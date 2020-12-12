##############
# Name: Andrey Shor
# email: ashor@purdue.edu
# Date: 10/09/2020


import numpy as np
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

def entropy(freqs):
    all_freq = sum(freqs)
    entropy = 0
    for fq in freqs:
        prob = fq * 1.0 / all_freq
        if abs(prob) > 1e-8:
            entropy += -prob * np.log2(prob)
    return entropy


def infor_gain(before_split_freqs, after_split_freqs):
    gain = entropy(before_split_freqs)
    overall_size = sum(before_split_freqs)
    for freq in after_split_freqs:
        ratio = sum(freq) * 1.0 / overall_size
        gain -= ratio * entropy(freq)
    return gain


def generateThreshold(attr):
    attr.sort()
    thresholdList = []
    for i in range(0, len(attr) - 1):
        avg = (attr[i] + attr[i + 1]) * 1.0 / 2.0
        if min(attr) < avg < max(attr):
            thresholdList.append(avg)

    return thresholdList


def generateDictionary(trainData):
    # print(trainData.to_string())
    thresholdDict = dict()
    for col in trainData:
        if not (col in thresholdDict):
            thresholdDict[col] = generateThreshold(list(trainData[col]))
    return thresholdDict


def mergeData(trainData, trainLabel):
    return pd.concat([trainData, trainLabel], axis=1)


def calculateSurvived(mergedData):
    survived = 0
    dead = 0
    for person in mergedData["survived"]:
        if person == 0:
            dead += 1
        else:
            survived += 1
    deadAliveList = [survived, dead]
    return deadAliveList


def optimalInfogainAndThreshold(mergedData, attr, thresholdList):
    beforeSplitList = calculateSurvived(mergedData)
    infoGainList = []

    if len(thresholdList) == 0:
        return -1

    for threshold in thresholdList:
        survivedOrDeadList = []
        df1 = mergedData[mergedData[attr] <= threshold]
        df2 = mergedData[mergedData[attr] > threshold]

        survivedOrDeadList.append(calculateSurvived(df1))
        survivedOrDeadList.append(calculateSurvived(df2))

        infoGainList.append([infor_gain(beforeSplitList, survivedOrDeadList), threshold])

    maxGainThreshold = max(infoGainList, key=lambda x: x[0])

    return [maxGainThreshold[0], maxGainThreshold[1], attr]


def majorityValue(mergedData):
    values = mergedData["survived"].value_counts().index.tolist()
    return values[0]


def printTree(root):
    if root.left_subtree is None and root.right_subtree is None:
        print(f"Leaf Label: {root.label}")
    if root.left_subtree is not None and root.right_subtree is not None:
        print(f"Root Attribute: {root.attribute}")
        print(f"Root Threshold: {root.threshold}")
        print(f"Root InfoGain: {root.infoGain}")
        printTree(root.left_subtree)
        printTree(root.right_subtree)
    if root.left_subtree is not None and root.right_subtree is None:
        print(f"Root Attribute: {root.attribute}")
        print(f"Root Threshold: {root.threshold}")
        print(f"Root InfoGain: {root.infoGain}")
        printTree(root.left_subtree)
    if root.right_subtree is not None and root.left_subtree is None:
        print(f"Root Attribute: {root.attribute}")
        print(f"Root Threshold: {root.threshold}")
        print(f"Root InfoGain: {root.infoGain}")
        printTree(root.right_subtree)


# part 0
def ID3(train_data, train_labels, currDepth=None, maxDepth=None, minSplit=None):
    mergedData = mergeData(train_data, train_labels)  # stores the entire dataset along with train labels
    trainDict = generateDictionary(train_data)  # dictionary that stores threshold lists for each attribute

    # print(len(train_data.columns))
    if len(train_data.columns) == 0:
        leaf_node = Node(None, None, None, None, None, None, True, majorityValue(mergedData))
        return leaf_node

    # minSplit Base Case:

    if minSplit is not None:
        numRows = len(mergedData.index)
        if numRows <= minSplit:
            leaf_node = Node(None, None, None, None, None, None, True, majorityValue(mergedData))
            return leaf_node

    # Depth Base Case

    if currDepth is not None and maxDepth is not None:
        if int(currDepth) >= int(maxDepth):
            leaf_node = Node(None, None, None, None, None, None, True, majorityValue(mergedData))
            return leaf_node

    # for col in mergedData.columns:
    #     print(col)
    # print(mergedData.head())

    if len(pd.unique(mergedData['survived'])) == 1:
        # print("terminated base case 1")
        label = pd.unique(mergedData['survived'])
        leaf_node = Node(None, None, None, None, None, None, True, label[0])
        return leaf_node

    # 1. use a for loop to calculate the infor-gain of every attribute
    optimalInfoGainList = []  # will store the optimal thresholds for each attribute and information gain
    noSplitList = []
    for attr in train_data:
        thresholdList = trainDict[attr]
        if optimalInfogainAndThreshold(mergedData, attr, thresholdList) == -1:
            noSplitList.append(attr)
        else:
            optimalInfoGainList.append(optimalInfogainAndThreshold(mergedData, attr, thresholdList))

    if len(noSplitList) == 7:
        leaf_node = Node(None, None, None, None, None, None, True, majorityValue(mergedData))
        return leaf_node
    # 2. pick the attribute that achieve the maximum infor-gain
    # 0th attribute = info_gain, 1st attribute = threshold, 2nd attribute = attr
    if len(optimalInfoGainList) == 0:
        leaf_node = Node(None, None, None, None, None, None, True, majorityValue(mergedData))
        return leaf_node
    bestAttr = max(optimalInfoGainList, key=lambda x: x[0])
    the_chosen_attribute = bestAttr[2]
    the_chosen_threshold = bestAttr[1]
    the_chosen_infogain = bestAttr[0]
    # 3. build a node to hold the data;
    # def __init__(self, l=None, r=None, attr=None, thresh=None, majorityVote=None, infogain=None, isLeafNode=False,
    #            label=None)
    current_node = Node(None, None, the_chosen_attribute, the_chosen_threshold, the_chosen_infogain, majorityValue(mergedData))

    # 4. split the data into two parts.
    leftMergedData = mergedData[mergedData[the_chosen_attribute] <= the_chosen_threshold]
    rightMergedData = mergedData[mergedData[the_chosen_attribute] > the_chosen_threshold]

    # drop the column
    leftMergedData = leftMergedData.drop(columns=the_chosen_attribute, axis=1)
    rightMergedData = rightMergedData.drop(columns=the_chosen_attribute, axis=1)

    listOfRemainingColumns = []
    for col in leftMergedData.columns:
        listOfRemainingColumns.append(col)


    # 5. call ID3() for the left parts of the data

    # Old approach
    # left_part_train_data = leftMergedData[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'relatives', 'IsAlone']]
    # left_part_train_label = leftMergedData["survived"]

    left_part_train_data = leftMergedData[listOfRemainingColumns]
    left_part_train_label = leftMergedData["survived"]
    left_part_train_data = left_part_train_data.drop(columns=["survived"], axis=1)

    left_subtree = None
    if currDepth is not None and maxDepth is not None:
        currDepth += 1
        # print("Entering Recursive Left condition")
        left_subtree = ID3(left_part_train_data, left_part_train_label, currDepth, maxDepth)
    elif minSplit is not None:
        left_subtree = ID3(left_part_train_data, left_part_train_label, None, None, minSplit)
    else:
        left_subtree = ID3(left_part_train_data, left_part_train_label)

    # 6. call ID3() for the right parts of the data.

    listOfRemainingColumns = []
    for col in rightMergedData.columns:
        listOfRemainingColumns.append(col)

    # Old approach
    # right_part_train_data = rightMergedData[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'relatives', 'IsAlone']]
    # right_part_train_label = rightMergedData['survived']

    right_part_train_data = rightMergedData[listOfRemainingColumns]
    right_part_train_label = rightMergedData['survived']
    right_part_train_data = right_part_train_data.drop(columns=["survived"], axis=1)

    right_subtree = None
    if currDepth is not None and maxDepth is not None:
        currDepth += 1
        right_subtree = ID3(right_part_train_data, right_part_train_label, currDepth, maxDepth)
    elif minSplit is not None:
        right_subtree = ID3(right_part_train_data, right_part_train_label, None, None, minSplit)
    else:
        right_subtree = ID3(right_part_train_data, right_part_train_label)

    current_node.left_subtree = left_subtree
    current_node.right_subtree = right_subtree
    return current_node


# Part 1

def evaluation(mergedData, mergedTestData, k=5, model=None, maxDepth=None, minSplit=None):
    # 1.shuffle dataset
    mergedData = mergedData.sample(frac=1)
    withHeldtestData = mergedTestData
    # 2. Split the dataset into k groups
    splitDf = np.array_split(mergedData, k)
    decisionTreeModels = []
    listOfTrainAccuracies = []
    listOfValidationAccuracies = []
    listOfNumberOfNodes = []
    for i in range(k):

        mergedTestData = splitDf[i]
        trainData = []

        for j in range(k):
            if not splitDf[j].equals(mergedTestData):  # takes into account duplicate case
                trainData.append(splitDf[j])

        # 3 merge all trainData pandas df together
        mergedTrainData = pd.DataFrame()
        for item in trainData:
            mergedTrainData = pd.concat([mergedTrainData, item], axis=0)

        unmergedTrainData = mergedTrainData[["Pclass", "Sex", "Age", "Fare", "Embarked", "relatives", "IsAlone"]]
        unmergedTrainLabel = mergedTrainData["survived"]

        trainModel = None
        if model == 'depth':
            trainModel = ID3(unmergedTrainData, unmergedTrainLabel, 1, maxDepth)
        elif model == 'minSplit':
            trainModel = ID3(unmergedTrainData, unmergedTrainLabel, None, None, int(minSplit))
        elif model == 'postPrune':
            trainModel = ID3(unmergedTrainData, unmergedTrainLabel)
            postPrune(trainModel, trainModel, accuracy(trainModel, mergedTestData), mergedTestData)
        else:
            trainModel = ID3(unmergedTrainData, unmergedTrainLabel)
        print(f"fold = {i + 1}, train set accuracy= {accuracy(trainModel, mergedTrainData) * 100:.2f}%, "
              f"validation set accuracy= {accuracy(trainModel, mergedTestData) * 100:.2f}%")

        listOfTrainAccuracies.append(accuracy(trainModel, mergedTrainData) * 100)
        listOfValidationAccuracies.append(accuracy(trainModel, mergedTestData) * 100)
        decisionTreeModels.append(trainModel)

    numRows = len(withHeldtestData.index)
    numCorrect = 0
    for treeModel in decisionTreeModels:
        listOfNumberOfNodes.append(countNodes(treeModel))

    for i in range(numRows):
        rowDict = buildRowDictionary(withHeldtestData.iloc[i])
        listOfLabels = []
        for treeModel in decisionTreeModels:
            predictedLabel = traverseTree(treeModel, rowDict)
            listOfLabels.append(predictedLabel)

        predictedLabel = countMostOccured(listOfLabels)
        if predictedLabel == rowDict['survived']:
            numCorrect += 1
    testAccuracy = numCorrect * 1.0 / numRows
    print(f"Test set accuracy= {testAccuracy * 100:.2f}%")

    # 0th element is trainAccuracies, 1st is validation accuracies, 2nd is numNodes for each model
    return (listOfTrainAccuracies, listOfValidationAccuracies, listOfNumberOfNodes)


def countMostOccured(listOfLabels):
    numZeros = 0
    numOnes = 0
    for i in listOfLabels:
        if i == 1:
            numOnes += 1
        else:
            numZeros += 1

    if numOnes > numZeros:
        return 1
    return 0


def postPrune(root, currRoot, currAccuracy, mergedTestData):
    if currRoot.isLeafNode == True:
        return

    postPrune(root, currRoot.left_subtree, currAccuracy, mergedTestData)
    postPrune(root, currRoot.right_subtree, currAccuracy, mergedTestData)

    leftSubTree = currRoot.left_subtree
    rightSubTree = currRoot.right_subtree


    currRoot.left_subtree = None
    currRoot.right_subtree = None
    currRoot.label = currRoot.majorityVote
    currRoot.isLeafNode = True

    if (accuracy(root, mergedTestData) < currAccuracy):
        currRoot.left_subtree = leftSubTree
        currRoot.right_subtree = rightSubTree
        currRoot.isLeafNode = False
        currRoot.label = None



def accuracy(trainedModel, testData):
    numRows = len(testData.index)
    # print(f"Number of rows is: {numRows}")
    numRight = 0
    for i in range(numRows):
        rowDict = buildRowDictionary(testData.iloc[i])

        predictedLabel = traverseTree(trainedModel, rowDict)
        # print(f"Predicted Label: {predictedLabel}")
        # print(f"Actual Label: {rowDict['survived']}")
        if rowDict['survived'] == predictedLabel:
            numRight += 1

    return numRight * 1.0 / numRows


def traverseTree(node, rowDict):
    label = 0
    if node.isLeafNode:  # Base case
        return node.label

    if rowDict[node.attribute] <= node.threshold:
        if node.left_subtree is not None:
            label = traverseTree(node.left_subtree, rowDict)

    if rowDict[node.attribute] > node.threshold:
        if node.right_subtree is not None:
            label = traverseTree(node.right_subtree, rowDict)

    return label


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


# Part 1 Cleaning dataset
# replace with most common data point
def clean_Data(trainData):
    columns = list(trainData)
    for col in columns:
        values = trainData[col].value_counts().index.tolist()
        # print(values[0])
        trainData[col] = trainData[col].replace(np.nan, values[0])
    return trainData

# Part 2 Analysis:

# def vanillaAnalysis(trainData, trainLabels):
#     uncleanedTrainData = trainData # use this to reset trainData

#     trainData = clean_Data(trainData)
#     mergedData = mergeData(trainData, trainLabel)
#     # Accuracies on cleaned data
#     print("Train Data cleaned using mode:")
#     evaluation(mergedData, )

#     trainData = uncleanedTrainData
#     trainData = clean_Data_Median(trainData)
#     mergedData = mergeData(trainData, trainLabel)
#     print()
#     print("Train data cleaned using median:")
#     evaluation(mergedData)

#     trainData = uncleanedTrainData
#     mergedData = mergeData(trainData, trainLabel)
#     trainData = clean_Data_Remove(mergedData)
#     print()
#     print("Train data cleaned by removing rows that have missing values:")
#     evaluation(mergedData)


def countNodes(root):
    numNodes = 1
    if root is None:
        return 0

    numNodes += countNodes(root.left_subtree)
    numNodes += countNodes(root.right_subtree)
    return numNodes

# assign median value
def clean_Data_Median(trainData):
    columns = list(trainData)
    for col in columns:
        tempArray = np.array(trainData[col])
        tempArray = tempArray[~(np.isnan(tempArray))]
        median = np.median(tempArray)
        trainData[col] = trainData[col].replace(np.nan, median)

    return trainData

# remove rows that have Nans
def clean_Data_Remove(mergedData):
    mergedData = mergedData.dropna()
    return mergedData


class Node(object):
    def __init__(self, l=None, r=None, attr=None, thresh=None, infogain = None, majorityVote = None, isLeafNode=False, label=None):
        self.left_subtree = l
        self.right_subtree = r
        self.attribute = attr
        self.threshold = thresh
        self.isLeafNode = isLeafNode
        self.infoGain = infogain
        self.majorityVote = majorityVote
        if self.isLeafNode:
            self.label = label


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='CS373 Homework2 Decision Tree')
    parser.add_argument('--trainFolder')
    parser.add_argument('--testFolder')
    parser.add_argument('--model')
    parser.add_argument('--crossValidK', type=int, default=5)
    parser.add_argument('--depth')
    parser.add_argument('--minSplit', type=int)
    args = parser.parse_args()
    print(args)

    trainPath = args.trainFolder
    testPath = args.testFolder

    trainDataFullPath = trainPath + "\\titanic-train.data"
    trainLabelFullPath = trainPath + "\\titanic-train.label"

    testDataFullPath = testPath + "\\titanic-test.data"
    testLabelFullPath = testPath + "\\titanic-test.label"

    
    trainData = pd.read_csv(trainDataFullPath)
    trainLabel = pd.read_csv(trainLabelFullPath)

    testData = pd.read_csv(testDataFullPath)
    testLabel = pd.read_csv(testLabelFullPath)


    trainData = clean_Data(trainData)
    testData = clean_Data(testData)

    mergedData = mergeData(trainData, trainLabel)
    mergedTestData = mergeData(testData, testLabel)

    if args.crossValidK is not None:
        if args.model == 'depth':
            evaluation(mergedData, mergedTestData, args.crossValidK, args.model, args.depth)
        elif args.model == 'minSplit':
            evaluation(mergedData, mergedTestData, args.crossValidK, args.model, None, args.minSplit)
        elif args.model == 'postPrune':
            evaluation(mergedData, mergedTestData, args.crossValidK, args.model, None, None)
        else:
            evaluation(mergedData, mergedTestData, args.crossValidK)
    else:
        evaluation(mergedData, mergedTestData)
