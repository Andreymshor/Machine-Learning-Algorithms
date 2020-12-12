# from ashorID3 import * # renamed file ashor-ID3 to ashorID3 for this part as otherwise it would not recognize.
# import matplotlib.pyplot as plt

# # First Graph

# def specificModels(trainData, trainLabels, model):
#     listOfValues = [2,4,6,8,10,12,14,16]
#     bestValue = [0,0]
#     # using mode
#     # trainData = clean_Data_Median(trainData)
#     mergedData = mergeData(trainData, trainLabel)
#     mergedData = clean_Data_Remove(mergedData)
#     avgNumNodes = []
#     avgValidationAccuracy = []
#     avgTrainAccuracy = []
#     print("Removing values that are missing ")
#     for value in listOfValues:
#         if model == "depth":
#             # 0th element is trainAccuracies, 1st is validation accuracies, 2nd is numNodes for each model
#             result = evaluation(mergedData,5,"depth", value, None)
#             avgTrainAccuracy.append(sum(result[0]) / len(result[0]))
#             avgValidationAccuracy.append(sum(result[1]) / len(result[1]))
#             avgNumNodes.append(sum(result[2]) / len(result[2]))
#             if avgValidationAccuracy[-1] > bestValue[1]:
#                 bestValue[0] = value
#                 bestValue[1] = avgValidationAccuracy[-1]
#         elif model == "minSplit":
#             result = evaluation(mergedData,5, "minSplit", None, value)
#             avgTrainAccuracy.append(sum(result[0]) / len(result[0]))
#             avgValidationAccuracy.append(sum(result[1]) / len(result[1]))
#             avgNumNodes.append(sum(result[2]) / len(result[2]))
#             if avgValidationAccuracy[-1] > bestValue[1]:
#                 bestValue[0] = value
#                 bestValue[1] = avgValidationAccuracy[-1]

#     print(f"The best depth and validation accuracy for data cleaned by removing datapoints that have missing values is: "
#           f"{bestValue[0], bestValue[1]}")
    
#     plt.plot(listOfValues, avgNumNodes, 'ro')
#     plt.xlabel('Depth')
#     plt.ylabel('Average Number of Nodes')
#     plt.title('Depth vs Average Number of Nodes')
#     plt.show()

# trainData = pd.read_csv("titanic-train.data")
# trainLabel = pd.read_csv("titanic-train.label")
# # this is for mode
# specificModels(trainData, trainLabel, "depth")
