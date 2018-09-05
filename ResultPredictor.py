"""This class calls a function that utilizing a values and results list will try to 
guess the result for a list with result unknown. It will use 6 different classifications 
and pick the result that comes from the biggest number of classifications. 
The class includes 1 example set for determining gender out of height, weight and shoe size
"""

# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39], [177, 70, 40], [159, 55, 37], [171, 75, 42], 
     [181, 85, 43], [169, 73, 37], [177, 76, 43], [176, 65, 40]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male', 'female', 'male', 'male']
predict = [[169, 73, 37], [177, 76, 43]]



def mostProbable(trainingValues, trainingResults, predict, predictAnswer = None):
    assert (len(trainingValues) == len(trainingResults)), "Values and results have different sizes"
    if predictAnswer is not None:
        assert (len(predict) == len(predictAnswer)), "Prediction and answer have different sizes"
    
    #imports
    from sklearn import tree
    from sklearn import ensemble
    from sklearn import neighbors
    from sklearn import metrics
    from sklearn import linear_model
    from sklearn import gaussian_process
    from sklearn import svm
    from sklearn import neural_network
    from sklearn import naive_bayes
    from sklearn import discriminant_analysis
    from collections import Counter
    from collections import defaultdict
    import numpy
    import unittest
    
    #Creating classifiers
    classifiers = [tree.DecisionTreeClassifier(), svm.SVC(), gaussian_process.GaussianProcessClassifier(),
                   neural_network.MLPClassifier(), naive_bayes.GaussianNB(), linear_model.Perceptron(),
                   ensemble.RandomForestClassifier(), neighbors.KNeighborsClassifier(), 
                   ensemble.AdaBoostClassifier(), discriminant_analysis.QuadraticDiscriminantAnalysis()]
    classifierNames = ['DecisionTree', 'SVC', 'GaussianProcess', 'MLPC', 'GaussianNB', 'Perceptron',
                       'RandomForest', 'KNeighbors', 'AdaBoost', 'QuadraticDiscriminant']
    classifierNamesDict = {value: key for value, key in enumerate(classifierNames)}    
    # Training them on the data
    classifiers = [classifier.fit(trainingValues, trainingResults) for classifier in classifiers]
    #Predicting with the given data
    predictions = [classifier.predict(predict) for classifier in classifiers ]
    
    
    #print results
    [print('Result for {}: {}'.format(classifierNamesDict[index], prediction)) 
     for index, prediction in enumerate(predictions)]
    print('')
    if predictAnswer is not None:
        #Getting accuracies
        accuracies = [metrics.accuracy_score(prediction, predictAnswer) * 100
                       for prediction in predictions]
        
        #print accuracies
        [print('Accuracy for {}: {}%'.format(classifierNamesDict[index], accuracy)) 
            for index, accuracy in enumerate(accuracies)]
        print('')
        #get best one
        max = numpy.max(accuracies)
        accuracyArray = numpy.array(accuracies)
        bestPredictors = []

    
    
    uniqueResults = defaultdict(int)
    for element in predictions:
        uniqueResults[tuple(element)] += 1
    generalResult = Counter(uniqueResults).most_common(1)
    result, occurences = generalResult[0]
    
    for i, name in classifierNamesDict.items():
        if predictAnswer is not None:
            if accuracyArray[i] == max:
                bestPredictors.append(classifierNamesDict[i])
            
    if predictAnswer is not None:
        print('Most accurate predictors: {}'.format(bestPredictors))        
    #get most frequent one
    print('\nMost probable result: {}'.format(result))
    return result
    
mostProbable(X, Y, predict, ['female', 'male'])
