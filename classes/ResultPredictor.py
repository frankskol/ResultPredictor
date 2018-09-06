"""This function utilizes a values and results list to guess the result for
a list with result unknown. It will use 10 different classifications 
and pick the result that comes from the biggest number of classifications. 
"""


def mostProbable(trainingValues, trainingResults, predict, predictAnswer = None, returnInfo = False):
    assert (len(trainingValues) == len(trainingResults)), "Values and results have different sizes"
    if predictAnswer is not None:
        assert (returnInfo), "There should be no return info, but accuracy info is requested"
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
    
    if returnInfo:
        infoString = ''
        #print results
        infoString += ''.join(['Result for {}: {}\n'.format(classifierNamesDict[index], prediction)
         for index, prediction in enumerate(predictions)])
        if predictAnswer is not None:
            #Getting accuracies
            accuracies = [metrics.accuracy_score(prediction, predictAnswer) * 100
                           for prediction in predictions]
            
            #print accuracies
            infoString += ''.join(['\nAccuracy for {}: {}%'.format(classifierNamesDict[index], accuracy) 
                for index, accuracy in enumerate(accuracies)])
            infoString += '\n'
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
    
    if returnInfo:        
        if predictAnswer is not None:
            infoString += ('\nMost accurate predictors: {}'.format(bestPredictors))        
        #get most frequent one
        infoString += ('\nMost probable result: {}'.format(result))

    return result if not returnInfo else (result, infoString)
    
