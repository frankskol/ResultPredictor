'''
Created on 05.09.2018

@author: frank
'''
import ResultPredictor as rp
# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39], [177, 70, 40], [159, 55, 37], [171, 75, 42], 
     [181, 85, 43], [169, 73, 37], [177, 76, 43], [176, 65, 40]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male', 'female', 'male', 'male']
predict = [[169, 73, 37], [177, 76, 43]]


print(rp.mostProbable(X, Y, predict))
print(rp.mostProbable(X, Y, predict, ['female', 'male'], True))
    