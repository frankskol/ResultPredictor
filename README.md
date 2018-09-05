# ResultPredictor
This program uses existing data to predict the result of a given set via machine learning

# How?
The function utilizes 10 different classifiers from the sklearn package. 
It trains all of them with the given data, and then uses each of them to predict
the result of the value given. It then groups up the different predictions and 
the one which the most classifiers gave is chosen.
Optionally, one can get the statistics of the test:
  - Individual predictions 
  - Accuracies
  - Best classifiers
  
# todo
  -Make it possible to configure the classifiers via optional arguments
  -Increase documentation
  -Add tests and wider examples
  -Make the package overall more usable
