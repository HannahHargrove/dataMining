import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn.metrics import *

# Read in the csv
q3train = pd.read_csv('q3_train.csv', sep= ',')

# set the categorical data
trainIn = pd.get_dummies(q3train.loc[:, 'Is_Home_or_Away':'Media'])
trainOut = q3train.Label

# Read in the csv
data = pd.read_csv('q3_test.csv', sep=',')

# set the test categorical data
testIn = pd.get_dummies(data.loc[:, 'Is_Home_or_Away':'Media'])
testOut = data.Label

naiveBay = GaussianNB()
le = preprocessing.LabelEncoder()
le.fit(trainOut)
trainOut = le.transform(trainOut)
naiveBay.fit(trainIn, trainOut)

# prediction
prediction = naiveBay.predict(testIn)
le = preprocessing.LabelEncoder()
le.fit(testOut)
testOut = le.transform(testOut)
print(type(testOut))

print(prediction)
print(classification_report(testOut, prediction))
# end
