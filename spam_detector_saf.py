'''The difference between the validation, test, and train sets is very confusing. In this project the train, valid, and test datasets are provided
hence no need for train_test_split. The project uses files from sklearn
to train different ML algorithms. The validation dataset is used to determine
the most accurate algorithm which is then applied on the test dataset.'''

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
valid = pd.read_csv('valid.csv')
Accuracy_Score = {}
Classifier = [SVC,MultinomialNB,LogisticRegression,DecisionTreeClassifier]

cv = CountVectorizer()
features = cv.fit_transform(train['text'])
features_valid = cv.fit_transform(valid['text'])
features_test = cv.fit_transform(test['text'])

highest_accuracy = 0

for classifier in Classifier:
    model = classifier()
    model.fit(features,train['label'])
    Accuracy = model.score(features,train['label'])
    Accuracy_Score[str(model)] = Accuracy
    if Accuracy > highest_accuracy:
        highest_accuracy = Accuracy
        most_accurate = model

most_accurate.fit(features_valid,valid['label'])
Accuracy = model.score(features_valid,valid['label'])
print('Accuracy on validation data = ',Accuracy)

if Accuracy > 0.85:
    most_accurate.fit(features_test,test['label'])
    test['predicted label'] = most_accurate.predict(features_test)

print(Accuracy_Score)
print(test)
