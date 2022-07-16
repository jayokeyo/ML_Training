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
z_train = train['text']
y_train = train['label']
z_test = test['text']
y_test = test['label']
z_valid = valid['text']
y_valid = valid['label']
Classifier = [SVC,MultinomialNB,LogisticRegression,DecisionTreeClassifier]
cv = CountVectorizer()
features = cv.fit_transform(z_train)

highest_accuracy = 0

for classifier in Classifier:
    model = classifier()
    model.fit(features,y_train)
    features_valid = cv.transform(z_valid)
    Accuracy = round(model.score(features_valid,y_valid),5)*100
    Accuracy_Score[str(model)] = Accuracy
    if Accuracy > highest_accuracy:
        highest_accuracy = Accuracy
        most_accurate = model

most_accurate.fit(features,y_train)
features_test = cv.transform(z_test)
Accuracy = most_accurate.score(features_test,y_test)*100
print(Accuracy_Score)
print ('Most accurate classifier is ',most_accurate , ': Accuracy = ',round(Accuracy,5),'% on the test dataset')
