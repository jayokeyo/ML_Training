'''The difference between the validation, test, and train sets is very confusing. In this project the train, valid, and test datasets are provided
hence no need for train_test_split. The project uses files from sklearn
to train different ML algorithms. The validation dataset is used to determine
the most accurate algorithm which is then applied on the test dataset.'''

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
valid = pd.read_csv('valid.csv')
Accuracy_Score = {}
Classifier = [SVC,MultinomialNB,LogisticRegression,DecisionTreeClassifier]

highest_accuracy = 0

for classifier in Classifier:
    clas = classifier()
    model = make_pipeline(CountVectorizer(),classifier())
    model.fit(train['text'],train['label'])
    Accuracy = model.score(train['text'],train['label'])
    Accuracy_Score[str(clas)] = Accuracy
    if Accuracy > highest_accuracy:
        highest_accuracy = Accuracy
        most_accurate = model

most_accurate.fit(valid['text'],valid['label'])
Accuracy = model.score(valid['text'],valid['label'])
print('Accuracy on validation data = ',Accuracy)

if Accuracy > 0.85:
    most_accurate.fit(test['text'],test['label'])
    test['predicted label'] = most_accurate.predict(test['text'])

print(Accuracy_Score)
print(test)
mat = confusion_matrix(test['label'],test['predicted label'])
print(mat)
report = classification_report(test['label'],test['predicted label'])
print(report)
