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
TinderSwindler = pd.read_csv('TinderSwindler.csv')
Unseen = pd.read_csv('Unseen.csv')
Accuracy_Score = {}
Classifier = [SVC,MultinomialNB,LogisticRegression,DecisionTreeClassifier]

highest_accuracy = 0

#Training the model and identifying the most accurate classifier.
for classifier in Classifier:
    clas = classifier()
    model = make_pipeline(CountVectorizer(),classifier())
    # label = 1 signifies the text is not a spam while 0 signifies the text is a spam.
    model.fit(train['text'],train['label'])
    Accuracy = model.score(train['text'],train['label'])
    Accuracy_Score[str(clas)] = Accuracy
    if Accuracy > highest_accuracy:
        highest_accuracy = Accuracy
        most_accurate = model

#Validating the model performance.
most_accurate.fit(valid['text'],valid['label'])
Accuracy = model.score(valid['text'],valid['label'])
print()
print('Accuracy on validation data = ',Accuracy)
print()

#Testing the model.
if highest_accuracy > 0.85:
    model.fit(train['text'],train['label'])
    test['predicted label'] = most_accurate.predict(test['text'])

print(Accuracy_Score)
print()
print(test)
print()

#Evaluating model's performance on the test dataset.
mat = confusion_matrix(test['label'],test['predicted label'])
print(mat)
print()
report = classification_report(test['label'],test['predicted label'])
print(report)
print()

#Predicting whether tweets on #TinderSwindler are spam messages or not.
TinderSwindler['predicted label'] = most_accurate.predict(TinderSwindler['text'])
del TinderSwindler['user_name']
del TinderSwindler['user_location']
del TinderSwindler['user_created']
del TinderSwindler['user_followers']
del TinderSwindler['user_verified']
del TinderSwindler['date']
del TinderSwindler['source']
del TinderSwindler['is_retweet']
print('label = 1 signifies the text is not a spam while 0 signifies the text is a spam.')
print()
print(TinderSwindler.head(20))
print()
print(TinderSwindler.tail(20))
print()
spam_count = TinderSwindler['predicted label'].value_counts()[0]
print('Number of spam tweets: ',spam_count)
print()

#Predicting whether texts on a independently created dataset are spam messages or not.
Unseen['predicted label'] = most_accurate.predict(Unseen['text'])
print(Unseen)
print()
