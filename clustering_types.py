'''This program was adopted from dphi. The code uses USArrest dataset
to demonstrate how hierarchical clustering is conducted and the different
types of heirarchical clustering.'''

import warnings
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

# ignoring any warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

crime = pd.read_csv("https://raw.githubusercontent.com/dphi-official/Datasets/master/US_violent_crime.csv")
crime.head(5)

#Understanding basic information about the dataset
crime.shape
crime.info()
crime.describe()
crime.isnull().sum()
plt.figure(figsize=(20,5))
crime.groupby('State')['Murder'].max().plot(kind='bar')
plt.figure(figsize=(20,5))
crime.groupby('State')['Assault'].max().plot(kind='bar')
plt.figure(figsize=(20,5))
crime.groupby('State')['Rape'].max().plot(kind='bar')
plt.figure(figsize=(20,5))
crime.groupby('State')['UrbanPop'].max().plot(kind='bar')
data = crime.iloc[:,1:].values

#Using sklearn.preprocessing.StandardScaler() to standardize the data points.
scaler= StandardScaler()
scaled_data = scaler.fit_transform(data)

#Generating dendrograms for the dataset using different distance matrices (single, average, complete, and ward).
plt.figure(figsize=(20,5))
plt.title("Crime Rate Dendograms")
dend = sch.dendrogram(sch.linkage(scaled_data, method='single'))
plt.xlabel('Crime Rate')
plt.ylabel('Euclidean distances')

plt.figure(figsize=(20,5))
plt.title("Crime Rate Dendograms")
dend = sch.dendrogram(sch.linkage(scaled_data, method='complete'))
plt.xlabel('Crime Rate')
plt.ylabel('Euclidean distances')

plt.figure(figsize=(20,5))
plt.title("Crime Rate Dendograms")
dend = sch.dendrogram(sch.linkage(scaled_data, method='average'))
plt.xlabel('Crime Rate')
plt.ylabel('Euclidean distances')

plt.figure(figsize=(20,8))
dendrogram = sch.dendrogram(sch.linkage(data, method = "ward"))
plt.title('Dendrogram')
plt.xlabel('Crime Rate')
plt.ylabel('Euclidean distances')
plt.show()

#Selecting the desired number of clusters and predicting labels for the different rows.
clusters = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage ='ward')
y_pred =clusters.fit_predict(data)
crime['cluster labels']= y_pred
print(crime[['State','cluster labels']])

#Using seaborn to display the result.
plt.figure(figsize=(10,5))
sns.boxplot(x='cluster labels', y='Murder', data=crime)
plt.figure(figsize=(10,5))
sns.boxplot(x='cluster labels', y='Rape', data=crime)
plt.figure(figsize=(10,5))
sns.boxplot(x='cluster labels', y='Assault', data=crime)

#Setting categories based on the cluster each state belongs to.
Safe_Zone= crime.groupby('cluster labels')['State'].unique()[0]
Safe_Zone
Danger_Zone= crime.groupby('cluster labels')['State'].unique()[1]
Danger_Zone
Moderate_Zone= crime.groupby('cluster labels')['State'].unique()[2]
Moderate_Zone
plt.figure(figsize=(10,5))
plt.scatter(data[y_pred==0, 0], data[y_pred==0, 1], s=100, c='red', label ='Safe_Zone')
plt.scatter(data[y_pred==1, 0], data[y_pred==1, 1], s=100, c='blue', label ='Danger_Zone')
plt.scatter(data[y_pred==2, 0], data[y_pred==2, 1], s=100, c='green', label ='Moderate_Zone')
plt.legend()
plt.show()
