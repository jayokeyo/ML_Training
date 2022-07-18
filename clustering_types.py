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
crime.columns=['State','Murder','Assault','UrbanPop','Rape']

fig=plt.figure(figsize=(20,80))

#Understanding basic information about the dataset
crime.shape
crime.info()
crime.describe()
crime.isnull().sum()
fig.add_subplot(5,2,1)
crime.groupby('State')['Murder'].max().plot(kind='bar')
plt.title('Murder per State')
fig.add_subplot(5,2,2)
crime.groupby('State')['Assault'].max().plot(kind='bar')
plt.title('Assault per State')
fig.add_subplot(5,2,3)
crime.groupby('State')['Rape'].max().plot(kind='bar')
plt.title('Rape per State')
fig.add_subplot(5,2,4)
crime.groupby('State')['UrbanPop'].max().plot(kind='bar')
plt.title('Urban Population per State')

data = crime.iloc[:,1:].values

#Using sklearn.preprocessing.StandardScaler() to standardize the data points.
scaler= StandardScaler()
scaled_data = scaler.fit_transform(data)

#Generating dendrograms for the dataset using different distance matrices (single, average, complete, and ward).
fig.add_subplot(5,2,5)
plt.title("Crime Rate Dendograms")
dend = sch.dendrogram(sch.linkage(scaled_data, method='single'))
plt.xlabel('Crime Rate')
plt.ylabel('Euclidean distances')

fig.add_subplot(5,2,6)
plt.title("Crime Rate Dendograms")
dend = sch.dendrogram(sch.linkage(scaled_data, method='complete'))
plt.xlabel('Crime Rate')
plt.ylabel('Euclidean distances')

fig.add_subplot(5,2,7)
plt.title("Crime Rate Dendograms")
dend = sch.dendrogram(sch.linkage(scaled_data, method='average'))
plt.xlabel('Crime Rate')
plt.ylabel('Euclidean distances')

fig.add_subplot(5,2,8)
dendrogram = sch.dendrogram(sch.linkage(data, method = "ward"))
plt.title('Dendrogram')
plt.xlabel('Crime Rate')
plt.ylabel('Euclidean distances')

#Selecting the desired number of clusters and predicting labels for the different rows.
clusters = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage ='ward')
crime['cluster labels'] =clusters.fit_predict(data)

#Setting categories based on the cluster each state belongs to.
Safe_Zone= crime.groupby('cluster labels')['State'].unique()[0]
Danger_Zone= crime.groupby('cluster labels')['State'].unique()[1]
Moderate_Zone= crime.groupby('cluster labels')['State'].unique()[2]
fig.add_subplot(5,2,9)
plt.scatter(data[crime['cluster labels']==0, 0], data[crime['cluster labels']==0, 1], s=100, c='red', label ='Safe_Zone')
plt.scatter(data[crime['cluster labels']==1, 0], data[crime['cluster labels']==1, 1], s=100, c='blue', label ='Danger_Zone')
plt.scatter(data[crime['cluster labels']==2, 0], data[crime['cluster labels']==2, 1], s=100, c='green', label ='Moderate_Zone')
plt.legend()

print(crime[['State','cluster labels']])