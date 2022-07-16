'''This program was adopted from dphi. The code uses wholesale_customer_data dataset
to demonstrate how hierarchical clustering is conducted.'''
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.cluster import AgglomerativeClustering as AC
import scipy.cluster.hierarchy as shc

#Read the csv document and print te first five rows.
data = pd.read_csv('https://raw.githubusercontent.com/dphi-official/Datasets/master/Wholesale_customers_data.csv')
x = data.head()
print(x)

#Scale the data to values between 0 and 1 to prevent items with large units from creating a bias on the algorithm. 
data_scaled = pd.DataFrame(normalize(data),columns=data.columns)
print(data_scaled)

#Create a grid for ploting the dendrogram.
plt.figure(figsize=(10,7))
plt.title('Dendograms')

#Plot the dendrogram within the grid.
dend = shc.dendrogram(shc.linkage(data_scaled,method='ward'))
#Draw a dotted line to mark the point delimiting the clusters.
plt.axhline(y=6,color='r',linestyle='--')

#Create two clusters on the dataset using euclidian distance as the distance matrix.
cluster = AC(n_clusters=2,affinity='euclidean',linkage='ward')
fit_data = pd.DataFrame(cluster.fit_predict(data_scaled))
print(fit_data)

#Use two columns to visualize the clusters and how they are distributed.
plt.figure(figsize=(10,7))
plt.scatter(data_scaled['Milk'],data_scaled['Grocery'],c=cluster.labels_)
