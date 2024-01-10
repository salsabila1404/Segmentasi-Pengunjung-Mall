#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
os.environ["OMP_NUM_THREADS"] = "1"

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans


# In[3]:


df = pd.read_csv("Mall_Customers.csv")
df.head()


# In[4]:


genders = df.Gender.value_counts()
sns.set_style("darkgrid")
plt.figure(figsize=(10,4))
sns.barplot(x=genders.index, y=genders.values)
plt.show()


# In[5]:


age18_25 = df.Age[(df.Age <= 25) & (df.Age >= 18)]
age26_35 = df.Age[(df.Age <= 35) & (df.Age >= 26)]
age36_45 = df.Age[(df.Age <= 45) & (df.Age >= 36)]
age46_55 = df.Age[(df.Age <= 55) & (df.Age >= 46)]
age55above = df.Age[df.Age >= 56]

x = ["18-25","26-35","36-45","46-55","55+"]
y = [len(age18_25.values),len(age26_35.values),len(age36_45.values),len(age46_55.values),len(age55above.values)]

plt.figure(figsize=(15,5))
sns.barplot(x=x, y=y, palette="rocket")
plt.title("Total Customers and Ages")
plt.xlabel("Ages")
plt.ylabel("Total Customers")
plt.show()


# In[6]:


x = df.iloc[:, [3,4]].values
print("Dataset di kolom 3 dan 4 = \n", x)


# In[7]:


#Elbow Method
wcss = []

print('WCSS (inertia)=')
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, n_init=10, random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
    print(f'Number of clusters: {i}, WCSS: {kmeans.inertia_}')


# In[8]:


plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('wcss')
plt.show()


# In[9]:


#Penerapan Kmeans
kmeansmodel = KMeans(n_clusters=5, n_init=10, random_state=0)
y_kmeans = kmeansmodel.fit_predict(x)
print('Fit Predict = \n', y_kmeans)


# In[10]:


plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0,1], s=100, c = 'red', label = 'Cluster 1')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1,1], s=100, c = 'green', label = 'Cluster 2')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2,1], s=100, c = 'purple', label = 'Cluster 3')
plt.scatter(x[y_kmeans == 3, 0], x[y_kmeans == 3,1], s=100, c = 'blue', label = 'Cluster 4')
plt.scatter(x[y_kmeans == 4, 0], x[y_kmeans == 4,1], s=100, c = 'black', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title("Clusters of Customers")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.show()


# In[11]:


#Penerapan Kmeans
kmeansmodel = KMeans(n_clusters=2, n_init=10, random_state=0)
y_kmeans = kmeansmodel.fit_predict(x)
print('Fit Predict = \n', y_kmeans)


# In[12]:


plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0,1], s=100, c = 'red', label = 'Cluster 1')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1,1], s=100, c = 'green', label = 'Cluster 2')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title("Clusters of Customers")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.show()


# In[13]:


#Penerapan Kmeans
kmeansmodel = KMeans(n_clusters=3, n_init=10, random_state=0)
y_kmeans = kmeansmodel.fit_predict(x)
print('Fit Predict = \n', y_kmeans)


# In[14]:


plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0,1], s=100, c = 'red', label = 'Cluster 1')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1,1], s=100, c = 'green', label = 'Cluster 2')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2,1], s=100, c = 'purple', label = 'Cluster 3')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title("Clusters of Customers")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.show()


# In[ ]:




