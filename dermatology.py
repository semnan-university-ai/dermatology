#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Author : Amir Shokri
# github link : https://github.com/amirshnll/dermatology
# dataset link : http://archive.ics.uci.edu/ml/datasets/Dermatology
# email : amirsh.nll@gmail.com


# In[2]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


# In[3]:


data = pd.read_csv('dermatology_data.csv', header=None)


# In[4]:


data = data.replace(to_replace="?", method='ffill')


# In[5]:


data.describe()


# In[6]:


properties = data[data.columns[:34]]
target = data[data.columns[34]]
scaler = MinMaxScaler()
scaled_x = scaler.fit_transform(properties)


# In[7]:


target.value_counts().plot.pie()


# In[8]:


pca = PCA(n_components=15)
reduced_x = pca.fit_transform(scaled_x)


# In[9]:


X_train, X_test, y_train, y_test = train_test_split(reduced_x, target, test_size=0.3, random_state=0)


# In[10]:


from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


# In[11]:


gnb = GaussianNB()
mlp = MLPClassifier(hidden_layer_sizes=(100, 100))
knn = KNeighborsClassifier(n_neighbors=5)
dt = DecisionTreeClassifier()
regressor = LogisticRegression()


# In[12]:


gnb.fit(X_train, y_train)
y_predgnb = gnb.predict(X_test)

mlp.fit(X_train, y_train)
y_predmlp = mlp.predict(X_test)

knn.fit(X_train, y_train)
y_predknn = knn.predict(X_test)

dt.fit(X_train, y_train)
y_preddt = dt.predict(X_test)

regressor.fit(X_train, y_train)
y_predregressor = regressor.predict(X_test)


# In[13]:


print('gnb f1: ', f1_score(y_test, y_predgnb, average='micro'))
print('gnb accuracy: ', accuracy_score(y_test, y_predgnb))

print('mlp f1: ', f1_score(y_test, y_predmlp, average='micro'))
print('mlp accuracy: ', accuracy_score(y_test, y_predmlp))

print('knn f1: ', f1_score(y_test, y_predgnb, average='micro'))
print('knn accuracy: ', accuracy_score(y_test, y_predknn))

print('decision tree f1: ', f1_score(y_test, y_predgnb, average='micro'))
print('decision tree accuracy: ', accuracy_score(y_test, y_preddt))

print('logistic regression f1: ', f1_score(y_test, y_predgnb, average='micro'))
print('logistic regression accuracy: ', accuracy_score(y_test, y_predregressor))

