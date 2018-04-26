
# coding: utf-8

# In[143]:


import pandas as pd
import numpy as np


# In[144]:


X_train = pd.read_csv("data/X_Train.csv")
Y_train = pd.read_csv("data/Y_Train.csv")


# In[145]:


X_test = pd.read_csv("data/X_Test.csv")


# In[146]:


from skmultilearn.problem_transform import BinaryRelevance
from sklearn import tree


# In[147]:


classifier = BinaryRelevance(classifier = tree.DecisionTreeClassifier(), require_dense = [False, True])

# train
classifier.fit(X_train.values, Y_train.values)


# In[152]:


# predict
predictions = classifier.predict(X_test.values)


# In[154]:


np.savetxt("data/dtree_test.csv", predictions.toarray(), delimiter=",")


# In[153]:


predictions.count_nonzero()


# In[141]:


X_test.values.shape

