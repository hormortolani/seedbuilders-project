#!/usr/bin/env python
# coding: utf-8

# EXTRACTION WAS DONE BY BARRY BECKER FROM THE 1994 CENSUS DATABASE.
# PREDICTION TASK IS T DETERMINE WHETHER A PERSON MAKES OVER 5OK A YEAR

# In[1]:


##IMPORT NECESSARY LIBARIES 
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
import plotly.express as px
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


###IMPORT THE DATASET 
census_salary = pd.read_csv('C:/data/census_salary.csv')


# In[3]:


census_salary.head()


# In[4]:


census_salary.info()


# In[5]:


census_salary.isna().sum()


# In[6]:


census_salary.describe()


# In[7]:


census_salary.dtypes


# In[8]:


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder


# In[9]:


le = LabelEncoder()
census_salary['age'] = le.fit_transform(census_salary['age'])
census_salary['capital-gain'] = le.fit_transform(census_salary['capital-gain'])
census_salary['capital-loss'] = le.fit_transform(census_salary['capital-loss'])
census_salary['hours-per-week'] = le.fit_transform(census_salary['hours-per-week'])
census_salary['occupation'] = le.fit_transform(census_salary['occupation'])
census_salary['workclass'] = le.fit_transform(census_salary['workclass'])
census_salary['education-num'] = le.fit_transform(census_salary['education-num'])
census_salary['education'] = le.fit_transform(census_salary['education'])


# In[10]:


import seaborn as sns


# In[11]:


sns.barplot(x='salary', y='capital-gain', hue = 'sex', data = census_salary)


# In[12]:


census_salary.salary.value_counts()


# In[13]:


from sklearn.model_selection import train_test_split


# In[14]:


x = np.array(census_salary[["capital-gain", "capital-loss", "hours-per-week", "education", "education-num"]])
y = np.array(census_salary[["salary"]])


# In[15]:


xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.10, random_state = 42)


# In[16]:


census_salary.shape


# In[17]:


lr = LogisticRegression()


# In[18]:


lr.fit(xtrain,ytrain)


# In[19]:


from sklearn.metrics import accuracy_score
y_pred = lr.predict(xtest)


# In[20]:


accuracy_score(y_pred,ytest)*100


# In[21]:


y_pred


# In[22]:


lst_ = list(y_pred)


# In[23]:


lst_


# In[24]:


dict_ = {"predictions":lst_}
dict_.keys()


# In[25]:


result = pd.DataFrame(dict_, columns = dict_.keys())


# In[26]:


result


# In[ ]:





# In[ ]:




