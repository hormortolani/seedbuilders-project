#!/usr/bin/env python
# coding: utf-8

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
credit_card = pd.read_csv('C:/data/Credit Card Customer Data.csv')


# In[3]:


credit_card.head()


# In[42]:


features = ['Total_visits_bank','Total_visits_online','Total_calls_made']


# In[4]:


credit_card.shape


# In[5]:


credit_card.value_counts()


# In[6]:


credit_card.isna().sum()


# In[7]:


credit_card.dtypes


# In[45]:


#Checking the distributions of the interactions

for feature in features:
    sns.distplot(credit_card[feature]) 
    plt.show()



# In[43]:


### Creating a new feature with total interaction with banks  for analysis
credit_card=credit_card.copy() 
credit_card['Total_interactions'] = credit_card['Total_visits_bank'] + credit_card['Total_visits_online'] + credit_card['Total_calls_made']
# Total interactions = total calls + totals visits in banks + total online visits
plt.figure(figsize=(12,8))
feature_perc=[]
for feature in features:
    feature_perc.append((credit_card[feature].sum()/credit_card['Total_interactions'].sum())*100)
plt.pie(feature_perc,labels=['Bank Visits','Online Visits','Calls Made'],autopct='%1.2f',textprops=dict(color="w"))
plt.legend()
plt.title("% age of interactions with respect to the medium")
plt.show()


# In[15]:


# Identify the duplicated customer keys
duplicate_keys = credit_card.duplicated('Customer Key') == True


# In[16]:


# Drop duplicated keys

credit_card = credit_card[duplicate_keys == False]


# In[17]:


credit_card.drop(columns = ['Sl_No', 'Customer Key'], inplace = True)


# In[19]:


credit_card=credit_card[~credit_card.duplicated()]


# In[21]:


credit_card.shape


# In[24]:


for col in credit_card.columns:
     print(col)
     print('Skew :',round(credit_card[col].skew(),2))
     plt.figure(figsize=(15,4))
     plt.subplot(1,2,1)
     credit_card[col].hist()
     plt.ylabel('count')
     plt.subplot(1,2,2)
     sns.boxplot(x=credit_card[col])
     plt.show()


# In[25]:


### checking the  correlation among different variables.

plt.figure(figsize=(10,10))
sns.heatmap(credit_card.corr(), annot=True, fmt='0.2f')
plt.show()


# In[29]:


###TO SCALE THE DATA 
from sklearn.preprocessing import StandardScaler


# In[31]:


#####SCALLING THE DATA

scaler=StandardScaler()
credit_card_scaled=pd.DataFrame(scaler.fit_transform(credit_card), columns=credit_card.columns)


# In[32]:


credit_card_scaled.head()


# In[33]:


###Creating copy of the data to store labels from each algorithm
credit_card_scaled_copy = credit_card.copy(deep=True)


# In[46]:


###Selecting the features
x = credit_card.iloc[:,2:].values 


# In[47]:


x


# In[48]:


x.shape


# In[34]:


###Let us now fit k-means algorithm on our scaled data and find out the optimum number of clusters to use

from sklearn.cluster import KMeans


# In[36]:


sse = {} 
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000, random_state=1).fit(credit_card_scaled)
    sse[k] = kmeans.inertia_

plt.figure()
plt.plot(list(sse.keys()), list(sse.values()), 'bx-')
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.show()


# In[37]:


###Fit the K-means algorithms on the scaled data
kmeans = KMeans(n_clusters=3, max_iter=1000, random_state=1) 

kmeans.fit(credit_card_scaled)

credit_card_scaled_copy['Labels'] = kmeans.predict(credit_card_scaled) 
credit_card['Labels'] = kmeans.predict(credit_card_scaled) 


# In[39]:


#Number of observations in each cluster
credit_card.Labels.value_counts()



# In[40]:


#Calculating summary statistics of the original data for each label
mean = credit_card.groupby('Labels').mean()
median = credit_card.groupby('Labels').median()
df_kmeans = pd.concat([mean, median], axis=0)
df_kmeans.index = ['group_0 Mean', 'group_1 Mean', 'group_2 Mean', 'group_0 Median', 'group_1 Median', 'group_2 Median']
df_kmeans.T



# In[41]:


#Visualizing different features w.r.t K-means labels
credit_card_scaled_copy.boxplot(by = 'Labels', layout = (1,5),figsize=(20,7))
plt.show()



# In[49]:


kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(x)


# In[52]:


###Converting the cluster to data frame 
convert = pd.DataFrame(y_kmeans,columns=['convert']) 
convert



# In[ ]:




