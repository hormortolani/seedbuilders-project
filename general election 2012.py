#!/usr/bin/env python
# coding: utf-8

# 
# Election Poll Data Analysis
# 
# In this project, we'll look at the opinion poll data from the upcoming 2012 General Election. In the process, we'll try to answer the following questions:
# 
#     Who was being polled and what was their party affiliation?
#     Did the poll results favour Romney or Obama?
#     How did voter sentiment change over time?
#     Can we see an effect in the polls from the debates?
# 
# So let's get started with the imports!
# 

# In[7]:


#Data Analysis imports
import pandas as pd
from pandas import Series,DataFrame
import numpy as np
from datetime import datetime

#Visualisation imports
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')

#Imports to grab and parse data from the web
import requests

from io import StringIO


# In[55]:


#url for the poll data
url = "https://elections.huffingtonpost.com/pollster/2012-general-election-romney-vs-obama.csv"
#Using requests to get the data in text form
source = requests.get(url).text

#Using String IO to prevent IO errors with pandas
poll_data = StringIO(source)


# In[56]:


#Reading the csv object into a pandas dataframe
poll = pd.read_csv(poll_data)


# In[63]:


poll.info()


# In[64]:


poll.dtypes


# In[65]:


poll.isna().sum()


# In[66]:


poll.head()


# In[69]:


###Quickly visualising the affiliations of the different pollsters.


sns.catplot(x='Affiliation',data=poll,kind='count')


# In[70]:


##Most of the polls have no affiliation; though there's stronger affiliation for Democrats than for Republicans.
sns.catplot(x='Affiliation',data=poll,hue='Population',kind='count')


# In[74]:


##As there's a strong sample of registered voters that are not affiliated, we can hope that the poll data is a good representation of the upcoming elections.



sns.catplot(x='Mode',data=poll,kind='count')


# In[75]:


##Did the poll results favour Romney or Obama?


avg = pd.DataFrame(poll.mean())

avg.drop(['Number of Observations','Question Iteration'],axis=0,inplace=True)

avg



# In[78]:


avg.plot(yerr=avg,kind='bar',legend=False)


# In[79]:


##How did voter sentiment change over time?


poll['Difference'] = (poll.Romney - poll.Obama)/100

poll.head()



# In[80]:


#Grouping polls by the start data
poll = poll.groupby(['Start Date'],as_index=False).mean()

poll.tail()



# In[84]:


poll.plot('Start Date','Difference',figsize=(20,6),marker='o',linestyle='-')


# In[85]:


poll.plot('Start Date','Difference',figsize=(15,4),marker='o',linestyle='-',xlim=(209,229))

#Vertical line for debate date
plt.axvline(x=228)


# In[86]:


poll.head()


# In[ ]:




