#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd

get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")


# In[3]:


dfo = pd.read_csv('C:\\Users\\USER\\Desktop\\OLUWATOMMY\\work\\housing.csv')


# In[4]:


dfo.head()


# In[5]:


dfo.isnull().sum()


# In[9]:


dfo['total_bedrooms']=dfo.total_bedrooms.fillna(0)


# In[18]:


dfo['rooms_per_household'] = dfo['total_rooms']/dfo['households']
dfo['bedrooms_per_room'] = dfo['total_bedrooms']/dfo['total_rooms']
dfo['population_per_household'] = dfo['population']/dfo['households']
dfo.head(3)


# In[19]:


dfo['ocean_proximity'].mode()


# In[22]:


dfo_numerical = dfo.copy
dfo_numerical = dfo.drop(['median_house_value', 'ocean_proximity'], axis=1)
dfo_numerical.describe()


# In[24]:


dfo_numerical.corr()


# In[32]:


plt.figure(figsize=(15,10))
sns.heatmap(dfo_numerical.corr(),annot=True, linewidth=5, cmap="Blues")
plt.title('Heatmap showing correlations between numerical data')
plt.show()


# In[37]:


dfo_numerical.corr().unstack().sort_values(ascending = False)


# In[ ]:


#The features with the biggest correlation
#in this datset are total_bedrooms and households


# In[42]:


dfo_class = dfo.copy
dfo_class('median_house_value').mean()
#mean for our median_house_value is 206855.816909


# In[52]:


dfo_class = dfo.copy()
mean = dfo_class['median_house_value'].mean()

dfo_class['above_average'] = np.where(dfo_class['median_house_value']>=mean,1,0)
dfo_class['above_average']


# In[ ]:


dfo_class = dfo_class.drop('median_house_value', axis=1)


# In[47]:


#Split your data in train/val/test sets, with 60%/20%/20% distribution.
from sklearn.model_selection import train_test_split


# In[48]:


# Divide the dataset into two; train and test 
df_full_train, df_test = train_test_split(dfo_class, test_size=0.2, random_state=42)


# In[49]:


len(df_full_train), len(df_test)


# In[50]:


#Split the train data into two; train and validation
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42)


# In[53]:


len(df_train), len(df_test), len(df_val)


# In[ ]:


#To reset the index of the 3 splitted data: Not compulsory just for data organisation
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)


# In[54]:


y_train =df_train.above_average.values
y_val = df_val.above_average.values
y_test = df_test.above_average.values


# In[59]:


#Mutual information
from sklearn.metrics import mutual_info_score


# In[60]:


categorical = ['ocean_proximity']


# In[62]:


def calculate_mi(series):
    return mutual_info_score(series, df_train.above_average)

df_mi =df_train[categorical].apply(calculate_mi)
df_mi =df_mi.sort_values(ascending=False).to_frame(name= 'mutual_information')
df_mi


# In[63]:


df_mi.round(2)


# In[57]:


df_train = df_train.drop('above_average', axis=1
df_val = df_val.drop('above_average', axis=1
df_test = df_test.drop('above_average', axis=1)


# In[66]:


from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[77]:


numerical = ['latitude', 'longitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'rooms_per_household', 'bedrooms_per_room', 'population_per_household' ]


# In[71]:


train_dict = df_train[categorical + numerical].to_dict(orient='records')


# In[72]:


dv = DictVectorizer(sparse=False)
dv.fit(train_dict)

X_train = dv.transform(train_dict)


# In[73]:


val_dicts = df_val[categorical + numerical].to_dict(orient='records')


# In[74]:


#We don't fit on validation dataset, we've already fitted on train, so we would just transform
x_val = dv.transform(val_dicts)


# In[75]:


model = LogisticRegression(solver='lbfgs', C=1.0, random_state=42)
model.fit(X_train, y_train)

#hard predictions
model.predict(x_val)
y_pred = model.predict(x_val)

accuracy = np.round(accuracy_score(y_val, y_pred),2)
print(accuracy)


# In[ ]:


#Accuracy of 0.98 which is close to the 0.95 amoung the options


# In[78]:


#Concatenation
features = categorical + numerical
features


# In[81]:


orig_acc = accuracy


for c in features:
    subset = features.copy()
    subset.remove(c)
    
    train_dict = df_train[subset].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    dv.fit(train_dict)

    x_train = dv.transform(train_dict)

    model = LogisticRegression(solver='lbfgs', C=1.0, random_state=42)
    model.fit(x_train, y_train)

    val_dict = df_val[subset].to_dict(orient='records')
    x_val = dv.transform(val_dict)

    y_pred = model.predict(x_val)

    new_score = accuracy_score(y_val, y_pred)
    print(c, orig_acc - new_score)


# In[ ]:


#feature with the smallest difference is households with 0.15030038759689923


# In[83]:


dfo['median_house_value']=np.log1p(dfo['median_house_value'])


# In[97]:


df_train_full, df_test = train_test_split(dfo, test_size=0.2, random_state=1)


# In[98]:


df_train, df_val = train_test_split(df_train_full, test_size=0.25, random_state=1)


# In[99]:


df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)


# In[100]:


y_train = df_train.median_house_value.values
y_val = df_val.median_house_value.values
y_test = df_test.median_house_value.values


# In[101]:


df_train = df_train.drop('median_house_value', axis=1)
df_val = df_val.drop('median_house_value', axis=1)
df_test = df_test.drop('median_house_value', axis=1)


# In[102]:


train_dict = df_train[categorical + numerical].to_dict(orient='records')


# In[103]:


dv = DictVectorizer(sparse=False)
dv.fit(train_dict)

x_train = dv.transform(train_dict)

val_dict = df_val[categorical + numerical].to_dict(orient='records')
x_val = dv.transform(val_dict)


# In[104]:


#Ridge regression
from sklearn.linear_model import Ridge
#For RMSE
from sklearn.metrics import mean_squared_error


# In[105]:


for a in [0, 0.01, 0.1, 1, 10]:
    model = Ridge(alpha=a, random_state=42, solver='sag')
    model.fit(x_train, y_train)
    
    y_pred = model.predict(x_val)
    
    score = np.sqrt(mean_squared_error(y_val, y_pred))
    
    print(a, round(score, 3))


# In[ ]:




