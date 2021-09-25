#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_csv("C:/Users/admin/Desktop/data science/Online-class.csv")


# In[3]:


data.head()


# In[4]:


data.columns


# In[5]:


data.describe()


# In[6]:


data.isnull().sum()


# In[7]:


data.columns


# # Relating the variable with scatterplots
# 

# In[8]:


sns.relplot(x="Impact of COVID19  in your Life",y="Self Analysis (before & After Covid)",hue='Fear of COVID19?',data=data)


# In[9]:


sns.pairplot(data)


# In[10]:


sns.relplot(x="Impact of COVID19  in your Life",y="Self Analysis (before & After Covid)",kind='line',data=data)


# In[11]:


sns.catplot(x="Impact of COVID19  in your Life",y="Self Analysis (before & After Covid)",data=data)


# In[12]:


data.head(5)


# In[13]:


data.info()


# In[14]:


data.shape


# In[15]:


data.columns


# In[16]:


#Accessing a single column
data['Students Name'].head()


# In[17]:


#To fetch the data which satisfies the given condition
data[data['Comment On Online Exams'] > 3]


# In[18]:


data[(data['Utilization of Online Classes'] > 2) & (data['vaccinated?']=='yes')].shape


# In[19]:


data.dtypes


# In[20]:


data.describe()


# In[21]:


data.columns


# In[22]:


data['Self Analysis (before & After Covid)'].value_counts()


# In[23]:


data.loc[2:8,'District':'State']


# In[24]:


data.iloc[2:10,3]


# In[25]:


min(data['Self Analysis (before & After Covid)'])


# In[26]:


max(data['Self Analysis (before & After Covid)'])


# In[27]:


min(data['About Online Classes?'])


# In[28]:


max(data['About Online Classes?'])


# In[29]:


min(data[' Knowledge  gained in Online Class? '])


# In[30]:


max(data[' Knowledge  gained in Online Class? '])


# In[31]:


min(data['Time management during Online classes'])


# In[32]:


max(data['Time management during Online classes'])


# In[33]:


min(data['Utilization of Online Classes'])


# In[34]:


max(data['Utilization of Online Classes'])


# In[35]:


min(data['Mentality about Online Education'])


# In[36]:


max(data['Mentality about Online Education'])


# In[37]:


min(data['Device Availability '])


# In[38]:


max(data['Device Availability '])


# In[39]:


min(data['Comment On Online Exams'])


# In[40]:


max(data['Comment On Online Exams'])


# In[41]:


min(data['Network Availability'])


# In[42]:


max(data['Network Availability'])


# In[43]:


min(data['Have any of your family members affected  by  Covid19?'])


# In[44]:


max(data['Have any of your family members affected  by  Covid19?'])


# In[45]:


min(data['vaccinated?'])


# In[46]:


max(data['vaccinated?'])


# In[47]:


min(data['Any Home remedies taken?   '])


# In[48]:


max(data['Any Home remedies taken?   '])


# In[49]:


min(data['Fear of COVID19?'])


# In[50]:


max(data['Fear of COVID19?'])


# In[51]:


min(data['Impact of COVID19  in your Life'])


# In[52]:


max(data['Impact of COVID19  in your Life'])


# In[53]:


data['Fear of COVID19?'].min


# In[54]:


data['Fear of COVID19?'].max


# In[55]:


#to find the unique countries playing cricket

data['State'].nunique()


# In[56]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[57]:


data.columns


# In[58]:


data['District'].value_counts().head(10).plot(kind='bar')


# In[59]:


data[' Knowledge  gained in Online Class? '].value_counts().head(10).plot(kind='bar')


# In[60]:


data['About Online Classes?'].value_counts().head(10).plot(kind='line')


# In[61]:


data['Time management during Online classes'].value_counts().head(10).plot(kind='bar')


# In[62]:


data['Utilization of Online Classes'].value_counts().head(10).plot(kind='bar')


# In[63]:


data['Mentality about Online Education'].value_counts().head(10).plot(kind='bar')


# In[64]:


data['Device Availability '].value_counts().head(10).plot(kind='bar')


# In[65]:


data['Comment On Online Exams'].value_counts().head(10).plot(kind='bar')


# In[66]:


data['Network Availability'].value_counts().head(10).plot(kind='line')


# In[67]:


data['Have any of your family members affected  by  Covid19?'].value_counts().head(10).plot(kind='bar')


# In[68]:


data['vaccinated?'].value_counts().head(10).plot(kind='bar')


# In[69]:


data['Any Home remedies taken?   '].value_counts().head(200).plot(kind='bar')


# In[70]:


data.columns


# In[71]:


cols=['Students Name','District','State','Country  (eg.India)']
users = pd.read_csv("C:/Users/admin/Desktop/data science/Online-class.csv",sep="|",names=cols)
users.head(10)


# In[72]:


import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


# In[73]:


#for getting the summary of categorical data

data['District'].value_counts()


# In[74]:


pwd


# In[75]:


cd Pandas DataSet


# In[76]:


D = data.groupby('About Online Classes?')
D


# In[77]:


D = data.groupby('About Online Classes?').mean
D


# In[78]:


D = data.groupby('About Online Classes?').count
D


# In[79]:


data.columns


# In[80]:


data['Comment On Online Exams'].value_counts()


# In[81]:


data['Comment On Online Exams'].nunique()


# In[82]:


data['Comment On Online Exams'].unique()


# In[83]:


D = data.groupby('Comment On Online Exams').mean().sort_values(ascending=False,by='Comment On Online Exams')
D


# In[84]:


Ground = data.groupby('State')['Students Name']
Ground


# In[85]:


Ground = data.groupby('State')['Students Name'].count
Ground


# In[86]:


#dimension of data dataset

dim = data.ndim


# In[87]:


print("ndim of dataframe = {}\nndim of series={}")
format(dim)


# In[88]:


type(data)


# In[89]:


data.head(10)


# In[90]:


#columns name to be displayed in lowercase

data=data.rename(columns=str.lower)


# In[91]:


data.head()


# In[92]:


data.loc()


# In[93]:


data.columns


# In[94]:


sns.boxplot(x='comment on online exams',y='utilization of online classes',data=data)


# In[95]:


data['district'].unique()


# In[96]:


plt.scatter(data['district'],data['state'])
plt.xlabel('district')
plt.ylabel('state')
plt.show()


# In[97]:


sns.heatmap(data.isnull())
plt.show()


# In[98]:


correlation_matrix = data.corr()

plt.subplots(figsize=(20,15))
sns.heatmap(correlation_matrix,annot=True)
plt.show()


# In[99]:


data.columns


# In[100]:


sns.relplot(x="time management during online classes",y="utilization of online classes",hue="comment on online exams",data=data)


# In[101]:


from sklearn import svm
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


# In[102]:


data.head()


# In[103]:


state = LabelEncoder()
data['state'] = state.fit_transform(data['state'])


# In[104]:


data


# In[105]:


data.columns


# In[106]:


sns.countplot(data['district'])


# In[107]:


sns.relplot(x='about online classes?',y='utilization of online classes',hue='comment on online exams',kind='line',data=data,height=5,aspect = 3)


# In[108]:


data.boxplot()


# In[109]:


data.tail(10)


# In[110]:


data.hist()


# In[111]:


data.isnull()


# In[112]:


data.isnull().sum()


# In[113]:


threshold = len(data)*0.1
threshold


# In[114]:


data.dropna(thresh=threshold,axis=1,inplace=True)


# In[115]:


print(data.isnull().sum())


# # Data imputation and Manipulation

# In[116]:


def impute_median(series):
    return series.fillna(series.median())


# In[117]:


data.state = data['state'].transform(impute_median)


# In[118]:


data.isnull().sum()


# In[119]:


data.columns


# # Data Visualization

# In[120]:


grp = data.groupby('vaccinated?')
x = grp['comment on online exams'].agg(np.mean)
y = grp['utilization of online classes'].agg(np.sum)

print(x)
print(y)


# In[121]:


plt.plot(x,'ro')
plt.xticks(rotation=90)
plt.show


# In[122]:


plt.figure(figsize=(16,5))
plt.plot(x,'ro',color='g')
plt.xticks(rotation=90)
plt.title('time management during online classes')
plt.xlabel('utilization of online classes')
plt.ylabel('mentality about online education')
plt.show()


# In[123]:


plt.figure(figsize=(16,5))
plt.plot(y,'r--',color='b')
plt.xticks(rotation=90)
plt.title('time management during online classes')
plt.xlabel('utilization of online classes')
plt.ylabel('mentality about online education')
plt.show()


# In[124]:


data.index


# In[125]:


data.head(2)


# In[126]:


data['about online classes?'] == 4


# In[127]:


data[data['about online classes?'] == 4]


# In[128]:


data.rename(columns={'about online classes?': 'Online_Classes'})


# In[129]:


data['utilization of online classes'].var()


# In[130]:


data['time management during online classes'].value_counts()


# In[131]:


data[data['time management during online classes']=='Effective']


# In[132]:


#str.contains
data[data['time management during online classes'].str.contains('Online_Classes')]


# In[133]:


#str.contains
data[data['time management during online classes'].str.contains('Online_Classes')].head(50)


# In[134]:


data.columns


# In[135]:


data[(data['utilization of online classes'] > 2) & (data['comment on online exams'] == 4)]


# In[136]:


data.columns


# In[137]:


data.rename(columns={'students name': 'Name'})


# In[138]:


data.rename(columns={'country  (eg.india)': 'Country'})


# In[139]:


data.rename(columns={' knowledge  gained in online class? ': 'knowledge'})


# In[140]:


data.rename(columns={'time management during online classes': 'Time'})


# In[142]:


data.groupby('utilization of online classes').min()


# In[143]:


(data['utilization of online classes']=='Clear') | (data['fear of covid19?']) 


# In[144]:


data.head()


# In[145]:


data.groupby('utilization of online classes').max()


# In[146]:


data.groupby('about online classes?').min()


# In[147]:


data.groupby('about online classes?').max()


# In[148]:


data.groupby('mentality about online education').min()


# In[149]:


data.groupby('mentality about online education').max()


# In[153]:


data.groupby('utilization of online classes').min()


# In[154]:


data.groupby('utilization of online classes').max()


# In[ ]:




