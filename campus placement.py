# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 17:32:06 2020

@author: HP
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_excel('file:///E:/kaggle/campus placement data.xlsx')

data.info()

data.isnull().sum()

data['salary'].fillna(0,inplace = True)

def plot(data,x,y):
    sns.boxplot(x = data['gender'],y= data['salary'])
    g = sns.FacetGrid(data, row = y)
    g = g.map(plt.hist,x)
    plt.show()
    
plot(data,"salary","gender")
sns.countplot(data['status'],hue=data['gender'])

plot(data,"salary","ssc_b")
sns.countplot(data['status'],hue=data['ssc_b'])

sns.countplot(data['status'],hue=data['hsc_s'])

from scipy.stats import pearsonr
corr, _ = pearsonr(data['ssc_p'], data['hsc_p'])
print('Pearsons correlation: %.3f' % corr)
sns.regplot(x='ssc_p',y='hsc_p',data = data)

sns.countplot(data['status'],hue=data['degree_t'])

sns.countplot(data['status'],hue=data['workex'])

plt.figure(figsize =(10,10))
sns.heatmap(data.corr())

list1 =[]
cor = data.corr()
for i in cor.columns:
    for j in cor.columns :
        if abs(cor[i][j])>0.5 and i!=j:
            list1.append(i)
            list1.append(j)
print(set(list1))

sns.catplot(x="status", y="ssc_p", data=data,kind="swarm",hue='gender')
sns.catplot(x="status", y="hsc_p", data=data,kind="swarm",hue='gender')
sns.catplot(x="status", y="degree_p", data=data,kind="swarm",hue='gender')

sns.violinplot(x="degree_t", y="salary", data=data)
sns.stripplot(x="degree_t", y="salary", data=data,hue='status')

columns_needed =['gender','ssc_p','ssc_b','hsc_b','hsc_p','degree_p','degree_t','salary']
data1 = data[columns_needed]

def cat_to_num(data1,col):
    dummy = pd.get_dummies(data1[col])
    del dummy[dummy.columns[-1]]
    data1= pd.concat([data1,dummy],axis =1)
    return data1

for i in data1.columns:
    if data1[i].dtype ==object:
        print(i)
        data1 =cat_to_num(data1,i)
        
data1.drop(['gender','ssc_b','hsc_b','degree_t'],inplace =True,axis =1)        
        
x = data1 
y = data1['salary']       

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)

from sklearn.tree import DecisionTreeRegressor as DTR
regr = DTR()
regr.fit(x_train,y_train)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train,y_train)

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rf.fit(x_test, y_test);

from sklearn.metrics import r2_score
print(r2_score(y_test,regr.predict(x_test)))

ax1 = sns.distplot(y_test,hist=False,color ="r",label ="Actual Value")
sns.distplot(regr.predict(x_test),color ="b",hist = False,label = "Preicted Value",ax =ax1)
