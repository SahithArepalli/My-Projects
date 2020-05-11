# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 18:12:40 2019

@author: HP
"""

import pandas as pd
import numpy as np
from nltk.corpus import stopwords

data = pd.read_excel('file:///E:/python/DATA/sn_compliance_control.xlsx', encoding = 'ISO-8859-1')

data.isnull().sum()

data = data.drop(['Additional Information','Additional comments','Enforcement','Domain Path', 'Description',
           'Name','Frequency','Owning group','Profile type','Profile type.1','Tags'], axis = 1)

data.describe()

data = data.replace(np.nan, '', regex=True)

data.columns

#data = data.apply(lambda x: x.astype(str).str.lower())
#d1 = data.apply(lambda x: x.astype(str).str.upper())

stop = set(stopwords.words('english'))
def remove_stopword(word):
    return word not in words
 
data['Policy Statement'] = data['Policy Statement'].str.lower().str.split()
d = data['Policy Statement'].apply(lambda x : [item for item in x if item not in stop])

data['Type'] = data['Type'].str.lower().str.split()
d1 = data['Type'].apply(lambda x : [item for item in x if item not in stop])

data['Category'] = data['Category'].str.lower().str.split()
d2 = data['Category'].apply(lambda x : [item for item in x if item not in stop])

d = pd.concat([d, d1,d2], axis=1)

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

d['Policy Statement'] = d["Policy Statement"].apply(lambda x: [stemmer.stem(y) for y in x])
d['Type'] = d["Type"].apply(lambda x: [stemmer.stem(y) for y in x])
d['Category'] = d["Category"].apply(lambda x: [stemmer.stem(y) for y in x])

data = data.drop(['Policy Statement','Type','Category'],axis=1)
data = pd.concat([data,d], axis=1)
data.columns


from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer

vectorizer = CountVectorizer()
bag_of_words1 = CountVectorizer(tokenizer=lambda doc: doc, lowercase=False).fit_transform(data['Type'])
bag_of_words2 = CountVectorizer(tokenizer=lambda doc: doc, lowercase=False).fit_transform(data['Category'])
bag_of_words3 = CountVectorizer(tokenizer=lambda doc: doc, lowercase=False).fit_transform(data['Policy Statement'])

from scipy.sparse import hstack
bag_of_words = hstack([bag_of_words1,bag_of_words2,bag_of_words3])
print(bag_of_words.toarray())
s=bag_of_words.toarray()

tfidf = TfidfTransformer()
X = tfidf.fit_transform(s)
print(X)
print(X.shape)
print(X.toarray())
s1=X.toarray()

from sklearn.naive_bayes import MultinomialNB
from collections import Counter
classifier = MultinomialNB()

x = s1
y = data['Classification']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)

classifier.fit(x_train,y_train)

expected = y_test
predicted = classifier.predict(x_test)

collections=Counter(y_test)

from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predicted))
print(confusion_matrix(y_test,predicted))

from sklearn.metrics import accuracy_score

accuracy_score(expected,predicted)




 





