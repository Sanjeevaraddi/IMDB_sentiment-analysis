#!/usr/bin/env python
# coding: utf-8

# In[1]:

#importing necessary libraries
import numpy as np
import pandas as pd
import streamlit as st

#Importing the IMDB dataset 
file = 'IMDB Dataset.csv'


# In[4]:

#setting the streamlit title 
st.title('IMDB sentiment analysis')
data_state = st.text('loading data....')
df_review = pd.read_csv(file)
data_state.text('Data loading done..')

#Setting a checkbox to showcase the raw data
agree = st.checkbox('Show raw data')

if agree:
    st.subheader('Raw data')
    st.write(df_review)
#print(df_review)
#print(df_review.head())


# In[ ]:


Pos_review =df_review[df_review['sentiment']=='positive']
print(Pos_review)


# In[ ]:


neg_review =df_review[df_review['sentiment']=='negative']
print(neg_review)


# In[ ]:

#Splitting the data in 3:1 ratio of train and test
st.subheader('Splitting data into 3:1 for train and test split')
from sklearn.model_selection import train_test_split
train, test = train_test_split(df_review, test_size=0.25, random_state = 1)
train_data = train.shape[0]
test_data = test.shape[0]
st.write('Total train data points')
st.write(train_data)
st.write('Total test data points')
st.write(test_data)

# In[ ]:


train_x, train_y = train['review'], train['sentiment']
test_x, test_y = test['review'], test['sentiment']
print(train_x)


# In[ ]:

#converting a text into vector for model building
data_state1 = st.text('Starting with the data processing and featurising the data using BoW')
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english')
train_vector = tfidf.fit_transform(train_x)
test_vector = tfidf.transform(test_x)
data_state1.text('data processing and featurising done !')

# In[ ]:

#training the model using Decision tree
data_state2 = st.text('Using decision tree classfier to train the model')
from sklearn.tree import DecisionTreeClassifier
dec_tree = DecisionTreeClassifier()
dec_tree.fit(train_vector, train_y)
data_state2.text('Model training done using decision tree classfier')

# In[ ]:

#understanding the output of the model using F1 score
from sklearn.metrics import f1_score
st.write('F1 score using BoW for vectorising data and decision tree for training is') 
print(f1_score(test_y,dec_tree.predict(test_vector)))

fn = st.text_input ("Give review of any movie you have watched recently!")
User_text = tfidf.fit_transform(fn)
Status = dec_tree.predict(User_text)
st.write('User rating is..')
st.write(status)






