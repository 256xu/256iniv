#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import nltk
import re
#from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
#from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity


# In[2]:


df = pd.read_csv('cleaned_data.csv')
dft = pd.read_csv('cleaned_data.csv')
df['skills'] = df[['skills']].fillna(value='unknown')
df['about'] = df[['about']].fillna(value='unknown')
df['name'] = df[['name']].fillna(value='unknown')
df['provider'] = df[['provider']].fillna(value='unknown')

df = pd.read_csv('cleaned_data.csv')
dft = pd.read_csv('cleaned_data.csv')
df['skills'] = df[['skills']].fillna(value='unknown')
df['about'] = df[['about']].fillna(value='unknown')
df['name'] = df[['name']].fillna(value='unknown')
df['provider'] = df[['provider']].fillna(value='unknown')

dftx = pd.read_csv('dftx.csv')
tfidf = pd.concat([dft, dftx], axis=1)


dftx1 = pd.read_csv('dftx1.csv')
tfidf1 = pd.concat([dft, dftx1], axis=1)


dftx2 = pd.read_csv('dftx2.csv')
tfidf2 = pd.concat([dft, dftx2], axis=1)


# In[8]:


#tf-idf  & cosine similarity

sim_a = cosine_similarity(dftx)
sim_n = cosine_similarity(dftx1)
sim_s = cosine_similarity(dftx2)
word_list_about = dftx.columns.get_values().tolist()
word_list_name = dftx1.columns.get_values().tolist()
word_list_skills = dftx2.columns.get_values().tolist()
dft['index']=dft.index

def tfidf_word(word):
    temp = df.copy()
    word = word.lower()
    
    if word in word_list_about:
        temp_a = tfidf.sort_values(by=word , ascending=False)[[word]]
        temp['result_a'] = temp_a[[word]]
    else:
        temp['result_a'] = 0.0
        
    if word in word_list_name:
        temp_n = tfidf1.sort_values(by=word , ascending=False)[[word]]
        temp['result_n'] = temp_n[[word]]
    else:
        temp['result_n'] = 0.0
    
    temp['result_weighted'] = 0.8 * temp['result_n'] + 0.2 * temp['result_a']
    temp = temp.sort_values(by=['result_weighted'] , ascending=False)
    #temp = temp[(temp['result_weighted'] > 0)]
    temp = temp.drop(columns=['result_a', 'result_n'])
    
    return temp

def tfidf_sim(index):
    dftx_temp = dft.copy()
    index = int(index)
    dftx_temp['sim_temp_about'] = sim_a[index]
    dftx_temp['sim_temp_name'] = sim_n[index]
    dftx_temp['sim_temp_skills'] = sim_s[index]
    dftx_temp['sim_result'] = 0.6 * dftx_temp['sim_temp_name'] + 0.2 * dftx_temp['sim_temp_about'] + 0.2 * dftx_temp['sim_temp_skills']
    dftx_temp = dftx_temp.sort_values(by="sim_result" , ascending=False)
    dftx_temp = dftx_temp.drop(columns=['category','sub-category','Unnamed: 0','sim_temp_about','sim_temp_name','sim_temp_skills'])
    dftx_temp = dftx_temp.drop_duplicates(subset='name', keep="first")
    dftx_temp = dftx_temp[1:6]
    dftx_temp = dftx_temp[(dftx_temp['sim_result'] > 0)]
    return dftx_temp

def tfidf_words(sentence):
    wordlist = re.sub("[^\w]", " ",  sentence).split()
    temp = pd.DataFrame()
    for i in range(len(wordlist)):
        temp[wordlist[i]] = tfidf_word(wordlist[i])['result_weighted']
    
    temp.loc[:,'Total'] = temp.sum(axis=1)
    temp = temp[['Total']]
    
    res = pd.concat([dft, temp], axis=1).sort_values(by="Total" , ascending=False)
    res = res.drop_duplicates(subset='name', keep="first")
    res= res.drop(columns=['category','sub-category','Unnamed: 0'])
    res = res[(res['Total'] > 0)]
    return res

