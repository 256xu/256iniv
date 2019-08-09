#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity


# In[2]:


df = pd.read_csv('cleaned_data.csv')
dft = pd.read_csv('cleaned_data.csv')
df['skills'] = df[['skills']].fillna(value='unknown')
df['about'] = df[['about']].fillna(value='unknown')
df['name'] = df[['name']].fillna(value='unknown')
df['provider'] = df[['provider']].fillna(value='unknown')
ps = PorterStemmer()
stop = stopwords.words('english')

dft['about'] = dft[['about']].fillna(value='unknown')
dft['about'] = dft['about'].str.replace('\d+', '')
dft['about'] = dft['about'].str.split(' ').apply(lambda x: [item for item in x if item not in stop])
dft['about'] = dft['about'].apply(', '.join)
v = TfidfVectorizer()
x = v.fit_transform(dft['about'])
dftx = pd.DataFrame(x.toarray(), columns=v.get_feature_names())
tfidf = pd.concat([dft, dftx], axis=1)

dft = pd.read_csv('cleaned_data.csv')
dft['name'] = dft[['name']].fillna(value='unknown')
dft['name'] = dft['name'].str.replace('\d+', '')
dft['name'] = dft['name'].str.split(' ').apply(lambda x: [item for item in x if item not in stop])
dft['name'] = dft['name'].apply(', '.join)
v = TfidfVectorizer()
x = v.fit_transform(dft['name'])
dftx1 = pd.DataFrame(x.toarray(), columns=v.get_feature_names())
tfidf1 = pd.concat([dft, dftx1], axis=1)

dft = pd.read_csv('cleaned_data.csv')
dft['skills'] = dft[['skills']].fillna(value='unknown')
dft['skills'] = dft['skills'].str.replace('\d+', '')
dft['skills'] = dft['skills'].str.split(' ').apply(lambda x: [item for item in x if item not in stop])
dft['skills'] = dft['skills'].apply(', '.join)
v = TfidfVectorizer()
x = v.fit_transform(dft['skills'])
dftx2 = pd.DataFrame(x.toarray(), columns=v.get_feature_names())
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

