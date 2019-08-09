#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import re


# In[29]:


df = pd.read_csv('cleaned_data.csv')
df['skills'] = df[['skills']].fillna(value='unknown')
df['about'] = df[['about']].fillna(value='unknown')
df['name'] = df[['name']].fillna(value='unknown')
df['provider'] = df[['provider']].fillna(value='unknown')
dfk = df.copy()
dfk['index']=dfk.index


def search_provider(provider):
    if provider == None:
        temp = dfk.copy()
        return temp
    else:
        temp = dfk.copy()
        temp = temp[temp['provider'].str.contains(provider, na = False, case=False)] #Ignore case and None value
        temp = temp.drop_duplicates()
        return temp

def search_keywords(keywords):
    if keywords == None:
        temp = dfk.copy()
        return temp
    else:
        temp_n = dfk[dfk['name'].str.contains(keywords, na = False, case=False)] #Ignore case and None value
        temp_a = dfk[dfk['about'].str.contains(keywords, na = False, case=False)] 
        temp_s = dfk[dfk['skills'].str.contains(keywords, na = False, case=False)] 
        frames = [temp_n, temp_a,temp_s]
        temp = pd.concat(frames)
        temp = temp.drop_duplicates()
        return temp

def search_sentences(sentence):
    if sentence == None:
        temp = dfk.copy()
        return temp
    else:
        wordlist = re.sub("[^\w]", " ",  sentence).split()
        result = pd.DataFrame()
        for i in range(len(wordlist)):
            temp = search_keywords(wordlist[i])
            result = pd.concat([result, temp])

        temp = temp.drop_duplicates()
        return temp

def search_skills(skills):
    if skills == None:
        temp = dfk.copy()
        return temp
    else:
        wordlist = re.sub("[^\w]", " ",  skills).split()
        temp = dfk.copy()
        for i in range(len(wordlist)):
            temp = temp[temp['skills'].str.contains(wordlist[i], na = False, case=False)] 
            #result = pd.concat([result, temp])

        temp = temp.drop_duplicates()
        return temp
    
def search_subtitles(subtitles):
    if subtitles == None:
        temp = dfk.copy()
        return temp
    else:
        wordlist = re.sub("[^\w]", " ",  subtitles).split()
        temp = dfk.copy()
        for i in range(len(wordlist)):
            temp = temp[temp['subtitles'].str.contains(wordlist[i], na = False, case=False)] 

        temp = temp.drop_duplicates()
        return temp

def search_ratings(overall): #Input:float
    if overall == None:
        temp = dfk.copy()
        return temp
    else:
        overall=float(overall)
        temp = dfk.copy()
        temp = temp[(temp['overall'] > overall)]
        return temp

def search_enrolled(enrolled): #Input:int
    if enrolled == None:
        temp = dfk.copy()
        return temp
    else:
        enrolled = int(enrolled)
        temp = dfk.copy()
        temp = temp[(temp['enrolled'] > enrolled)]
        return temp

def search_topic(topic):
    if topic == None:
        temp = dfk.copy()
        return temp
    else:
        temp = dfk.copy()
        temp = temp[(temp['topic'] == topic)]
        return temp

def inter_set(provider, sentence, skills, subtitles, overall, enrolled, topic):
    pro = search_provider(provider)['index'].tolist()
    sen = search_sentences(sentence)['index'].tolist()
    ski = search_skills(skills)['index'].tolist()
    sub = search_subtitles(subtitles)['index'].tolist()
    rat = search_ratings(overall)['index'].tolist()
    enr = search_enrolled(enrolled)['index'].tolist()
    top = search_topic(topic)['index'].tolist()
    
    result = list(set(pro)&set(sen)&set(ski)&set(sub)&set(rat)&set(enr)&set(top))
    temp = dfk.copy()
    temp = temp[temp['index'].isin(result)].drop_duplicates(subset='name', keep="first")
    temp = temp.drop(columns=['category', 'sub-category','Unnamed: 0'])
    return temp





