#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 13:32:45 2017

@author: shubhabrataroy
"""

############ ENRON recommendation system ######################
import numpy as np
import pandas as pd
from os.path import join
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
import itertools
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import networkx as nx    
from __future__ import division

def DataQuality(df):
    return df.isnull().sum()



def tfidf(txt, vectorizer):
    try: 
        X = vectorizer.fit_transform(txt.split('\n'))
        Y = list(X.tocoo().data)
        Y = [i/sum(Y) for i in Y]
    except ValueError:
        Y = [0]
    return Y

def tfidfCentroid(dfs): # Computes tfidf features from the email body
    dFeatures = []
    sender_ = list(set(dfs['from']))
    for jj in sender_:
        df = dfs[dfs['from'] == jj]
        addrbook_ = df['to'].tolist() + df['cc'].tolist() + df['bcc'].tolist()
        addrbook_ = list(itertools.chain(*addrbook_))
        addrbook_ = list(set(addrbook_))
        addrbook_ = data_clean(addrbook_)
        SendingPerson = []
        Recipient = []
        TfIdfCebtroid = []
        for ii in addrbook_:
            t = [1 if ii in j else 0 for j in df['to']]
            c = [1 if ii in j else 0 for j in df['cc']]
            bc = [1 if ii in j else 0 for j in df['bcc']]
            df['t'] = t
            df['c'] = c
            df['bc'] = bc
            df_ = df[(df['t']==1) | (df['c']==1) | (df['bc']==1)]
            tfidfcentroid = [sum(x) for x in itertools.izip_longest(*df_['tfidffeat'].tolist(), fillvalue=0)]        
            SendingPerson.append(jj)
            Recipient.append(ii)
            TfIdfCebtroid.append(tfidfcentroid)
        dFeat = pd.DataFrame({'sender': SendingPerson, 'receiver': Recipient, 'tfidfcentroid': TfIdfCebtroid})
        dFeatures.append(dFeat)
    dFeatures = pd.concat(dFeatures, axis=0)
    return dFeatures
    
def similarity(l1, l2): # padding the uneven vectors by zero and compute cosine similarity 
    diff = len(l1) - len(l2)
    if (diff < 0):
        l1 = l1 + np.abs(diff)*[0]
    else:
        l2 = l2 + np.abs(diff)*[0]
    similarityscore = cosine_similarity(l1, l2)
    return similarityscore[0][0]

def compute_perfromance(df):
    f = df[['to', 'from', 'cc', 'bcc', 'PredictedRecipient']]
    Accto_ = []
    Acccc_ = []
    Accbcc_ = []
    Accto_cc_bcc = []
    
    for t in sender_:
        f_ = f[f['from'] == t]
        count_t = 0
        count_cc = 0
        count_bcc = 0
        count_to_cc_bcc = 0
        for s in range(len(f_)):
            tO = f_['to'][s]
            cC = f_['cc'][s]
            bcC = f_['bcc'][s]
            preD = f_['PredictedRecipient'][s]
            if preD in tO:
                count_t += 1
            if preD in cC:
                count_cc += 1
            if preD in bcC:
                count_bcc += 1
            if ((preD in tO) | (preD in cC )| (preD in bcC)):
                count_to_cc_bcc += 1
        Accto_.append(count_t/len(f_))
        Acccc_.append(count_cc/len(f_))
        Accbcc_.append(count_bcc/len(f_))
        Accto_cc_bcc.append(count_to_cc_bcc/len(f_))    
    dPerformance = pd.DataFrame({'sender': sender_, 'AcuTo': Accto_, 'AcuCc': Acccc_, 'AcuBcc': Accbcc_, 'AcuAll': Accto_cc_bcc})
    return dPerformance

    
       

path = "/Users/shubhabrataroy/Desktop/Freelance/PhilipMoris/maildir"
dEmail = pd.read_csv(join(path, 'emails.csv'))
dEmail = dEmail[['bcc', 'body', 'cc', 'date', 'from', 'path',
       'subject', 'to']]

dEmail.fillna(0, inplace = True)
print DataQuality(dEmail)

## We decided to focus only the sent_email part in this work
dEmail['sent_items'] = [1 if 'sent_items' in j else 0 for j in dEmail['path']]
dEmail = dEmail[dEmail['sent_items'] == 1]

Stop_words = list(text.ENGLISH_STOP_WORDS)
vectorizer = TfidfVectorizer(stop_words= Stop_words, sublinear_tf=True)

########### Compute the tfidf for the email bodies

S = []  # vectorizer
for j in dEmail['body'].tolist():
    if j == 0:
        s = [0]
    else:
        s = tfidf(j, vectorizer)
    S.append(s)

dEmail['tfidffeat'] = S
dEmail['to'] = [s.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ') if s != 0 else s for s in dEmail['to'].tolist()]
dEmail['to'] = [s.split() if s != 0 else s for s in dEmail['to'].tolist()]
dEmail['to'] = [[s] if s == 0 else s for s in dEmail['to'].tolist()]
dEmail['cc'] = [s.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ') if s != 0 else s for s in dEmail['cc'].tolist()]
dEmail['cc'] = [s.split() if s != 0 else s for s in dEmail['cc'].tolist()]
dEmail['cc'] = [[s] if s == 0 else s for s in dEmail['cc'].tolist()]
dEmail['bcc'] = [s.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ') if s != 0 else s for s in dEmail['bcc'].tolist()]
dEmail['bcc'] = [s.split() if s != 0 else s for s in dEmail['bcc'].tolist()]
dEmail['bcc'] = [[s] if s == 0 else s for s in dEmail['bcc'].tolist()]

############### create the addressbook for each sender

AddrBook = []
Sender = []

senderlist = list(set(dEmail['from']))
senderlist = data_clean(senderlist)
n = 15 # number of times a receiver should be there in order to be considered

for j in senderlist:
    dEmail_ = dEmail[dEmail['from'] == j]
    if (len(dEmail_) > 100): # put a threshold of atleast 100 messages
        addrbook = dEmail_['to'].tolist() + dEmail_['cc'].tolist() + dEmail_['bcc'].tolist()
        addrbook = list(itertools.chain(*addrbook))
        count = Counter(addrbook)
        count = Counter(el for el in count.elements() if count[el] >= n)
        addrbook = count.keys()
        addrbook = data_clean(addrbook)
        AddrBook.append(addrbook)
        Sender.append(j)
    
################ Select suitable data from the entire set

appended_data = []
for k, v in zip(Sender, AddrBook):
    dEmail1 = dEmail[dEmail['from'] == k]
    t = [1 if (bool(set(j) & set(v)) == True) else 0 for j in dEmail1['to']]
    c = [1 if (bool(set(j) & set(v)) == True) else 0 for j in dEmail1['cc']]
    bc = [1 if (bool(set(j) & set(v)) == True) else 0 for j in dEmail1['bcc']]
    dEmail1['t'] = t
    dEmail1['c'] = c
    dEmail1['bc'] = bc
    dEmail2 = dEmail1[(dEmail1['t']==1) | (dEmail1['c']==1) | (dEmail1['bc']==1)]
    dEmail2 = dEmail2[['tfidffeat', 'to', 'from', 'cc', 'bcc']]
    appended_data.append(dEmail2)
appended_data = pd.concat(appended_data, axis=0)

########## Create a train-testing data set (take last 'm' messages from each sender)

df_train = []
df_test = []
m = 20 # number of test messages

for l in Sender:
    df = appended_data[appended_data['from'] == l]
    if len(df > 50):
        df1 = df.tail(m)
        df2 = df.head(len(df) - m)
        df_train.append(df2)
        df_test.append(df1)
df_train = pd.concat(df_train, axis=0)
df_test = pd.concat(df_test, axis=0)


########## Compute centroid

dFeatures = tfidfCentroid(df_train)  

########## Testing if the best ranked name appear in to or cc or bcc

sender_ = list(set(df_test['from']))
finalDf = []
for kk in sender_:
    df = df_test[df_test['from'] == kk]
    df = df.reset_index(drop=True)
    df1 = dFeatures[dFeatures['sender'] == kk]
    df1 = df1.reset_index(drop=True)
    PredictedRecipient = []
    for ll in range(len(df)):
        recipient = []
        score = []
        for mm in range(len(df1)):
            l1 = df['tfidffeat'][ll]
            l2 = df1['tfidfcentroid'][mm]
            Sc = similarity(l1,l2)
            recipient.append(df1['receiver'][mm])
            score.append(Sc)
        d = pd.DataFrame({'recipients': recipient,'score': score})
        d = d.sort_values(['score'], ascending=False)
        predictedPerson = d['recipients'].iloc[0]
        PredictedRecipient.append(predictedPerson)
    df['PredictedRecipient'] = PredictedRecipient
    finalDf.append(df)
finalDf = pd.concat(finalDf, axis=0)

############# Compute  Accuracy

dPerformance = compute_perfromance(finalDf)
  
## Networking aspect



G = nx.from_pandas_dataframe(dFeatures, 'sender', 'receiver')

plt.figure(figsize=(20,20))
pos = nx.spring_layout(G, k=.1)
nx.draw_networkx(G, pos, node_size=25, node_color='red', with_labels=True, edge_color='blue')
plt.show()    




    
    


