#importing libraries
import numpy as np
import pandas as pd
data=pd.read_csv('Text_Similarity_Dataset.csv')
import gensim.downloader as api
from gensim.models import Word2Vec

import re
#importing pre-trained word2vec models
model=api.load('word2vec-.google-news-300')
#model = api.load("glove-twitter-25")
#model1=api.load("word2vec-google-news-300")

#saving model's weights
from sklearn.externals import joblib
joblib.dump(model,"weights.h5")

#model1=joblib.load("abc.h5")
#model.wv.most_similar("marvel",topn=10)

#importing NLP libraries
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
nltk.download('wordnet')

data.head()

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lemmatizer=WordNetLemmatizer()

#applying NLP and model based on cosine similarity

"""s1 = 'I have to support my family. I want to find a job &'
s2 = 'I want to find a job to support my family' """ 
for i in range(1,4025):  
   s1=data['text1'][i] 
   s2=data['text2'][i]

   s1.lower()
   s1=re.sub('[^a-zA-Z]',' ',s1)
 
   s1=s1.split()
 
 
   s1=[lemmatizer.lemmatize(word) for word in s1 if not word in set(stopwords.words('english'))]

   s2.lower()
   s2=re.sub('[^a-zA-Z]',' ',s2)
   s2=s2.split()
   s2=[lemmatizer.lemmatize(word) for word in s2 if not word in set(stopwords.words('english'))]
   s1=s1[0:100]
   s2=s2[0:100]
   distance = model.wv.n_similarity(s1, s2)
   if distance>0.5:
    data['Label'][i]=1
   else:
    data['Label'][i]=0





#saving results to csv
df=pd.read_csv('results(1).csv')
df['Unique_ID']=data['Unique_ID']
df["Label"]=data["Label"]
df.head()
df.to_csv('results(1).csv')