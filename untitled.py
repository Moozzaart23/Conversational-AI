from gensim.models import Word2Vec
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import os
import numpy as np
from pymongo import MongoClient


def correct_encoding(dictionary):
    """Correct the encoding of python dictionaries so they can be encoded to mongodb
    inputs
    -------
    dictionary : dictionary instance to add as document
    output
    -------
    new : new dictionary with (hopefully) corrected encodings"""
    
    new = {}
    for key1, val1 in dictionary.items():
        # Nested dictionaries
        if isinstance(val1, dict):
            val1 = correct_encoding(val1)

        if isinstance(val1, np.bool_):
            val1 = bool(val1)

        if isinstance(val1, np.int64):
            val1 = int(val1)

        if isinstance(val1, np.float64):
            val1 = float(val1)
            
        if isinstance(val1, np.ndarray):
            val1 = val1.tolist()

        new[key1] = val1

    return new

def cleaning(data):
    for i in range(0, len(data)):
        temp = ""
        for j in range(0, len(data[i])):
            if(data[i][j] != ',' and data[i][j] != ')' and data[i][j] != '(' and data[i][j] != '"' and data[i][j] != ':' and data[i][j] != '!' and data[i][j] != '%' and not(data[i][j] >= '0' and data[i][j] <= '9') and data[i][j] != '-' and data[i][j] != '/' and not(data[i][j] >= 'a' and data[i][j] <= 'z') and not(data[i][j] >= 'A'  and data[i][j] <= 'Z') and data[i][j] != '/' and data[i][j] != '-'):
                temp=temp+data[i][j]
        data[i]=temp
    return data

def stopwords_removal(data):
    stop_file=open("stopwords-hi.txt",encoding='UTF-8').read()
    stopwords=stop_file.split()
    dictionary={}
    for word in stopwords:
        dictionary[word]=1
    for i in  range(len(data)):
        temp = ""
        words = data[i].split(" ")
        for j in words:
            if j == '?':
                continue
            elif j not in dictionary:
                temp += (j + " ")
        data[i]=temp
    return data

def tokenization(data):
    return data.split(" ")

## Preprocessing  
doc_content = ""
count = 0
corpus = []
for dirname in os.listdir('./Agriculture Data'):
    if dirname == '.DS_Store':
        continue
#     print(dirname)
    for filename in os.listdir('./Agriculture Data/' + str(dirname)):
#         print("\t"+filename)
        if filename == '.ipynb_checkpoints':
            continue
        filepath = './Agriculture Data/'+ dirname + '/' + filename
        doc = open(filepath, encoding='UTF-8').read()
        doc = doc[:-1]
        doc = doc.replace('\n', '')
        filename = filename.split('.')
        filename = filename[0][1:-1]
        doc_content += doc
        count +=1
        for x in filename.split(" "):
            corpus.append(x)
    for x in dirname.split(" "):
        corpus.append(x)
# corpus = tokenization(doc_content)
corpus1=[corpus]
# print(corpus1)


## Creating word2vec with IITB corpus

with open('monolingual/monolingual.hi', encoding="utf8") as f:
    head = [next(f) for x in range(10000)]
sen = []
for line in head:
    words = line.split(" ")
    sen.append(words)
allwords = []
for l in sen:
    allwords += l
corpus+=allwords

# print("कटाई" in corpus)

model = Word2Vec(corpus1,min_count=1)

# print(model['ज़मीन'])
## Creating dictionaries for crops

def getEmbedding(topic_sentence):
    emb = None
    sentence = topic_sentence.split(' ')
#     print(sentence)
    for i in range(len(sentence)):
        if i == 0:
            w = model[sentence[i]]
            emb = w
        else:
            w = model[sentence[i]]
            emb.setflags(write=1)
            emb += w
    return emb

client = MongoClient()
crops = client.crops


def reading_data():
    for dirname in os.listdir('./Agriculture Data'):
        if dirname == '.DS_Store':
            continue
    #     print(dirname)
        crop={}
        for filename in os.listdir('./Agriculture Data/' + str(dirname)):
    #         print("\t"+filename)
            if filename == '.ipynb_checkpoints':
                continue
            filepath = './Agriculture Data/'+ dirname + '/' + filename
            doc = open(filepath, encoding='UTF-8').read()
            doc = doc[:-1]
            doc = doc.replace('\n', '')
            filename = filename.split('.')
            filename = filename[0][1:-1]
            crop[filename]={}
            crop[filename]['Text'] = doc.replace('\n', '')
            crop[filename]['Embedding'] = getEmbedding(filename)
        collection=crops[dirname]
        crop=correct_encoding(crop)
        collection.insert_one(crop)

reading_data()

