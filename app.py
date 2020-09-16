#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  19 2020
This script contains Python code for deploying the Russian Twitter Troll Classifier model on streamlit.io
@author: Jon Lindenauer
"""
import streamlit as st
import pickle
import sklearn
from PIL import Image
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import NMF
import nltk
from nltk.stem import WordNetLemmatizer

def tokenized_lem(text):
    tokens = nltk.word_tokenize(text)
    lem = []

    for item in tokens:
        lem.append(WordNetLemmatizer().lemmatize(item))

    return lem

# page heading
st.title('')
st.header('Right Troll Tweet Classifier')

# main image background
background = Image.open('troll_cover.jpg')
st.image(background, width=400)

# load pickled model, vectorizer and factorizer
model = pickle.load(open("my_pickled_model.p", "rb"))
cv = pickle.load(open("my_pickled_vectorizer.p", "rb"))
nmf = pickle.load(open("my_pickled_factorizer.p", "rb"))

# get text input from user
user_input = st.text_input("Enter Tweet:", "put text here")

user_lst = []
user_lst.append(user_input)

# vectorize the user_input ext
vector_txt = cv.transform(user_lst)

# matrix factorization of the vector_txt
factor_txt = nmf.transform(vector_txt)

# create features list
#input_df = []


# model threshold, prediction, hard classification, message list, output message
threshold = 0.583
#prediction = model.predict(factor_txt)[0]
prediction = int(model.predict_proba(factor_txt)[:, 1] > threshold)
#pred_prob = model.predict_proba(factor_txt)[:,1] >= threshold
message_array = ["Not Right Troll", "Right Troll"]
message_out = message_array[prediction]

# make the output message a title so it is BIG
st.title(message_out)
#st.write(model.predict_proba(factor_txt))
