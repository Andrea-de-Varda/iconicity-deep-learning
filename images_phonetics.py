#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 11:30:41 2020

@author: anonymous
"""

import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Embedding, Dense, Dropout, LSTM
from tensorflow.keras.applications import VGG16, Xception
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import glob
import panphon
import epitran
import re
from os import chdir, getcwd
import pickle
from sPickle import sPickle
import random
import pandas as pd
import statistics
import researchpy as rp
from scipy.stats import ttest_ind
import nltk
from word2word import Word2word
import gc
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import os.path
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from itertools import combinations
from scipy.stats import pearsonr


getcwd()
wd = "/your_wd/"
chdir(wd)

# starting from the THINGS dataset (https://osf.io/ykbne/)

images = '/your_wd/images/'
folders = glob.glob('/your_wd/images/*')

imagenames_list = []
for folder in folders:
    for f in glob.glob(folder+'/*.jpg'):
        imagenames_list.append(f)
len(imagenames_list)

read_images = []        
for image_path in imagenames_list:
    read_images.append(image.load_img(image_path, target_size=(224, 224)))
len(read_images)

model = VGG16(weights='imagenet', include_top=False) # transfer learning; extracting features
fts = []
nm = []
for img, name in zip(read_images, imagenames_list):
    x = preprocess_input(np.expand_dims(image.img_to_array(img), axis=0))
    features = model.predict(x).flatten()
    name1 = re.sub(r'(.*\/images\/)(.*?)(/.*\.jpg)', r'\2', name)
    name2 = re.sub(r'[0-9\_]', ' ', name1).strip()
    fts.append(features)
    nm.append(name2)

fts = np.array(fts)
fts.shape # (26107, 25088)

#with open('fts', 'wb') as f:
#    pickle.dump(fts, f)
with open('fts', 'rb') as handle:
    fts = pickle.load(handle)
    
#with open('nm', 'wb') as f:
#    pickle.dump(nm, f)
with open('nm', 'rb') as handle:
    nm = pickle.load(handle)
    
# remove COMPOUNDS (might add noise)
features = []
names = []
for img, word in zip(fts, nm):
    if len(word.split()) > 1:
        pass
    else:
        features.append(img)
        names.append(word)
print("Len =", len(features), "=", len(names), "\nLen (set) =", len(set(names))) # 22268 items

# BILIGUAL DICTIONARIES FROM https://pypi.org/project/word2word/ and FastText

def trans(language):
    l2l = Word2word("en", language)
    fastlist = [couple.split() for couple in open("en-"+language+".txt", encoding="utf-8").read().split("\n")]
    d = {}
    for item in fastlist:
        if len(item) == 2:
            if item[0] == item[1]: # exclude translations that are identical to the source word
                pass
            else:
                d[item[0]] = item[1]
    c = 0
    c1 = 0
    lang_dict = {}
    for item in set(names):
        try: 
            candidate_trans = l2l(item)
            stop_counter = 0
            for word in candidate_trans:
                if word == item:
                    pass
                else:
                    if stop_counter == 0:
                        lang_dict[item] = word # word2word
                        stop_counter += 1
        except KeyError:
            try:
                lang_dict[item] = d[item] # FastText dictionary
                c += 1
            except KeyError:
                pass
                c1 += 1
    print("length (should be 1549) = ", len(lang_dict.keys()), "\tFastText =", round(((c/1549)*100), 2), "%", "\tNoTranslation =", round(((c1/1549)*100), 2), "%")
    return lang_dict

# find how many items have valid translations in all the languages
languages = ["ar", "hu", "id", "vi", "tr"]
all_ = []
for language in languages:
    a = [item for item in trans(language).keys()]
    all_.append(set(a))
u = set.intersection(*all_)
len(u) # 1161

ar = trans("ar") # Arabic, Afroasiatic
hu = trans("hu") # Hungarian, Uralic
dd = trans("id") # Indonesian, Austronesian
vi = trans("vi") # Vietnamese, Austroasiatic
tr = trans("tr") # Turkish, Turkic

img_df = []
for img, word in zip(features, names):
    if word in u:
        arabic = ar[word]
        hungarian = hu[word]
        indonesian = dd[word]
        vietnamese = vi[word]
        turkish = tr[word]
        img_df.append([word, arabic, hungarian, indonesian, vietnamese, turkish, img])
img_df = pd.DataFrame(img_df, columns=["word", "arabic", "hungarian", "indonesian", "vietnamese", "turkish", "img"]) # 16820 rows

#with open('img_df', 'wb') as f:
#    pickle.dump(img_df, f)
with open('img_df', 'rb') as handle:
    img_df = pickle.load(handle)

# I should test on items that (A) are not in the training set and (B) are not in the same language: this way I could completely exclude the impact of etymologically related words. I need the two dfs not to share any image, but also not to share any word: the splitting treshold should be set on the SET of words, not on the words themselves.
    
train_seed = set(random.sample(u, int(round(len(u)*80/100, 0)))); print("80% =", len(train_seed)) # 80% training --> 929 items
test_seed = u-train_seed; print("20% =", len(test_seed)) # 20% test --> 232 items
len(train_seed)+len(test_seed) 
train_seed&test_seed # empty intersection

# SAVING train and test seeds for reproducibility
#with open('train_seed', 'wb') as f:
#    pickle.dump(train_seed, f)
with open('train_seed', 'rb') as handle:
    train_seed = pickle.load(handle)
#with open('test_seed', 'wb') as f:
#    pickle.dump(test_seed, f)
with open('test_seed', 'rb') as handle:
    test_seed = pickle.load(handle)

train =  img_df[img_df["word"].isin(train_seed)]; train # 13397 items
test = img_df[img_df["word"].isin(test_seed)]; test # 3423 items

f = open("languages_epi&fasttext", "r").read().split('\n')
dict_ipa = {}
for item in f:
    a = re.sub(r"([A-Z][a-z]+)([ \t]+)([a-z]+\-[A-Z][a-z]+)", r"\1", item)
    b = re.sub(r"([A-Z][a-z]+)([ \t]+)([a-z]+\-[A-Z][a-z]+)", r"\3", item)
    dict_ipa.update( {a : b} )
    
ft = panphon.FeatureTable()
phon_features = ['syl', 'son', 'cons', 'cont', 'delrel', 'lat', 'nas', 'strid', 'voi', 'sg', 'cg', 'ant', 'cor', 'distr', 'lab', 'hi', 'lo', 'back', 'round', 'velaric', 'tense', 'long']

def phon_vectorizer(data, lang):
    epi = epitran.Epitran(dict_ipa.get(lang))
    phon = []
    for word in data:
            ipa_word = epi.transliterate(word)
            phon_vec = ft.word_array(phon_features, ipa_word)
            phon.append(phon_vec)
    phon = tf.keras.preprocessing.sequence.pad_sequences(np.array(phon), padding='post', maxlen=15) # set maxlen
    print("(samples, timesteps, features) =", phon.shape) 
    return phon
    
##############################################################################################################################################################################

def cos_sim(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

# Try all train-test combinations
    
ar_train = phon_vectorizer(train["arabic"], "Arabic"); print("ar_train shape =", ar_train.shape)
hu_train = phon_vectorizer(train["hungarian"], "Hungarian"); print("hu_train shape =", hu_train.shape)
id_train = phon_vectorizer(train["indonesian"], "Indonesian"); print("id_train shape =", id_train.shape)
vi_train = phon_vectorizer(train["vietnamese"], "Vietnamese"); print("vi_train shape =", vi_train.shape)
tr_train = phon_vectorizer(train["turkish"], "Turkish"); print("tr_train shape =", tr_train.shape)
en_train = phon_vectorizer(train["word"], "English"); print("en_train shape =", en_train.shape)

train_dict = {"arabic" : ar_train, "hungarian" : hu_train, "indonesian" : id_train, "vietnamese" : vi_train, "turkish" : tr_train, "english" : en_train}

#with open('train_dict', 'wb') as f:
#    pickle.dump(train_dict, f)
with open('train_dict', 'rb') as handle:
    train_dict = pickle.load(handle)

ar_test = phon_vectorizer(test["arabic"], "Arabic"); print("ar_test shape =", ar_test.shape)
hu_test = phon_vectorizer(test["hungarian"], "Hungarian"); print("hu_test shape =", hu_test.shape)
id_test = phon_vectorizer(test["indonesian"], "Indonesian"); print("id_test shape =", id_test.shape)
vi_test = phon_vectorizer(test["vietnamese"], "Vietnamese"); print("vi_test shape =", vi_test.shape)
tr_test = phon_vectorizer(test["turkish"], "Turkish"); print("tr_test shape =", tr_test.shape)
en_test = phon_vectorizer(test["word"], "English"); print("en_test shape =", en_test.shape)

test_dict = {"arabic" : ar_test, "hungarian" : hu_test, "indonesian" : id_test, "vietnamese" : vi_test, "turkish" : tr_test, "english" : en_test}

#with open('test_dict', 'wb') as f:
#    pickle.dump(test_dict, f)
with open('test_dict', 'rb') as handle:
    test_dict = pickle.load(handle)


def network_multilingual(language):
    with open('img_df', 'rb') as handle:
        img_df = pickle.load(handle)
    with open('train_seed', 'rb') as handle:
        train_seed = pickle.load(handle)
    with open('test_seed', 'rb') as handle:
        test_seed = pickle.load(handle)    
    train =  img_df[img_df["word"].isin(train_seed)]
    test = img_df[img_df["word"].isin(test_seed)]
    del img_df
    with open('train_dict', 'rb') as handle:
        train_dict = pickle.load(handle)
    with open('test_dict', 'rb') as handle:
        test_dict = pickle.load(handle)
    ls = [w for w in ["arabic", "hungarian", "indonesian", "vietnamese", "turkish", "english"] if w != language]
    input_train = []
    target_train = []
    for l in ls: 
        input_train.append(train_dict[l])
        target_train.append(np.array([vec for vec in train["img"]]))
    input_train = np.concatenate((input_train)); print("input_train shape (66985, 15, 22) =", input_train.shape) 
    target_train = np.concatenate((target_train)); print("target_train shape (66985, 25088) =", target_train.shape)
    input_test = test_dict[language]; print("input_test shape (3423, 15, 22) =", input_test.shape)
    target_test = np.array([vec for vec in test["img"]]); print("target_test shape (3423, 25088) =", target_test.shape)
    # LSTM - multilingual
    print("Starting to train LSTM - MULTILINGUAL CONDITION")
    model=keras.models.Sequential()
    model.add(keras.layers.Masking(mask_value=0., input_shape=(15, 22)))
    model.add(keras.layers.LSTM(units=500, batch_input_shape=(None, 15, 22), dropout=0.2, recurrent_dropout=0.2, return_sequences=False))
    model.add(keras.layers.Dense(25088, activation='relu'))
    model.compile(loss='cosine_similarity', optimizer='adam', metrics=[tf.keras.metrics.CosineSimilarity(axis=1)])
    model.summary()
    model.fit(input_train, target_train, epochs=1, validation_data=(input_test, target_test))
    model.save("model_"+language+".h5")
    results = model.evaluate(input_test, target_test); print("RESULTS (multilingual) =", results)
    prediction = model.predict(input_test)
    with open('prediction'+language, 'wb') as f:
        pickle.dump(prediction, f)
        
def network_random(language):
    with open('img_df', 'rb') as handle:
        img_df = pickle.load(handle)
    with open('train_seed', 'rb') as handle:
        train_seed = pickle.load(handle)
    with open('test_seed', 'rb') as handle:
        test_seed = pickle.load(handle)    
    train =  img_df[img_df["word"].isin(train_seed)]
    test = img_df[img_df["word"].isin(test_seed)]
    del img_df
    with open('train_dict', 'rb') as handle:
        train_dict = pickle.load(handle)
    with open('test_dict', 'rb') as handle:
        test_dict = pickle.load(handle)
    ls = [w for w in ["arabic", "hungarian", "indonesian", "vietnamese", "turkish", "english"] if w != language]
    input_train = []
    target_train = []
    for l in ls: 
        input_train.append(train_dict[l])
        target_train.append(np.array([vec for vec in train["img"]]))
    input_train = np.concatenate((input_train)); print("input_train shape (66985, 15, 22) =", input_train.shape) 
    target_train = np.concatenate((target_train)); print("target_train shape (66985, 25088) =", target_train.shape)
    input_test = test_dict[language]; print("input_test shape (3423, 15, 22) =", input_test.shape)
    target_test = np.array([vec for vec in test["img"]]); print("target_test shape (3423, 25088) =", target_test.shape)
    print("Starting to train LSTM - RANDOM CONDITION")
    np.random.seed(0)
    np.random.shuffle(target_train)
    s_model=keras.models.Sequential()
    s_model.add(keras.layers.Masking(mask_value=0., input_shape=(15, 22)))
    s_model.add(keras.layers.LSTM(units=500, batch_input_shape=(None, 15, 22), dropout=0.2, recurrent_dropout=0.2, return_sequences=False))
    s_model.add(keras.layers.Dense(25088, activation='relu'))
    s_model.compile(loss='cosine_similarity', optimizer='adam', metrics=[tf.keras.metrics.CosineSimilarity(axis=1)])
    s_model.summary()
    s_model.fit(input_train, target_train, epochs=1, validation_data=(input_test, target_test))
    s_model.save("s_model_"+language+".h5")
    s_results = s_model.evaluate(input_test, target_test); print("RESULTS (random) =", s_results)
    s_prediction = s_model.predict(input_test)
    with open('s_prediction'+language, 'wb') as f:
        pickle.dump(s_prediction, f)

for language in ["arabic", "hungarian", "indonesian", "vietnamese", "turkish"]:
	network_multilingual(language)        
	network_random(language)

def df_maker(language):
    with open('s_prediction'+language, 'rb') as handle:
        s_prediction = pickle.load(handle)
    with open('prediction'+language, 'rb') as handle:
        prediction = pickle.load(handle)  
    results_df = pd.DataFrame(zip(test["word"], test["img"], prediction, s_prediction), columns=["word", "img", "pred", "shuf"])
    cos = []
    s_cos = []
    for index, row in results_df.iterrows():
        cos.append(cos_sim(row['img'], row['pred']))
        s_cos.append(cos_sim(row['img'], row['shuf']))
    results_df['cos'] = [value for value in cos]
    results_df['s_cos'] = [value for value in s_cos]
    print("Mean cos (multilingual) =", statistics.mean(results_df['cos']))
    print("Mean cos (random) =", statistics.mean(results_df['s_cos']))
    print("T-test =", ttest_ind(results_df['cos'], results_df['s_cos'])) 
    with open('results_df'+language, 'wb') as f:
        pickle.dump(results_df, f)
        
for language in ["arabic", "hungarian", "indonesian", "vietnamese", "turkish"]:
    df_maker(language)

from researchpy import ttest
from collections import Counter

for df in ["results_dfarabic", "results_dfhungarian", "results_dfindonesian", "results_dfvietnamese", "results_dfturkish"]:
    with open(df, 'rb') as handle:
        results_df = pickle.load(handle) 
        print(ttest(results_df["cos"], results_df["s_cos"], group1_name= None, group2_name= None, equal_variances= True, paired= True, correction= None), "\n\n")


