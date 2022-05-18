#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 11:55:24 2021

@author: anonymous
"""

import epitran, panphon
import pandas as pd
from os import chdir
import numpy as np
import re
from word2word import Word2word
import random
import pickle
import tensorflow as tf
from tensorflow import keras
from sklearn.utils import compute_class_weight
import os
from statistics import stdev, mean
from imblearn.over_sampling import RandomOverSampler
os.environ['PYTHONHASHSEED'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
np.random.seed(0)
tf.random.set_seed(0)

oversample = RandomOverSampler() 

wd = "/your_wd/"
chdir(wd)

df = pd.read_csv("all.num.o5", sep="\s") # http://www.kilgarriff.co.uk/bnc-readme.html#raw
df.columns = ["freq", "word", "pos", "freq2"]

coarse = {'aj0': 'a',
 'ajc': 'a',
 'ajs': 'a',
 'at0': 'd',
 'av0': 'adv',
 'avp': 'adv',
 'avq': 'adv',
 'cjc': 'c',
 'cjs': 'c',
 'cjt': 'c',
 'crd': 'a',
 'dps': 'd',
 'dt0': 'd',
 'dtq': 'd',
 'ex0': 'd',
 'itj': 'int',
 'nn0': 'n',
 'nn1': 'n',
 'nn2': 'n',
 'np0': 'n',
 'ord': 'a',
 'pni': 'pron',
 'pnp': 'pron',
 'pnq': 'pron',
 'pnx': 'pron',
 'pos': 'pron',
 'prf': 'prep',
 'prp': 'prep',
 'to0': 'inf',
 'vbb': 'v',
 'vbd': 'v',
 'vbg': 'v',
 'vbi': 'v',
 'vbn': 'v',
 'vbz': 'v',
 'vdb': 'v',
 'vdd': 'v',
 'vdg': 'v',
 'vdi': 'v',
 'vdn': 'v',
 'vdz': 'v',
 'vhb': 'v',
 'vhd': 'v',
 'vhg': 'v',
 'vhi': 'v',
 'vhn': 'v',
 'vhz': 'v',
 'vm0': 'modal',
 'vvb': 'v',
 'vvd': 'v',
 'vvg': 'v',
 'vvi': 'v',
 'vvn': 'v',
 'vvz': 'v',
 'xx0': 'adv'}

pos_dictionary = {}
for index, row in df.iterrows():
    try:
        pos_dictionary[row.word] = coarse[row.pos]
    except KeyError:
        pass

coarse_tags = set(coarse.values())
len(coarse_tags)

d = {}
d_argmax = {}
for i, p in zip(range(len(coarse_tags)), coarse_tags):
    v = np.zeros(len(coarse_tags))
    v[i] = 1
    d[p] = v
    d_argmax[p] = i

names = []; vecs = []; argmax_word = {}; vec_map = {}
for index, row in df.iterrows():
    if "-" in row.pos:
        pass
    else:
        try:
            coarse_pos = coarse[row.pos]
            names.append(row.word)
            vecs.append(d[coarse_pos])
            argmax_word[row.word] = d_argmax[coarse_pos]
            vec_map[row.word] = d[coarse_pos]
        except KeyError: # remove single letters, other PoSs not relevant (see coarse)
            pass

def trans(language):
    l2l = Word2word("en", language)
    fastlist = [couple.split() for couple in open("dicts/en-"+language+".txt", encoding="utf-8").read().split("\n")]
    d = {}
    for item in fastlist:
        if len(item) == 2:
            if item[0] == item[1]: # exclude translations that are identical to the source word
                pass
            else:
                d[item[0]] = item[1]
    c = 0
    c1 = 0
    c_w2w = 0
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
                        c_w2w += 1
                        stop_counter += 1
        except KeyError:
            try:
                lang_dict[item] = d[item] # FastText dictionary
                c += 1
            except KeyError:
                pass
                c1 += 1
    #print("length = ", len(lang_dict.keys()),"\tWord2Word =", round(((c_w2w/len(set(names)))*100), 2), "\tFastText =", round(((c/len(set(names)))*100), 2), "%", "\tNoTranslation =", round(((c1/len(set(names)))*100), 2), "%")
    print(round(((c_w2w/len(set(names)))*100), 2), "\% & ", round(((c/len(set(names)))*100), 2), "\% & ", round(((c1/len(set(names)))*100), 2), "\%")
    return lang_dict

languages = ["ar", "hu", "id", "vi", "tr"]
all_ = []
for language in languages:
    a = [item for item in trans(language).keys()]
    all_.append(set(a))
u = set.intersection(*all_)
len(u) # 24246

train_seed = set(random.sample(u, int(round(len(u)*0.5, 0)))); print("50% =", len(train_seed)) # 50%
test_seed = u-train_seed; print("50% =", len(test_seed)) # 50% items
len(train_seed)+len(test_seed) 
train_seed&test_seed # empty intersection

ar = trans("ar") # Arabic, Afroasiatic
hu = trans("hu") # Hungarian, Uralic
dd = trans("id") # Indonesian, Austronesian
vi = trans("vi") # Vietnamese, Austroasiatic
tr = trans("tr") # Turkish, Turkic

X, y, idx_map= [], [], {} 
for index, item in enumerate(train_seed):
    X.append(index)
    y.append(argmax_word[item])
    idx_map[index] = item
X, y = oversample.fit_resample(np.array(X).reshape(-1, 1), np.array(y))
y.shape
X.shape
sum(y)

train = []
for idx in X:
    w = idx_map[idx[0]]
    vec = np.delete(vec_map[w], 0) # no infinitive markers, remove (see below)
    train.append([w, argmax_word[w], vec, ar[w], hu[w], dd[w], vi[w], tr[w]])
train = pd.DataFrame(train, columns=["word", "argmax", "vec", "arabic", "hungarian", "indonesian", "vietnamese", "turkish"])

test = []
for w in test_seed:
    vec = np.delete(vec_map[w], 0) # no infinitive markers, remove (see below)
    test.append([w, argmax_word[w], vec, ar[w], hu[w], dd[w], vi[w], tr[w]])
test = pd.DataFrame(test, columns=["word", "argmax", "vec", "arabic", "hungarian", "indonesian", "vietnamese", "turkish"])

with open('train', 'wb') as f:
    pickle.dump(train, f)
with open('test', 'wb') as f:
    pickle.dump(test, f)    

f = open("languages_epi&fasttext", "r").read().split('\n')
dict_ipa = {}
for item in f:
    a = re.sub(r"([A-Z][a-z]+)([ \t]+)([a-z]+\-[A-Z][a-z]+)", r"\1", item)
    b = re.sub(r"([A-Z][a-z]+)([ \t]+)([a-z]+\-[A-Z][a-z]+)", r"\3", item)
    dict_ipa.update( {a : b} )
dict_ipa.update({"English" : "eng-Latn"})
    
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

ar_train = phon_vectorizer(train["arabic"], "Arabic"); print("ar_train shape =", ar_train.shape)
hu_train = phon_vectorizer(train["hungarian"], "Hungarian"); print("hu_train shape =", hu_train.shape)
id_train = phon_vectorizer(train["indonesian"], "Indonesian"); print("id_train shape =", id_train.shape)
vi_train = phon_vectorizer(train["vietnamese"], "Vietnamese"); print("vi_train shape =", vi_train.shape)
tr_train = phon_vectorizer(train["turkish"], "Turkish"); print("tr_train shape =", tr_train.shape)
en_train = phon_vectorizer(train["word"], "English"); print("en_train shape =", en_train.shape)

train_dict = {"arabic" : ar_train, "hungarian" : hu_train, "indonesian" : id_train, "vietnamese" : vi_train, "turkish" : tr_train, "english" : en_train}

with open('train_dict', 'wb') as f:
   pickle.dump(train_dict, f)
# with open('train_dict', 'rb') as handle:
#     train_dict = pickle.load(handle)

train_dict["arabic"].shape    
ar_test = phon_vectorizer(test["arabic"], "Arabic"); print("ar_test shape =", ar_test.shape)
hu_test = phon_vectorizer(test["hungarian"], "Hungarian"); print("hu_test shape =", hu_test.shape)
id_test = phon_vectorizer(test["indonesian"], "Indonesian"); print("id_test shape =", id_test.shape)
vi_test = phon_vectorizer(test["vietnamese"], "Vietnamese"); print("vi_test shape =", vi_test.shape)
tr_test = phon_vectorizer(test["turkish"], "Turkish"); print("tr_test shape =", tr_test.shape)
en_test = phon_vectorizer(test["word"], "English"); print("en_test shape =", en_test.shape)

test_dict = {"arabic" : ar_test, "hungarian" : hu_test, "indonesian" : id_test, "vietnamese" : vi_test, "turkish" : tr_test, "english" : en_test}

with open('newdata/new/test_dict', 'wb') as f:
   pickle.dump(test_dict, f)
# with open('newdata/new/test_dict', 'rb') as handle:
#     test_dict = pickle.load(handle)

###################
# network testing #
###################

def network_multilingual(language):
    ls = [w for w in ["arabic", "hungarian", "indonesian", "vietnamese", "turkish", "english"] if w != language]
    input_train = []
    target_train = []
    for l in ls: 
        input_train.append(train_dict[l])
        target_train.append(np.array([vec for vec in train["vec"]]))
    input_train = np.concatenate((input_train)); print("input_train shape =", input_train.shape) 
    target_train = np.concatenate((target_train)); print("target_train shape =", target_train.shape)
    print("Sum =", sum(target_train))
    input_test = test_dict[language]; print("input_test shape =", input_test.shape)
    target_test = np.array([vec for vec in test["vec"]]); print("target_test shape =", target_test.shape)
    print("Sum =", sum(target_test))
    # LSTM - multilingual
    print("Starting to train LSTM - MULTILINGUAL CONDITION")
    model=keras.models.Sequential()
    model.add(keras.layers.Masking(mask_value=0., input_shape=(15, 22)))
    model.add(keras.layers.LSTM(units=25, batch_input_shape=(None, 15, 22), dropout=0.2, recurrent_dropout=0.2, return_sequences=False))
    model.add(keras.layers.Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    model.fit(input_train, target_train, epochs=1, validation_data=(input_test, target_test))
    model.save("model_25_"+language+".h5")
    results = model.evaluate(input_test, target_test); print("RESULTS (multilingual) =", results)
    prediction = model.predict(input_test)
    with open('prediction_25_'+language, 'wb') as f:
        pickle.dump(prediction, f)
    return results
        
def network_random(language): 
    ls = [w for w in ["arabic", "hungarian", "indonesian", "vietnamese", "turkish", "english"] if w != language]
    input_train = []
    target_train = []
    for l in ls: 
        input_train.append(train_dict[l])
        target_train.append(np.array([vec for vec in train["vec"]]))
    input_train = np.concatenate((input_train)); print("input_train shape =", input_train.shape) 
    target_train = np.concatenate((target_train)); print("target_train shape =", target_train.shape)
    print("Sum =", sum(target_train))
    input_test = test_dict[language]; print("input_test shape =", input_test.shape)
    target_test = np.array([vec for vec in test["vec"]]); print("target_test shape =", target_test.shape)
    print("Sum =", sum(target_test))
    print("Starting to train LSTM - RANDOM CONDITION")
    np.random.seed(0)
    np.random.shuffle(target_train)
    s_model=keras.models.Sequential()
    s_model.add(keras.layers.Masking(mask_value=0., input_shape=(15, 22)))
    s_model.add(keras.layers.LSTM(units=25, batch_input_shape=(None, 15, 22), dropout=0.2, recurrent_dropout=0.2, return_sequences=False))
    s_model.add(keras.layers.Dense(10, activation='softmax'))
    s_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    s_model.summary()
    s_model.fit(input_train, target_train, epochs=1, validation_data=(input_test, target_test))
    s_model.save("s_model_25_"+language+".h5")
    s_results = s_model.evaluate(input_test, target_test); print("RESULTS (random) =", s_results)
    s_prediction = s_model.predict(input_test)
    with open('s_prediction_25_'+language, 'wb') as f:
        pickle.dump(s_prediction, f)
    return s_results

res = []
for l in ["arabic", "hungarian", "indonesian", "vietnamese", "turkish", "english"]:
    print("\n\n")
    print(l.upper(), "\n")
    results = network_multilingual(l)
    s_results = network_random(l)
    res.append([l, round(results[0], 4), round(results[1], 4), round(s_results[0], 4), round(s_results[1], 4)])

############################################################################################
############################################################################################
############################################################################################


test["pred_arabic"] = [v for v in pickle.load(open('prediction_25_arabic', 'rb'))]
test["pred_hungarian"] = [v for v in pickle.load(open('prediction_25_hungarian', 'rb'))]
test["pred_indonesian"] = [v for v in pickle.load(open('prediction_25_indonesian', 'rb'))]
test["pred_vietnamese"] = [v for v in pickle.load(open('prediction_25_vietnamese', 'rb'))]
test["pred_turkish"] = [v for v in pickle.load(open('prediction_25_turkish', 'rb'))]
test["pred_english"] = [v for v in pickle.load(open('prediction_25_english', 'rb'))]

test["s_pred_arabic"] = [v for v in pickle.load(open('s_prediction_25_arabic', 'rb'))]
test["s_pred_hungarian"] = [v for v in pickle.load(open('s_prediction_25_hungarian', 'rb'))]
test["s_pred_indonesian"] = [v for v in pickle.load(open('s_prediction_25_indonesian', 'rb'))]
test["s_pred_vietnamese"] = [v for v in pickle.load(open('s_prediction_25_vietnamese', 'rb'))]
test["s_pred_turkish"] = [v for v in pickle.load(open('s_prediction_25_turkish', 'rb'))]
test["s_pred_english"] = [v for v in pickle.load(open('s_prediction_25_english', 'rb'))]

test.columns

cross_entropy = tf.keras.losses.CategoricalCrossentropy()
d_acc = {"arabic":[], "hungarian":[], "indonesian":[], "vietnamese":[], "turkish":[], "english":[]}
d_loss = {"arabic":[], "hungarian":[], "indonesian":[], "vietnamese":[], "turkish":[], "english":[]}
for l in ["arabic", "hungarian", "indonesian", "vietnamese", "turkish", "english"]:
    col = test["pred_"+l]
    for i, value in enumerate(col):
        v = np.zeros(10)
        v[np.argmax(value)] = 1
        d_loss[l].append(cross_entropy(value, test.iloc[i]["vec"]).numpy())
        x = v == test.iloc[i]["vec"]
        if x.all() == True:
        #if x == True:
            d_acc[l].append(1)
        else:
            d_acc[l].append(0)
      
d_acc_r = {"arabic":[], "hungarian":[], "indonesian":[], "vietnamese":[], "turkish":[], "english":[]}
d_loss_r = {"arabic":[], "hungarian":[], "indonesian":[], "vietnamese":[], "turkish":[], "english":[]}
for l in ["arabic", "hungarian", "indonesian", "vietnamese", "turkish", "english"]:
    col = test["s_pred_"+l]
    for i, value in enumerate(col):
        v = np.zeros(10)
        v[np.argmax(value)] = 1
        d_loss[l].append(cross_entropy(value, test.iloc[i]["vec"]).numpy())
        x = v == test.iloc[i]["vec"]
        if x.all() == True:
            d_acc_r[l].append(1)
        else:
            d_acc_r[l].append(0)

# https://en.wikipedia.org/wiki/McNemar%27s_test 
# McNemar test is a statistical test used on paired nominal data
# see also here https://machinelearningmastery.com/mcnemars-test-for-machine-learning/
#
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.sandbox.stats.runs import mcnemar as mcnemar1
#from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score, accuracy_score
#from sklearn.metrics import precision_recall_fscore_support

# important: weighted average
average="weighted"
for l in ["arabic", "hungarian", "indonesian", "vietnamese", "turkish", "english"]:
    col = [np.argmax(v) for v in test["pred_"+l]]
    col_r = [np.argmax(v) for v in test["s_pred_"+l]]
    true = [np.argmax(v) for v in test.vec]
    print("\n", l.upper(), "cross-lingual")
    #print(precision_recall_fscore_support(col, true, average='weighted'))
    #print(precision_recall_fscore_support(col_r, true, average='weighted'))
    #print(col[:20], col_r[:20], true[:20])
    print(round(accuracy_score(true, col), 4), "&", round(precision_score(true, col, average=average), 4),"&", round(recall_score(true, col, average=average), 4),"&", round(f1_score(true, col, average=average), 4),"&", round(accuracy_score(true, col_r), 4),"&", round(precision_score(true, col_r, average=average), 4),"&", round(recall_score(true, col_r, average=average), 4),"&", round(f1_score(true, col_r, average=average), 4),"&")

len(test)

def McNemar(language):
    random_acc = d_acc_r[language]
    multi_acc = d_acc[language]
    acc = zip(multi_acc, random_acc)
    a, b, c, d = 0, 0, 0, 0
    for z in acc:
        if z == (1, 1):
            a += 1
        if z == (1, 0):
            b += 1
        if z == (0, 1):
            c += 1
        if z == (0, 0):
            d += 1
    table = [[a, b], [c, d]]
    print(table, "\n")
    #if min([a, b, c, d]) < 25:
    if b+c < 25:
        result = mcnemar(table, exact=True)
        result1 = mcnemar1(table, exact=True)
        print("EXACT TEST (< 25)\n")
    else:
        result = mcnemar(table, exact=False, correction=True)
        result1 = mcnemar1(table, exact=False, correction=True)
        print("Standard calculation (> 25)\n")
    print(result1)
    print('statistic=%.3f, p-value=%.3f' % (float(result.statistic), float(result.pvalue)))
    
for l in ["arabic", "hungarian", "indonesian", "vietnamese", "turkish", "english"]:
    print("\n\n",l.upper())
    McNemar(l)

test["acc_arabic"] = d_acc["arabic"]
test["acc_hungarian"] = d_acc["hungarian"]
test["acc_indonesian"] = d_acc["indonesian"]
test["acc_vietnamese"] = d_acc["vietnamese"]
test["acc_turkish"] = d_acc["turkish"]
test["acc_english"] = d_acc["english"]

pos_test = []
for index, row in test.iterrows():
    pos_test.append(pos_dictionary[row.word])
test["pos"] = pos_test

ds = []
for l in ["arabic", "hungarian", "indonesian", "vietnamese", "turkish", "english"]:
    d = test.groupby('pos', as_index=False)['acc_'+l].mean()
    d["l"] = [l for i in range(len(d))]
    d.columns = ["pos", "acc", "l"]
    ds.append(d)
    
ds = pd.concat(ds)
ds_tot = ds.groupby('pos', as_index=False)['acc'].mean().sort_values(by="acc", ascending=False)
ds_tot["l"] = ["Mean" for n in range(0, len(ds_tot.acc))]
ds = pd.concat([ds, ds_tot])

# plotting 
import seaborn as sns
import matplotlib.pyplot as plt
# import matplotlib as mpl


plot_data = {}
for pos in ds_tot.pos:
    d_temp = list(ds["acc"][ds.pos == pos])
    plot_data[pos] = d_temp
plot_data = pd.DataFrame(plot_data, index=[l for l in ["Arabic", "Hungarian", "Indonesian", "Vietnamese", "Turkish", "English", "Mean"]])

font = {'family' : "Times New Roman",
        'weight' : 'normal',
        'size'   : 24}
plt.rc('font', **font)

fig, ax = plt.subplots(figsize=(18, 11))
sns.heatmap(plot_data, annot=True, cmap="Blues",linewidth=0.3, cbar_kws={"shrink": .8})

# compute average pairwise correlation
from itertools import combinations, permutations

rows = []
for index, row in plot_data.iterrows():
    rows.append(np.array(row))

corr = []
for c in combinations(rows, 2):
    corr.append(np.corrcoef(c)[0][1])
mean(corr)

