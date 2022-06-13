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
from researchpy import corr_pair
from scipy.stats import ttest_rel, ttest_ind
from math import sqrt
from researchpy import ttest
os.environ['PYTHONHASHSEED'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
np.random.seed(1)
tf.random.set_seed(1)

wd = "/your_wd/"
chdir(wd)

w2v = open("data") # "data"
names = []
vecs = []
# https://www.marekrei.com/projects/vectorsets/
for line in w2v.readlines():
    try:
        w = re.sub("(.*?)(\s)(.*)", r"\1", line)
        if "ENTITY" in w:
            pass
        else:
            v = np.array([float(item) for item in re.sub("(.*?)(\s)(.*)", r"\3", line).split()])
            names.append(w)
            vecs.append(v)
    except ValueError:
        print("ValueError")
len(vecs)

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
    print("length = ", len(lang_dict.keys()), "\tFastText =", round(((c/1937423)*100), 2), "%", "\tNoTranslation =", round(((c1/1937423)*100), 2), "%")
    return lang_dict

# find how many items have valid translations in all the languages
languages = ["ar", "hu", "id", "vi", "tr"]
all_ = []
for language in languages:
    a = [item for item in trans(language).keys()]
    all_.append(set(a))
u = set.intersection(*all_)
len(u)


ar = trans("ar") # Arabic, Afroasiatic
hu = trans("hu") # Hungarian, Uralic
dd = trans("id") # Indonesian, Austronesian
vi = trans("vi") # Vietnamese, Austroasiatic
tr = trans("tr") # Turkish, Turkic

train_seed = set(random.sample(u, int(round(len(u)*80/100, 0)))); print("80% =", len(train_seed)) # 80% training
test_seed = u-train_seed; print("20% =", len(test_seed)) # 20% test
len(train_seed)+len(test_seed) 
train_seed&test_seed # empty intersection

#with open('train_seed', 'wb') as f:
#    pickle.dump(train_seed, f)
with open('train_seed', 'rb') as handle:
    train_seed = pickle.load(handle)
#with open('test_seed', 'wb') as f:
#    pickle.dump(test_seed, f)
with open('test_seed', 'rb') as handle:
    test_seed = pickle.load(handle)

multi_df = []
for vec, word in zip(vecs, names):
    if word in u:
        arabic = ar[word]
        hungarian = hu[word]
        indonesian = dd[word]
        vietnamese = vi[word]
        turkish = tr[word]
        multi_df.append([word, arabic, hungarian, indonesian, vietnamese, turkish, vec])
multi_df = pd.DataFrame(multi_df, columns=["word", "arabic", "hungarian", "indonesian", "vietnamese", "turkish", "vec"]) # 24612 rows

#with open('multi_df', 'wb') as f:
#    pickle.dump(multi_df, f)
with open('multi_df', 'rb') as handle:
    multi_df = pickle.load(handle)

train =  multi_df[multi_df["word"].isin(train_seed)]; train
test = multi_df[multi_df["word"].isin(test_seed)]; test

# deal with multiclass classification umbalance

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

# with open('train_dict', 'wb') as f:
#     pickle.dump(train_dict, f)
with open('train_dict', 'rb') as handle:
    train_dict = pickle.load(handle)
    
ar_test = phon_vectorizer(test["arabic"], "Arabic"); print("ar_test shape =", ar_test.shape)
hu_test = phon_vectorizer(test["hungarian"], "Hungarian"); print("hu_test shape =", hu_test.shape)
id_test = phon_vectorizer(test["indonesian"], "Indonesian"); print("id_test shape =", id_test.shape)
vi_test = phon_vectorizer(test["vietnamese"], "Vietnamese"); print("vi_test shape =", vi_test.shape)
tr_test = phon_vectorizer(test["turkish"], "Turkish"); print("tr_test shape =", tr_test.shape)
en_test = phon_vectorizer(test["word"], "English"); print("en_test shape =", en_test.shape)

test_dict = {"arabic" : ar_test, "hungarian" : hu_test, "indonesian" : id_test, "vietnamese" : vi_test, "turkish" : tr_test, "english" : en_test}

# with open('test_dict', 'wb') as f:
#    pickle.dump(test_dict, f)
with open('test_dict', 'rb') as handle:
    test_dict = pickle.load(handle)


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
    input_test = test_dict[language]; print("input_test shape =", input_test.shape)
    target_test = np.array([vec for vec in test["vec"]]); print("target_test shape =", target_test.shape)
    # LSTM - multilingual
    print("Starting to train LSTM - MULTILINGUAL CONDITION")
    model=keras.models.Sequential()
    model.add(keras.layers.Masking(mask_value=0., input_shape=(15, 22)))
    model.add(keras.layers.LSTM(units=50, batch_input_shape=(None, 15, 22), dropout=0.2, recurrent_dropout=0.2, return_sequences=False))
    model.add(keras.layers.Dense(100, activation=keras.layers.LeakyReLU(alpha=0.05)))
    model.compile(loss='cosine_similarity', optimizer='adam', metrics=[tf.keras.metrics.CosineSimilarity(axis=1)])
    model.summary()
    model.fit(input_train, target_train, epochs=1, validation_data=(input_test, target_test), shuffle=True)
    model.save("model_testing_"+language+".h5")
    results = model.evaluate(input_test, target_test); print("RESULTS (multilingual) =", results)
    prediction = model.predict(input_test)
    with open('prediction_testing_'+language, 'wb') as f:
        pickle.dump(prediction, f)
    return results, prediction
        
def network_random(language): 
    ls = [w for w in ["arabic", "hungarian", "indonesian", "vietnamese", "turkish", "english"] if w != language]
    input_train = []
    target_train = []
    for l in ls: 
        input_train.append(train_dict[l])
        target_train.append(np.array([vec for vec in train["vec"]]))
    input_train = np.concatenate((input_train)); print("input_train shape =", input_train.shape) 
    target_train = np.concatenate((target_train)); print("target_train shape =", target_train.shape)
    input_test = test_dict[language]; print("input_test shape =", input_test.shape)
    target_test = np.array([vec for vec in test["vec"]]); print("target_test shape =", target_test.shape)
    print("Starting to train LSTM - RANDOM CONDITION")
    np.random.seed(0)
    np.random.shuffle(target_train)
    s_model=keras.models.Sequential()
    s_model.add(keras.layers.Masking(mask_value=0., input_shape=(15, 22)))
    s_model.add(keras.layers.LSTM(units=50, batch_input_shape=(None, 15, 22), dropout=0.2, recurrent_dropout=0.2, return_sequences=False))
    s_model.add(keras.layers.Dense(100, activation=keras.layers.LeakyReLU(alpha=0.05)))
    s_model.compile(loss='cosine_similarity', optimizer='adam', metrics=[tf.keras.metrics.CosineSimilarity(axis=1)])
    s_model.summary()
    s_model.fit(input_train, target_train, epochs=1, validation_data=(input_test, target_test), shuffle=True)
    s_model.save("s_model_testing_"+language+".h5")
    s_results = s_model.evaluate(input_test, target_test); print("RESULTS (random) =", s_results)
    s_prediction = s_model.predict(input_test)
    with open('s_prediction__testing_'+language, 'wb') as f:
        pickle.dump(s_prediction, f)
    return s_results, s_prediction

def cos_sim(v1, v2):
    if len(v1) != len(v2):
        return 0.0
    num = np.dot(v1, v2)
    den_a = np.dot(v1, v1)
    den_b = np.dot(v2, v2)
    return num / (sqrt(den_a) * sqrt(den_b))

res = []
out_resdict = {}
for l in ["vietnamese", "hungarian", "indonesian", "arabic", "turkish", "english"]:
    print("\n\n\n")
    print(l.upper(), "\n")
    results, predictions = network_multilingual(l)
    s_results, s_predictions = network_random(l)
    res.append([l, round(results[0], 4), round(results[1], 4), round(s_results[0], 4), round(s_results[1], 4)])
    cos_r = []
    cos = []
    for pred, vec in zip(test["vec"], predictions):
        cos.append(cos_sim(pred, vec))
    for pred, vec in zip(test["vec"], s_predictions):
        cos_r.append(cos_sim(pred, vec))
    out_resdict[l+"_random"] = cos_r
    out_resdict[l] = cos
    print("\nT-test results", ttest(pd.Series(cos), pd.Series(cos_r), group1_name= None, group2_name= None, equal_variances= True, paired= True, correction= None))
print(pd.DataFrame(res, columns = ["language", "loss", "cosine", "loss_random", "cosine_random"]))
    
#with open('out_resdict_sem', 'wb') as f:
#    pickle.dump(out_resdict, f)
#with open('out_resdict_sem', 'rb') as handle:
#    out_resdict = pickle.load(handle)

for l in ["vietnamese", "hungarian", "indonesian", "arabic", "turkish", "english"]:
    rand = out_resdict[l+"_random"]
    cos = out_resdict[l]
    print(l, ttest(pd.Series(cos), pd.Series(rand), paired= True))
