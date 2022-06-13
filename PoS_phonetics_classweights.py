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
os.environ['PYTHONHASHSEED'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
np.random.seed(0)
tf.random.set_seed(0)


wd = "/path/to/wd"
chdir(wd)

# http://www.kilgarriff.co.uk/bnc-readme.html#raw
df = pd.read_csv("word_data1.txt", sep="\s")
df.columns

pos = set(df["pos"])
d = {}
for i, p in zip(range(len(pos)), pos):
    v = np.zeros(len(pos))
    v[i] = 1
    d[p] = v
 
names = []
vecs = []
for index, row in df.iterrows():
    names.append(row.word)
    vecs.append(d[row.pos])


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
    print("length (should be 6318) = ", len(lang_dict.keys()), "\tFastText =", round(((c/6318)*100), 2), "%", "\tNoTranslation =", round(((c1/6318)*100), 2), "%")
    return lang_dict

# find how many items have valid translations in all the languages
languages = ["ar", "hu", "id", "vi", "tr"]
all_ = []
for language in languages:
    a = [item for item in trans(language).keys()]
    all_.append(set(a))
u = set.intersection(*all_)
len(u) # 5260

ar = trans("ar") # Arabic, Afroasiatic
hu = trans("hu") # Hungarian, Uralic
dd = trans("id") # Indonesian, Austronesian
vi = trans("vi") # Vietnamese, Austroasiatic
tr = trans("tr") # Turkish, Turkic

train_seed = set(random.sample(u, int(round(len(u)*80/100, 0)))); print("80% =", len(train_seed)) # 80% training --> 929 items
test_seed = u-train_seed; print("20% =", len(test_seed)) # 20% test --> 232 items
len(train_seed)+len(test_seed) 
train_seed&test_seed # empty intersection

#with open('train_seed', 'wb') as f:
#    pickle.dump(train_seed, f)
#with open('train_seed', 'rb') as handle:
#    train_seed = pickle.load(handle)
#with open('test_seed', 'wb') as f:
#    pickle.dump(test_seed, f)
#with open('test_seed', 'rb') as handle:
#    test_seed = pickle.load(handle)

multi_df = []
for vec, word in zip(vecs, names):
    if word in u:
        arabic = ar[word]
        hungarian = hu[word]
        indonesian = dd[word]
        vietnamese = vi[word]
        turkish = tr[word]
        multi_df.append([word, arabic, hungarian, indonesian, vietnamese, turkish, vec])
multi_df = pd.DataFrame(multi_df, columns=["word", "arabic", "hungarian", "indonesian", "vietnamese", "turkish", "vec"]) # 16820 rows

#with open('multi_df', 'wb') as f:
#    pickle.dump(multi_df, f)
with open('multi_df', 'rb') as handle:
    multi_df = pickle.load(handle)

train =  multi_df[multi_df["word"].isin(train_seed)]; train # 4881 items
test = multi_df[multi_df["word"].isin(test_seed)]; test # 1233 items

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

#with open('train_dict', 'wb') as f:
#    pickle.dump(train_dict, f)
#with open('train_dict', 'rb') as handle:
#    train_dict = pickle.load(handle)
    
ar_test = phon_vectorizer(test["arabic"], "Arabic"); print("ar_test shape =", ar_test.shape)
hu_test = phon_vectorizer(test["hungarian"], "Hungarian"); print("hu_test shape =", hu_test.shape)
id_test = phon_vectorizer(test["indonesian"], "Indonesian"); print("id_test shape =", id_test.shape)
vi_test = phon_vectorizer(test["vietnamese"], "Vietnamese"); print("vi_test shape =", vi_test.shape)
tr_test = phon_vectorizer(test["turkish"], "Turkish"); print("tr_test shape =", tr_test.shape)
en_test = phon_vectorizer(test["word"], "English"); print("en_test shape =", en_test.shape)

test_dict = {"arabic" : ar_test, "hungarian" : hu_test, "indonesian" : id_test, "vietnamese" : vi_test, "turkish" : tr_test, "english" : en_test}

#with open('test_dict', 'wb') as f:
#   pickle.dump(test_dict, f)
with open('test_dict', 'rb') as handle:
    test_dict = pickle.load(handle)

####################################################
# Dealing with Class Imbalance using Class Weights #
####################################################

y = np.stack(multi_df["vec"], axis=0)
y.shape

y_integers = np.argmax(y, axis=1)
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_integers), y=y_integers)
d_class_weights = dict(enumerate(class_weights))

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
    input_train = np.concatenate((input_train)); print("input_train shape (24405, 15, 22) =", input_train.shape) 
    target_train = np.concatenate((target_train)); print("target_train shape (24405, 11) =", target_train.shape)
    input_test = test_dict[language]; print("input_test shape (1233, 15, 22) =", input_test.shape)
    target_test = np.array([vec for vec in test["vec"]]); print("target_test shape (1233, 11) =", target_test.shape)
    # LSTM - multilingual
    print("Starting to train LSTM - MULTILINGUAL CONDITION")
    model=keras.models.Sequential()
    model.add(keras.layers.Masking(mask_value=0., input_shape=(15, 22)))
    model.add(keras.layers.LSTM(units=25, batch_input_shape=(None, 15, 22), dropout=0.2, recurrent_dropout=0.2, return_sequences=False)) # 100
    model.add(keras.layers.Dense(11, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    model.fit(input_train, target_train, epochs=1, validation_data=(input_test, target_test), class_weight=d_class_weights, shuffle=True)
    #model.save("model_25_"+language+".h5")
    results = model.evaluate(input_test, target_test); print("RESULTS (multilingual) =", results)
    prediction = model.predict(input_test)
    #with open('prediction_25_'+language, 'wb') as f:
    #    pickle.dump(prediction, f)
    return results
        
def network_random(language): 
    ls = [w for w in ["arabic", "hungarian", "indonesian", "vietnamese", "turkish", "english"] if w != language]
    input_train = []
    target_train = []
    for l in ls: 
        input_train.append(train_dict[l])
        target_train.append(np.array([vec for vec in train["vec"]]))
    input_train = np.concatenate((input_train)); print("input_train shape (24405, 15, 22) =", input_train.shape) 
    target_train = np.concatenate((target_train)); print("target_train shape (24405, 11) =", target_train.shape)
    input_test = test_dict[language]; print("input_test shape (1233, 15, 22) =", input_test.shape)
    target_test = np.array([vec for vec in test["vec"]]); print("target_test shape (1233, 11) =", target_test.shape)
    print("Starting to train LSTM - RANDOM CONDITION")
    np.random.seed(0)
    np.random.shuffle(target_train)
    s_model=keras.models.Sequential()
    s_model.add(keras.layers.Masking(mask_value=0., input_shape=(15, 22)))
    s_model.add(keras.layers.LSTM(units=25, batch_input_shape=(None, 15, 22), dropout=0.2, recurrent_dropout=0.2, return_sequences=False))
    s_model.add(keras.layers.Dense(11, activation='softmax'))
    s_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    s_model.summary()
    s_model.fit(input_train, target_train, epochs=1, validation_data=(input_test, target_test), class_weight=d_class_weights, shuffle=True)
    #s_model.save("s_model_25_"+language+".h5")
    s_results = s_model.evaluate(input_test, target_test); print("RESULTS (random) =", s_results)
    s_prediction = s_model.predict(input_test)
    #with open('s_prediction_25_'+language, 'wb') as f:
    #    pickle.dump(s_prediction, f)
    return s_results

res = []
for l in ["arabic", "hungarian", "indonesian", "vietnamese", "turkish", "english"]:
    print("\n\n")
    print(l.upper(), "\n")
    results = network_multilingual(l)
    s_results = network_random(l)
    res.append([l, round(results[0], 4), round(results[1], 4), round(s_results[0], 4), round(s_results[1], 4)])

# No upsampling (overfitting) but class weighting; no downsampling (little data)

############################################################################################
############################################################################################
############################################################################################

with open('multi_df', 'rb') as handle:
    multi_df = pickle.load(handle)
with open('test_seed', 'rb') as handle:
    test_seed = pickle.load(handle)

test = multi_df[multi_df["word"].isin(test_seed)]; test # 1233 items

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

#cross_entropy = tf.keras.losses.CategoricalCrossentropy()
d_acc = {"arabic":[], "hungarian":[], "indonesian":[], "vietnamese":[], "turkish":[], "english":[]}
#d_loss = {"arabic":[], "hungarian":[], "indonesian":[], "vietnamese":[], "turkish":[], "english":[]}
for l in ["arabic", "hungarian", "indonesian", "vietnamese", "turkish", "english"]:
    col = test["pred_"+l]
    for i, value in enumerate(col):
        v = np.zeros(11)
        v[np.argmax(value)] = 1
        #d_loss[l].append(cross_entropy(value, test.iloc[i]["vec"]).numpy())
        x = v == test.iloc[i]["vec"]
        if x.all() == True:
            d_acc[l].append(1)
        else:
            d_acc[l].append(0)
            
d_acc_r = {"arabic":[], "hungarian":[], "indonesian":[], "vietnamese":[], "turkish":[], "english":[]}
#d_loss_r = {"arabic":[], "hungarian":[], "indonesian":[], "vietnamese":[], "turkish":[], "english":[]}
for l in ["arabic", "hungarian", "indonesian", "vietnamese", "turkish", "english"]:
    col = test["s_pred_"+l]
    for i, value in enumerate(col):
        v = np.zeros(11)
        v[np.argmax(value)] = 1
        #d_loss[l].append(cross_entropy(value, test.iloc[i]["vec"]).numpy())
        x = v == test.iloc[i]["vec"]
        if x.all() == True:
            d_acc_r[l].append(1)
        else:
            d_acc_r[l].append(0)


from statsmodels.stats.contingency_tables import mcnemar

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
    if min([a, b, c, d]) < 25: 
        result = mcnemar(table, exact=True)
        print("EXACT TEST (< 25)\n")
    else:
        result = mcnemar(table, exact=False, correction=True)
        print("Standard calculation (> 25)\n")
    print('statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))
    
for l in ["arabic", "hungarian", "indonesian", "vietnamese", "turkish", "english"]:
    print(l.upper())
    McNemar(l)
