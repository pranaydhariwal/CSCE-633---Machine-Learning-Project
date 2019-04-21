import pandas as pd
import numpy as np
from string import punctuation
import pickle

max_length = 150
min_length = 10


def encode(train,map):
    x = list()
    for i in train:
        val =  [map[j] for j in i.split()]
        x.append(val)
    return np.asarray(x)

train = pd.read_excel("./SST2/train.xlsx")
test = pd.read_excel("./SST2/test.xlsx")
val = pd.read_excel("./SST2/dev.xlsx")

train['sentence'] = train['sentence'].astype('str')
test['sentence'] = test['sentence'].astype('str')
val['sentence'] = val['sentence'].astype('str')

# convert to lower case
train['sentence'] = train['sentence'].str.lower()
test['sentence'] = test['sentence'].str.lower()
val['sentence'] = val['sentence'].str.lower()

# remove punctuation
train['sentence'] = train['sentence'].str.replace('[{}]'.format(punctuation),'')
test['sentence'] = test['sentence'].str.replace('[{}]'.format(punctuation),'')
val['sentence'] = val['sentence'].str.replace('[{}]'.format(punctuation),'')

# # remove stop words
# from nltk.corpus import stopwords
# stop = stopwords.words('english')
# train['sentence'] = train['sentence'].apply(lambda x: ' '.join([word for word in str(x).split() if word not in (stop) and len(word) > 1]))
# test['sentence'] = test['sentence'].apply(lambda x: ' '.join([word for word in str(x).split() if word not in (stop) and len(word) > 1]))
# val['sentence'] = val['sentence'].apply(lambda x: ' '.join([word for word in str(x).split() if word not in (stop) and len(word) > 1]))

train = train.query("sentence.str.len() < {} and sentence.str.len() > {}".format(max_length,min_length))
val = val.query("sentence.str.len() < {} and sentence.str.len() > {}".format(max_length,min_length))
test = test.query("sentence.str.len() < {} and sentence.str.len() > {}".format(max_length,min_length))

# Split it into train
train_y = train['label'].values
val_y = val['label'].values
test_y = test['label'].values

all_words = list()
for i in train['sentence']:
    all_words.extend(i.split())
for i in test['sentence']:
    all_words.extend(i.split())
for i in val['sentence']:
    all_words.extend(i.split())
word2id = {w:i+1 for i,w in enumerate(all_words)}

train_x = encode(train['sentence'],word2id)
test_x = encode(test['sentence'],word2id)
val_x = encode(val['sentence'],word2id)

from keras.preprocessing import sequence
X_train = sequence.pad_sequences(train_x, maxlen=max_length)
X_test = sequence.pad_sequences(test_x, maxlen=max_length)
X_val = sequence.pad_sequences(val_x, maxlen=max_length)

pickle.dump(len(all_words),open("./SST2/len","wb"))
pickle.dump(X_train,open("./SST2/train_x","wb"))
pickle.dump(X_test,open("./SST2/test_x","wb"))
pickle.dump(X_val,open("./SST2/val_x","wb"))
pickle.dump(train_y,open("./SST2/train_y","wb"))
pickle.dump(test_y,open("./SST2/test_y","wb"))
pickle.dump(val_y,open("./SST2/val_y","wb"))
