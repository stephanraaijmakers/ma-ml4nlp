from gensim import utils
import gensim.models
import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import Perceptron
import numpy as np
import gensim.downloader
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Conv1D, MaxPooling1D, Flatten, Embedding
from tensorflow.keras.preprocessing.text import one_hot,Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer




class MyCorpus:
        """An iterator that yields sentences (lists of str)."""
        def __iter__(self):
            corpus_path = './Small_talk_Intent.lines'
            for line in open(corpus_path,"rb"):
                # assume there's one document per line, tokens separated by whitespace
                yield utils.simple_preprocess(line)

# Build your own W2V model                
def build_w2v_model():
    sentences = MyCorpus()
    return gensim.models.Word2Vec(sentences=sentences)



# Turn a text into a concatenation of W2V vectors.
def word2vec_transformer(texts, labels, w2v_model, dimension=100):
        vectors=[]
        y=[]
        text_l=[]
        for (text,label) in zip(texts,labels):
           if label not in ['smalltalk_agent_beautiful','smalltalk_agent_busy','smalltalk_agent_fired','smalltalk_agent_good','smalltalk_agent_my_friend','smalltalk_agent_talk_to_me','smalltalk_dialog_sorry']:
              continue
           words=text.split(" ")
           vector=[]
           if len(words)==5:
                for word in words:
                    if word in w2v_model.wv:
                        vector.extend(np.array(w2v_model.wv[word]))
                    else:
                        vector.extend(np.zeros(dimension))
                vectors.append(vector)
                y.append(label)
                text_l.append(text)
        return vectors, y, set(y), text_l


def build_keras_model(input_dim, nb_classes):
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim))
    model.add(Activation('relu'))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.summary()    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[tf.keras.metrics.CategoricalAccuracy()])    
    return model
    

    
df=pd.read_csv("smalltalk_intent.csv",encoding= 'unicode_escape') # try encoding = 'utf-8'; encoding = 'ISO-8859-1' for unicode errors
X=list(df['utterance'])
y=list(df['intent'])

# Split data into training, test, and training/test labels. Test=10% of all data.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,random_state = 3, stratify=y)
X_test_texts=X_test
X_test_labels=y_test
    
# Word2vec
w2v_model=build_w2v_model()
dimension=100

print(y_train)
X_train, y_train,y_train_set, _X_train_text=word2vec_transformer(X_train,y_train, w2v_model,dimension)
X_test, y_test,y_test_set, X_test_texts,=word2vec_transformer(X_test,y_test,w2v_model,dimension)

nb_classes=len(set(y_train_set.union(y_test_set)))
model=build_keras_model(dimension*5, nb_classes)
X_train = np.asarray(X_train)
X_test = np.asarray(X_test)

label_encoder = LabelBinarizer()
print(y_train)
y_train = label_encoder.fit_transform(y_train)
y_test_orig=y_test
y_test=label_encoder.transform(y_test)

model.fit(X_train, y_train, epochs=200, validation_split=0.1, verbose=1)
results = model.evaluate(X_test, y_test, batch_size=64)
print("Loss, accuracy:", results)
print("Predicting...")
preds = model.predict(X_test, verbose=0)

n=0
for p in preds:
    x=np.zeros(nb_classes,dtype=int)
    max_pos=np.argmax(p,axis=0)
    x[max_pos]=1
    print(X_test_texts[n], y_test_orig[n],label_encoder.inverse_transform(np.array([x])))
    n+=1
     

             
