from pyexpat import model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding, Input
from keras.layers import LSTM, Bidirectional
from keras.layers import Layer
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.preprocessing.text import one_hot
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import f1_score
import keras.backend as K
import itertools
import tensorflow as tf
import numpy as np
import sys
import os
import re
import matplotlib.pyplot as plt
import shutil

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')


from numpy.random import seed
seed(42)
tf.random.set_seed(42)

from keract import get_activations
from tensorflow.python.keras.utils.vis_utils import plot_model

os.environ['KERAS_ATTENTION_DEBUG'] = '1'
from attention import Attention
from pathlib import Path


class MyAttention(Layer):
    def __init__(self,**kwargs):
        super(MyAttention,self).__init__(**kwargs)
 
    def build(self,input_shape):
        self.W=self.add_weight(name='my_attention_weight', shape=(input_shape[-1],1), 
                               initializer='random_normal', trainable=True)
        self.b=self.add_weight(name='attention_bias', shape=(input_shape[1],1), 
                               initializer='zeros', trainable=True)        
        super(MyAttention, self).build(input_shape)
 
    def call(self,x):
        # Alignment scores. Pass them through tanh function
        e = K.tanh(K.dot(x,self.W)+self.b)
        # Remove dimension of size 1
        e = K.squeeze(e, axis=-1)   
        # Compute the weights
        alpha = K.softmax(e)
        # Reshape to tensorFlow format
        alpha = K.expand_dims(alpha, axis=-1)
        # Compute the context vector
        context = x * alpha
        context = K.sum(context, axis=1)
        return context



# Using words, no fancy embeddings
def process_data_words(fn, windowSize, focusPosition):
    fp=open(fn,"r")
    max_len=0
    X=[]
    y=[]
    D={}
    for line in fp:
        try:
            fields=line.rstrip().split("\t")
        except:
            continue
        if "session.ID" in line: # skip header
            continue
        dialogue_id=fields[0]
        utterance=fields[4]
        label=fields[-1]
        m=re.match("^([^=;]+)\=.+", label)
        if m:
            label=m.group(1)+")"
        if dialogue_id in D:
            D[dialogue_id]+=[(utterance, label)]
        else:
            D[dialogue_id]=[(utterance,label)]

    # Windowing
    Words={}
    Texts=[]
    max_len=0
    for dialogue_id in D:
        if int(dialogue_id)>300: # tiny dataset: just 300 #EXPERIMENT: use higher number (more training data)
            break
        print("Processing dialogue",dialogue_id)
        utterances=D[dialogue_id]
        ngrams = [utterances[i: i + windowSize] for i in range(len(utterances) - windowSize + 1)]
        for ngram in ngrams:
            label=ngram[focusPosition-1][1]
            utterances=[x[0] for x in ngram]
            x=[]
            for u in utterances:   
                for w in u.split(" "):
                    Words[w]=1
                    x.append(w)
            if len(x)>max_len:
                max_len=len(x)
            Texts.append(' '.join(x))
            y.append(label)
    nb_classes=len(set(y))
    X=[one_hot(text, len(Words)) for text in Texts] 
    X=pad_sequences(X,maxlen=max_len,padding='post')
    lb = preprocessing.LabelBinarizer()
    y=lb.fit_transform(y)    
    run_model_words_lstm(X,y,nb_classes)


def run_model_words_lstm(X,y,nb_classes): # EXPERIMENT: see options for run_model_sentence_transformers_lstm() below
    X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.1, random_state=42)
    X_train = np.reshape(X_train, (X_train.shape[0],1,X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0],1,X_test.shape[1]))

    model = Sequential()
    model.add(LSTM(128, input_shape=(1,max_len), activation="relu",return_sequences=True))  
    model.add(MyAttention())
    model.add(Dense(64, activation="relu"))
    model.add(Dense(nb_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    output_dir = Path('intent_attention')
    if output_dir.exists():
        shutil.rmtree(str(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    class VisualiseAttentionMap(Callback):
        def on_epoch_end(self, epoch, logs=None):

            names = [weight.name for layer in model.layers for weight in layer.weights]
            weights = model.get_weights()

            for name, weight in zip(names, weights):
                if name=="my_attention/my_attention_weight:0":
                    attention_map=weight.transpose()
            
            plt.imshow(attention_map,cmap='hot')
            iteration_no = str(epoch).zfill(3)
            plt.axis('off')
            plt.title(f'Iteration {iteration_no}')
            output_filename = f'{output_dir}/epoch_{iteration_no}.png'
            print(f' Saving to {output_filename}.')
            plt.savefig(output_filename)
            plt.close()
    
    model.fit(X_train, y_train, epochs=300, batch_size=16, validation_split=0.1,callbacks=[VisualiseAttentionMap()])

    score = model.evaluate(X_test, y_test, batch_size=8)
    print("Loss=",score[0], " Accuracy=",score[1])
    pred = np.argmax(model.predict(X_test),axis=1)
    gt=np.argmax(y_test,axis=1)
    print("F1=",f1_score(gt, pred, average='micro'))


# Using Sentence Transformers
def process_data_sentence_transformers(fn, windowSize, focusPosition):
    st_model = SentenceTransformer('all-MiniLM-L6-v2')
    fp=open(fn,"r")
    max_len=0
    X=[]
    y=[]
    D={}
    for line in fp:
        try:
            fields=line.rstrip().split("\t")
        except:
            continue
        if "session.ID" in line: # skip header
            continue
        dialogue_id=fields[0]
        utterance=fields[4]
        label=fields[-1]
        m=re.match("^([^=;]+)\=.+", label)
        if m:
            label=m.group(1)+")"
        if dialogue_id in D:
            D[dialogue_id]+=[(utterance, label)]
        else:
            D[dialogue_id]=[(utterance,label)]

    # Windowing
    
    for dialogue_id in D:
        if int(dialogue_id)>1000: # tiny dataset: just 100
            break
        print("Processing dialogue",dialogue_id)
        utterances=D[dialogue_id]
        ngrams = [utterances[i: i + windowSize] for i in range(len(utterances) - windowSize + 1)]
        for ngram in ngrams:
            label=ngram[focusPosition-1][1]
            utterances=[x[0] for x in ngram]
            encodings=st_model.encode(utterances)
            x=[]
            for encoding in encodings:
                x+=list(encoding)
            X.append(x)
            y.append(label)

    nb_classes=len(set(y))
    lb = preprocessing.LabelBinarizer()
    y=lb.fit_transform(y)    

    run_model_sentence_transformers_lstm(X, y, nb_classes, windowSize)


def run_model_sentence_transformers_lstm(X,y, nb_classes, windowSize):
    X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.1, random_state=42)
    X_train=np.array(X_train)
    X_test=np.array(X_test)
    X_train = np.reshape(X_train, (X_train.shape[0],1,X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0],1,X_test.shape[1]))

    model = Sequential()
    model.add(LSTM(128, input_shape=(1,384*windowSize), activation="relu",return_sequences=True)) # 384=size SentenceTransformer embedding # EXPERIMENT: change 128 to ...
    # EXPERIMENT: model.add(Bidirectional(LSTM(128, nput_shape=(1,384*windowSize), activation="relu",return_sequences=True)))
    model.add(MyAttention())
    model.add(Dense(64, activation="relu")) # EXPERIMENT: add a few layewrs (just copy). Use different values of output dimension (32, 16,...)
    model.add(Dense(nb_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    output_dir = Path('intent_attention')
    if output_dir.exists():
        shutil.rmtree(str(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    class VisualiseAttentionMap(Callback):
        def on_epoch_end(self, epoch, logs=None):

            names = [weight.name for layer in model.layers for weight in layer.weights]
            weights = model.get_weights()

            for name, weight in zip(names, weights):
                if name=="my_attention/my_attention_weight:0":
                    attention_map=weight.transpose()
            
            plt.imshow(attention_map,cmap='hot')
            iteration_no = str(epoch).zfill(3)
            plt.axis('off')
            plt.title(f'Iteration {iteration_no}')
            output_filename = f'{output_dir}/epoch_{iteration_no}.png'
            print(f' Saving to {output_filename}.')
            plt.savefig(output_filename)
            plt.close()
    
    model.fit(X_train, y_train, epochs=100, batch_size=8, validation_split=0.1,callbacks=[VisualiseAttentionMap()])

    score = model.evaluate(X_test, y_test)
    print("Loss=",score[0], " Accuracy=",score[1])
    pred = np.argmax(model.predict(X_test),axis=1)
    gt=np.argmax(y_test,axis=1)
    print("F1=",f1_score(gt, pred, average='micro'))

    



def main(fn, windowSize, focusPosition):
    #process_data_words(fn, windowSize, focusPosition)  # EXPERIMENT: use simple word embeddings (integers)
    process_data_sentence_transformers(fn, windowSize, focusPosition) # EXPERIMENT: use BERT/Sentence Transformer embeddings
    
    

if __name__=="__main__":
    # TSV file, window size, focus position
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3])) # EXPERIMENT: use different window sizes and focus positions. 
