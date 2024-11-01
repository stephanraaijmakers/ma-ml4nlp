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
import matplotlib as mpl
import shutil
from numpy.random import seed
#seed(42)
#tf.random.set_seed(42)
from keract import get_activations
from tensorflow.python.keras.utils.vis_utils import plot_model

os.environ['KERAS_ATTENTION_DEBUG'] = '1'
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
        # Alignment scores. 
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
    Words={}
    n=0
    for line in fp:
        try:
            fields=line.rstrip().split("\t")
        except:
            continue
        if "session.ID" in line: # skip header
            continue
        utterance=fields[4]
        label=fields[-1]
        m=re.match("^([^\(]+)", label)
        if m:
            label=m.group(1)
        X.append(utterance)
        words=utterance.split(" ")
        l=len(words)
        if l>max_len:
            max_len=l
        for word in utterance.split(" "):
            Words[word]=1
        y.append(label)
        n+=1
        if n>300:
            break

    nb_classes=len(set(y))
    X=[one_hot(text, len(Words)) for text in X] 
    X=pad_sequences(X,maxlen=max_len,padding='post',value=-1.0)
    lb = preprocessing.LabelBinarizer()
    y=lb.fit_transform(y)    
    run_model_words_lstm(X,y,nb_classes, max_len)

def softmax(v):
  exponential = np.exp(v)
  probabilities = exponential / np.sum(exponential)
  return probabilities
 

def colorize(words, color_array):
    cmap=mpl.cm.Reds
    template = '<span class="barcode"; style="color: black; background-color: {}">{}</span>'
    colored_string = ''
    for word, color in zip(words, color_array):
        color = mpl.colors.rgb2hex(cmap(color)[:3])
        colored_string += template.format(color, '&nbsp' + word + '&nbsp')
    return colored_string


def run_model_words_lstm(X,y,nb_classes, max_len): # EXPERIMENT: see options for run_model_sentence_transformers_lstm() below
    X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.1, random_state=42)
    X_train=np.array(X_train)
    y_train=np.array(y_train)
    X_test=np.array(X_test)
    y_test=np.array(y_test)
    X_train = np.reshape(X_train, (X_train.shape[0],1,X_train.shape[1])) 
    X_test = np.reshape(X_test, (X_test.shape[0],1,X_test.shape[1]))
    # (n_samples, 12) => (n_samples, 12,1) = n_samples of (12 rows, 1 column)
    

    model = Sequential()
    model.add(LSTM(16, input_shape=(1,max_len), activation="relu",return_sequences=True)) # EXPERIMENT: 16, 32, 64, 128,...
    #model.add(Dense(32, activation="relu")) # EXPERIMENT - ADAPT "words" below
    model.add(MyAttention())
    model.add(Dense(8, activation="relu"))
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
            attention_map=None
            for name, weight in zip(names, weights):
                if name=="my_attention/my_attention_weight:0":
                    attention_map=weight.transpose()
            
            words=[str(x) for x in range(16)] # EQUAL to number of outputs of previous layer of attention layer!
            s = colorize(words, attention_map[0])
            iteration_no = str(epoch).zfill(3)
            out_fn = f'{output_dir}/epoch_{iteration_no}.html'
            with open(out_fn, 'w') as f:
                f.write(s)


            #attention_map[0]=softmax(attention_map[0])*100
            plt.imshow(attention_map,cmap='hot')
            iteration_no = str(epoch).zfill(3)
            plt.axis('off')
            plt.title(f'Iteration {iteration_no}')
            output_filename = f'{output_dir}/epoch_{iteration_no}.png'
            print(f' Saving to {output_filename}.')
            plt.savefig(output_filename)
            plt.close()
    
    model.fit(X_train, y_train, epochs=100, batch_size=8, validation_split=0.1,callbacks=[VisualiseAttentionMap()])

    score = model.evaluate(X_test, y_test, batch_size=8)
    print("Loss=",score[0], " Accuracy=",score[1])
    pred = np.argmax(model.predict(X_test),axis=1)
    gt=np.argmax(y_test,axis=1)
    print("F1=",f1_score(gt, pred, average='micro'))



def main(fn, windowSize, focusPosition):
    process_data_words(fn, windowSize, focusPosition)  # EXPERIMENT: use simple word embeddings (integers)
    
    

if __name__=="__main__":
    # TSV file, window size, focus position
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
