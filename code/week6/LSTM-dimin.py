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
import matplotlib.pyplot as plt
import shutil

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


def process_data(fn):
    fp=open(fn,"r")
    max_len=0
    texts=[]
    y=[]
    Values={}
    for line in fp:
        try:
            values=line.rstrip().split(",")
        except:
            continue
        for v in values:
            Values[v]=1
        l=len(values)-1
        if l>max_len:
            max_len=l
        #texts.append(' '.join(list(reversed(values[:-1]))))
        texts.append(' '.join(values[:-1]))
        label=values[-1]
        y.append(label)

    vocab_size=len(Values)

    X=[one_hot(text, vocab_size) for text in texts] 
    
    nb_classes=len(set(y))
    lb = preprocessing.LabelBinarizer()
    y=lb.fit_transform(y)    

    X=pad_sequences(X,maxlen=max_len,padding='post')
    X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.1, random_state=42)
    
    X_train=np.array(X_train)
    y_train=np.array(y_train)
    X_test=np.array(X_test)
    y_test=np.array(y_test) #, max_len, vocab_size, nb_classes

    model = Sequential()
    #model.add(Embedding(vocab_size, 128, input_length=max_len))
    #model.add(Bidirectional(LSTM(128, activation="relu")))
    model.add(LSTM(12, input_shape=(12,1), activation="relu",return_sequences=True))
    model.add(MyAttention())
    model.add(Dense(64, activation="relu"))
    model.add(Dense(nb_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    


    output_dir = Path('dimin_attention')
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

    score = model.evaluate(X_test, y_test, batch_size=16)
    print("Loss=",score[0], " Accuracy=",score[1])
    pred = np.argmax(model.predict(X_test),axis=1)
    gt=np.argmax(y_test,axis=1)
    print("F1=",f1_score(gt, pred, average='micro'))

    



def main(fn):
    process_data(fn)
    

if __name__=="__main__":
    main(sys.argv[1])
