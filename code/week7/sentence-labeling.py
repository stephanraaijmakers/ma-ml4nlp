from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, LSTM, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import sys
import itertools
from sklearn.utils import class_weight
import re
import Levenshtein as lev
from tqdm import tqdm


from transformers import AutoModelWithLMHead, AutoTokenizer

#https://huggingface.co/mrm8488/t5-base-finetuned-e2m-intent
tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-e2m-intent")
intent_model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-e2m-intent")


def featurize_data(sentences): 
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    print("Featurizing data...be patient")

    X=[]
    Words={}
    for sentence in tqdm(sentences):
        emb=model.encode([sentence])[0]
        X.append(list(emb))
        for word in sentence.split(" "):
            Words[word]=1
    return np.array(X), len(Words)


def create_keras_model(X, y, vocab_len, max_len):
        nb_dims=X.shape[1]
        nb_classes = 1

        model = Sequential()
        model.add(Dense(128,input_shape=(nb_dims,1)))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
       
        model.add(Dense(32))
        model.add(Activation('relu'))
        model.add(Dropout(0.1))

        model.add(Flatten())    
        model.add(Dense(nb_classes))
        model.add(Activation('sigmoid'))

        model.summary()
        opt=tf.keras.optimizers.Adam()
        #opt = tf.keras.optimizers.SGD(learning_rate=0.001)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        
        model.summary()
        print("Training...")
    
        model.fit(X, y, epochs=100, batch_size=16, validation_split=0.1, verbose=2)
        return model



def read_sentence_data(fn): # <dialogue id><TAB>sentence<TAB>label:0/1
    sentences=[]
    labels=[]
    fp=open(fn,"r")
    for line in fp:
        fields=line.rstrip().split("\t")
        if fields:
            labels.append(float(fields[1]))
            sentences.append(fields[0])
    fp.close()
    return sentences, labels    

            

def main(fn):
    sentences, y=read_sentence_data(fn)
    X, vocab_len=featurize_data(sentences)  
    max_len=len(X[0])
    X_train, X_test, y_train, y_test=train_test_split(X, np.array(y), test_size=0.2) # random_state=42)  
    model=create_keras_model(X_train,y_train, vocab_len, max_len)
    res=model.evaluate(X_test, y_test)
    print("Loss:",res[0]," Accuracy:",res[1])
    pred=model.predict(X_test) 
    pred=[p[0] for p in pred]
    pred=list(np.vectorize(lambda x: int(x >= 0.5))(pred))
    print("PRED:",pred)
    print("GT:",y_test)
    print("F1:",f1_score(y_test, pred))
    


if __name__=="__main__":
    main(sys.argv[1])
