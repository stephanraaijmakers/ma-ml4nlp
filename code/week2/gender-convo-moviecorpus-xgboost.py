import sys
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics

import convokit
from convokit import Corpus, download, TextCleaner, FightingWords

from numpy import mean
from numpy import std
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold

import os

# Stephan Raaijmakers LUCL - July 2022

def process_data():
#    movie_corpus = Corpus(download('movie-corpus')) # do this once
    print("Loading data...")

    home_dir=os.getenv("HOME")
    movie_corpus=Corpus(filename=home_dir+"/.convokit/downloads/movie-corpus")
    print("Done.")
    
    male=[]
    female=[]

    print("Processing data")

    
    for utt in movie_corpus.iter_utterances():
        text=utt.text
        speaker=utt.speaker
        gender=utt.speaker.meta['gender']
        if gender=='f':
            female.append(text)
        elif gender=='m':
            male.append(text)
    print("Done; %d samples"%(len(male)+len(female)))
    return male[:5000],female[:5000]


if __name__=="__main__":
    male,female=process_data()
    vectorizer=TfidfVectorizer(stop_words='english', max_features=500)
    vectorizer.fit(male+female)

    X=[]
    y=[]
    text=[]

    labeled_utt=[(m,0.0) for m in male]+[(f,1.0) for f in female]
    labels=[l for (_utt,l) in labeled_utt]

    labeled_utt_train, labeled_utt_test, labels_train, labels_test = train_test_split(labeled_utt, labels, test_size=0.1)

    X_train=[]
    X_test=[]
    y_train=[]
    y_test=[]
    text_test=[]
    
    for (utt, label) in labeled_utt_train:
        values=list(vectorizer.transform([utt]).toarray()[0])
        X_train.append(values)
        y_train.append(label)

    for (utt, label) in labeled_utt_test:
        values=list(vectorizer.transform([utt]).toarray()[0])
        X_test.append(values)
        y_test.append(label)
        text_test.append(utt)

        # xgboost for classification
    model = XGBClassifier()
    #cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    #n_scores = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    #print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
    # fit the model on the whole dataset
    model = XGBClassifier()
    model.fit(X_train, y_train)

        
    y_pred=model.predict(X_test)
    n=0
    for pred in y_pred:
        print(text_test[n],y_test[n], pred)
        n+=1

    print("Precision:",metrics.precision_score(y_test, y_pred))
    print("Recall:",metrics.recall_score(y_test, y_pred))
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

    exit(0)    
