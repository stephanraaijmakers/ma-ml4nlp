import sys
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from xgboost import XGBClassifier

import convokit
from convokit import Corpus, download, TextCleaner, FightingWords


def process_data():
#    dailydialog_corpus = Corpus(download('dailydialog-corpus'))
    print("Loading data...")

    dailydialog_corpus=Corpus(filename='/content/dailydialog-corpus')
    print("Data loaded.")
    
    pos=[]
    neg=[]

    print("Processing data")

    
    for utt in dailydialog_corpus.iter_utterances():
        text=utt.text
        speaker=utt.speaker
        #gender=utt.speaker.meta['gender']
        sentiment=utt.meta['sentiment']
        if sentiment=="0":
            neg.append(text)
        elif sentiment=="1":
            pos.append(text)
    print("Done.")
    print("No. pos",len(pos),"No.neg:",len(neg))
    return pos[:1000], neg[:1000]


if __name__=="__main__":
    pos,neg=process_data()
    vectorizer=TfidfVectorizer(stop_words='english', max_features=5000)
    vectorizer.fit(pos+neg)

    X=[]
    y=[]
    text=[]

    labeled_utt=[(n,0.0) for n in neg]+[(p,1.0) for p in pos]
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


    clf = XGBClassifier()
    clf.fit(X_train, y_train)

    y_pred=clf.predict(X_test)
    n=0
    for pred in y_pred:
        print(text_test[n],y_test[n], pred)
        n+=1

    

    print("Precision:",metrics.precision_score(y_test, y_pred))
    print("Recall:",metrics.recall_score(y_test, y_pred))
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


    exit(0)    
