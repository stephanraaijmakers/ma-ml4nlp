import sys
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics

import convokit
from convokit import Corpus, download, TextCleaner, FightingWords


def process_data():
#    simpsons_corpus = Corpus(download('simpsons-corpus'))
    print("Loading data...")

    simpsons_corpus=Corpus(filename='/content/simpsons-corpus')
    print("Data loaded.")
    
    male=[]
    female=[]

    print("Processing data")

    
    for utt in simpsons_corpus.iter_utterances():
        text=utt.text
        speaker=utt.speaker
        gender=utt.speaker.meta['gender']
        if gender=='f':
            female.append(text)
        elif gender=='m':
            male.append(text)
    print("Done.")
    print(len(male),len(female))
    return male[:1000], female[:1000]


if __name__=="__main__":
    male,female=process_data()
    vectorizer=TfidfVectorizer(stop_words='english', max_features=5000)
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


    clf = svm.SVC(kernel = "rbf")
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
