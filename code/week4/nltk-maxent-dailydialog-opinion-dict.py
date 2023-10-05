import nltk
import sys
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
import convokit
from convokit import Corpus, download, TextCleaner, FightingWords
from tqdm import tqdm

# Update 05.10.2023

# Retrieve n most salient words with MNB, per pos/neg
# Re-use those with maxent
# Does the resulting maxent classifier outperform MNB?


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



def detect_salient_words(utt, neg_salient_words, pos_salient_words):
    d=dict()
    utt_w=utt.split(" ")
    for word in neg_salient_words:
        if word in utt_w:
            d[word]=1
        #else:
        #    d[word]=0
    for word in pos_salient_words:
        if word in utt_w:
            d[word]=1
        #else:
        #    d[word]=0
    return d
     

def run_maxent(train,test):
    print("Training MaxEnt classifier on %d cases"%(len(train)))
    classifier = nltk.classify.MaxentClassifier.train(train, trace=3, max_iter=1000)
    print("Done.")
    
    mfeat=open("maxent-predictions-sentiment.txt","w")
    for (featureset, label) in test: # don't use the label, keep it for printing 
        pdist = classifier.prob_classify(featureset)
        print("GT=%s %s positive=%f negative=%f" %(label, featureset, pdist.prob('positive'),pdist.prob('negative')), file=mfeat)
    mfeat.close()
    print("See maxent-predictions-sentiment.txt")
    print("Top most important features:")
    print("------------------------------------")
    classifier.show_most_informative_features()


def read_pos_neg():
    with open('positive-words.txt',"r") as fp:
        pos=[line.rstrip() for line in fp.readlines()]
    fp.close()
    with open('negative-words.txt',"r") as fp:
        neg=[line.rstrip() for line in fp.readlines()]
    fp.close()
    
    return pos, neg
        

def main():
    pos,neg=process_data()
#    vectorizer=TfidfVectorizer(stop_words='english', max_features=5000)
#    vectorizer.fit(pos+neg)

    X=[]
    y=[]
    text=[]

    labeled_utt=[(n,0.0) for n in neg]+[(p,1.0) for p in pos]
    labels=[l for (_utt,l) in labeled_utt]


    labeled_utt_train, labeled_utt_test, labels_train, labels_test = train_test_split(labeled_utt, labels, test_size=0.1)


    pos_salient, neg_salient=read_pos_neg()


    X_train=[]
    y_train=[]
    X_test=[]
    y_test=[]
    text_test=[]
    
    print("Preparing maxent data...")
    for (utt, label) in labeled_utt_train:
        if label==0:
            label="negative"
        else:
            label="positive"
        d=detect_salient_words(utt, neg_salient, pos_salient)
        X_train.append((d, label)) # we don't use the label of course
        y_train.append(label)

    for (utt, label) in labeled_utt_test:
        if label==0:
            label="negative"
        else:
            label="positive"
        d=detect_salient_words(utt, neg_salient, pos_salient)
        X_test.append((d,label)) # we don't use the label of course
        y_test.append(label)
        text_test.append(utt)
    print("Done.")

    run_maxent(X_train[:200],X_test[:200])

    
if __name__=="__main__":
    main()
