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


# Retrieve n most salient words with MNB, per pos/neg
# Re-use those with maxent
# Does the resulting maxent classifier outperform MNB?


def process_data():
#    dailydialog_corpus = Corpus(download('dailydialog-corpus'))
    print("Loading data...")

    dailydialog_corpus=Corpus(filename='/home/stephan/.convokit/saved-corpora/dailydialog-corpus')
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





def run_maxent(train,test):
    print("Training MaxEnt classifier on %d cases"%(len(train)))
    classifier = nltk.classify.MaxentClassifier.train(train, trace=3, max_iter=100)
    print("Done.")

    classes=["B-LOC","B-MISC","B-ORG","B-PER","I-LOC","I-MISC","I-ORG","I-PER","O"]
    
    mfeat=open("maxent-feature-weights-ner.txt","w")
    for (featureset, label) in test:
        pdist = classifier.prob_classify(featureset)
        focusword="???"
        for key in featureset:
            m=re.match("curr_w_(.+)", key)
            if m:
                focusword=m.group(1)
        print("Focus word: %s GT label=%s"%(focusword, label), end=" ", file=mfeat)
        for c in classes:
            print("%s:%f"%(c,pdist.prob(c)), end=" ", file=mfeat)
        print(file=mfeat)
    mfeat.close()
    print("See maxent-feature-weights-ner.txt")
    print("Top most important features:")
    print("------------------------------------")
    classifier.show_most_informative_features(1000)


def read_conll():
   fp=open("ned.train","r",encoding = "ISO-8859-1")
   data=[]
   tags=[]
   for line in fp:
    line=line.rstrip()
    m=re.match("^([^\s]+)\s+([^\s]+)\s+([^\s]+)",line)
    if m:
        tags.append((m.group(1), m.group(2), m.group(3)))
    else:
        if len(tags)!=0:
            data.append(tags)
        tags=[]
   return data

def main():
    
    labeled_utt=read_conll()

    X_train=[]
    y_train=[]
    X_test=[]
    y_test=[]
    text_test=[]

    train_len=int(.66*len(labeled_utt))
    labeled_utt_train=labeled_utt[:train_len]
    labeled_utt_test=labeled_utt[train_len:]
    
    print("Preparing maxent data...")
    for utt in labeled_utt_train: #[(word, pods, NER),...]
        bigrams = [utt[i:i+2] for i in range(len(utt)-2+1)]
        for bigram in bigrams:
            d=dict()
            left_w="left_w_%s"%(bigram[0][0])
            left_pos="left_pos_%s"%(bigram[0][1])
            left_ner="left_ner_%s"%(bigram[0][2])
            d[left_w]=1
            d[left_pos]=1
            d[left_ner]=1
            curr_w="curr_w_%s"%(bigram[1][0])
            curr_pos="curr_pos_%s"%(bigram[1][1])
            d[curr_w]=1
            d[curr_pos]=1
            if bigram[1][0][0].isupper():
                d["curr_upper"]=1
            label=bigram[1][2]
            X_train.append((d, label))
            y_train.append(label)

    for utt in labeled_utt_test: #[(word, pods, NER),...]
        bigrams = [utt[i:i+2] for i in range(len(utt)-2+1)]
        for bigram in bigrams:
            d=dict()
            left_w="left_w_%s"%(bigram[0][0])
            left_pos="left_pos_%s"%(bigram[0][1])
            left_ner="left_ner_%s"%(bigram[0][2])
            d[left_w]=1
            d[left_pos]=1
            d[left_ner]=1
            curr_w="curr_w_%s"%(bigram[1][0])
            curr_pos="curr_pos_%s"%(bigram[1][1])
            d[curr_w]=1
            d[curr_pos]=1
            if bigram[1][0][0].isupper():
                d["curr_upper"]=1
            label=bigram[1][2]
            X_test.append((d, label))
            y_test.append(label)
            text_test.append(utt)

    print("Done.")

    run_maxent(X_train[:1000],X_test[:100])

    
if __name__=="__main__":
    main()
