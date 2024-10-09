import sys
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
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


def get_salient_words(model, vectorizer, class_id):
    words = vectorizer.get_feature_names_out() # for scikit-learn > 0.24
    zipped = list(zip(words, model.feature_log_prob_[class_id]))
    sorted_zip = sorted(zipped, key=lambda t: t[1], reverse=True)
    return sorted_zip



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


    clf = MultinomialNB()
    clf.fit(X_train, y_train)

    #top_neg_words = clf.feature_log_prob_[0, :].argsort()[::-1]
    #top_pos_words = clf.feature_log_prob_[1, :].argsort()[::-1]
    #print(np.take(vectorizer.get_feature_names(), top_neg_words[:10]))
    #print(np.take(vectorizer.get_feature_names(), top_pos_words[:10]))

    fp_w=open("dailydialog-salient-words-100.txt","w")
    fp_w.write("Neg:"+str(get_salient_words(clf, vectorizer, 0)[:100])+"\n\n")
    fp_w.write("Pos:"+str(get_salient_words(clf, vectorizer, 1)[:100]))
    fp_w.close()
    print("See dailydialog-salient-words-100.txt for top-100 most salient words per class.")
             
    
    
    y_pred=clf.predict(X_test)
    n=0
    for pred in y_pred:
        print(text_test[n],y_test[n], pred)
        n+=1

    

    print("Precision:",metrics.precision_score(y_test, y_pred))
    print("Recall:",metrics.recall_score(y_test, y_pred))
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


    exit(0)    
