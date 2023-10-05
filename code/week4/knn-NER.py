import nltk
import sys
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold

LEX={}
def lookup(x):
   global LEX
   if x not in LEX:
      LEX[x]=len(LEX)
   return LEX[x]

def read_conll():
   fp=open("ned.train","r",encoding = "ISO-8859-1")
   data=[]
   tags=[]
   for line in fp:
    line=line.rstrip()
    m=re.match("^([^\s]+)\s+([^\s]+)\s+([^\s]+)",line)
    if m:
       tags.append(m.group(2)+"_"+m.group(3)) # PoS_NER
    else:
        if len(tags)!=0:
            data.append(tags)
        tags=[]

   n=5 # window size
   focus=3 # 2 left, focus word, 2 right

   # TWEAK: Always choose even number left, focus, even right: like n=9,focus=5 (4 left, 4 right)
   
   X=[]
   y=[]
   for sentence in data:
       grams = [sentence[i: i + n] for i in range(len(sentence) - n + 1)]
       for gram in grams:
          m=re.match("^.+_([^$^_]+)$",gram[focus])
          if m:
             y.append(m.group(1))
             focus_word=re.sub("_[^$^_]+$","",gram[focus-1])
             gram[focus-1]=focus_word
             vct=[lookup(x) for x in gram]
             X.append(vct)
          else:
             continue
   return np.array(X),np.array(y)

def cv_classification_report(y_true, y_pred):
   print(classification_report(y_true, y_pred)) # print classification report

def main():
    X,y=read_conll()
    skf=StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
       print("Fold ",i)
       X_train=X[train_index]
       y_train=y[train_index]
       X_test=X[test_index]
       y_test=y[test_index]
       clf = KNeighborsClassifier(n_neighbors = 5) # TWEAK: vary number of neighbors
       y_pred=clf.fit(X_train,y_train).predict(X_test)
       report = classification_report(y_test, y_pred)
       print(report)
    
    
if __name__=="__main__":
    main()
