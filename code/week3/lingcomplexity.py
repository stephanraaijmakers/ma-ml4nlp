import glob
import pandas as pd
import seaborn as sns
import spacy
import textract  # To read .docx files
import TRUNAJOD.givenness
import TRUNAJOD.ttr
from TRUNAJOD import surface_proxies
from TRUNAJOD.syllabizer import Syllabizer

from collections import Counter
import itertools
import nltk
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import linalg
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
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
import nltk
from nltk.collocations import *

import matplotlib.pyplot as plt


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
    print("No. male:",len(male),"No.female:",len(female))
    stopwords_set = set(stopwords.words('english'))
    male_clean=[]
    female_clean=[]
    for text in male:
        text=' '.join([tok for tok in text.split(" ") if tok not in stopwords_set])
        male_clean.append(text)
        # Male: if "Holy-moly..." in text:
        #    print(text) # try "talk"
    for text in female:
        text=' '.join([tok for tok in text.split(" ") if tok not in stopwords_set])
        female_clean.append(text)

    #return male_clean[:1000], female_clean[:1000] # Check stopword filtering effect
    return male[:1000],female[:1000]



plt.rcParams["figure.figsize"] = (11, 4)
plt.rcParams["figure.dpi"] = 200




def measure_complexity(male, female):
    nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"])

    # Homework:
    # See https://trunajod20.readthedocs.io/en/latest/api_reference/ttr.html: here you will find the names of other metrics to be used, prefixed with TRUNAJOD.ttr. Strip that, and 
    # add to features {...} below. Example: TRUNAJOD.ttr.yule_k => "yule_k":[], (don't forget quotes and comma). Add TRUNAJOD.ttr.<name> to Male and Female as below. Plus add them to the plot (sns.boxplot), by analogy.
    features = {
        "lexical_diversity_mltd": [],
         "lexical_density": [],
        "pos_dissimilarity": [],
        "connection_words_ratio": [],
        "gender": [],
    }

    nsent = 10
    male_sentgrams = [' '.join(male[i:i+nsent]) for i in range(len(male)-nsent+1)]
    female_sentgrams = [' '.join(female[i:i+nsent]) for i in range(len(female)-nsent+1)]

    print("Male...")
    for text in tqdm(male_sentgrams):
        doc = nlp(text)
        features["lexical_diversity_mltd"].append(TRUNAJOD.ttr.lexical_diversity_mtld(doc))
        features["lexical_density"].append(surface_proxies.lexical_density(doc))
        features["pos_dissimilarity"].append(surface_proxies.pos_dissimilarity(doc))
        features["connection_words_ratio"].append(surface_proxies.connection_words_ratio(doc))
        features["gender"].append("male")

    print("Female...")
    for text in tqdm(female_sentgrams):
        doc = nlp(text)
        features["lexical_diversity_mltd"].append(TRUNAJOD.ttr.lexical_diversity_mtld(doc))
        features["lexical_density"].append(surface_proxies.lexical_density(doc))
        features["pos_dissimilarity"].append(surface_proxies.pos_dissimilarity(doc))
        features["connection_words_ratio"].append(surface_proxies.connection_words_ratio(doc))
        features["gender"].append("female")
    df = pd.DataFrame(features)
    _fig, axes = plt.subplots(2, 2)
    sns.boxplot(x="gender", y="lexical_diversity_mltd", data=df, ax=axes[0, 0])
    sns.boxplot(x="gender", y="lexical_density", data=df, ax=axes[0, 1]) 
    sns.boxplot(x="gender", y="pos_dissimilarity", data=df, ax=axes[1, 0])
    sns.boxplot(x="gender", y="connection_words_ratio", data=df, ax=axes[1, 1])
    plt.savefig('linguistic_complexity_diversity.png')


def main():
    male, female=process_data()
    measure_complexity(male, female)
    print("Done. See linguistic_complexity_diversity.png")
    exit(0)

if __name__=="__main__":
    main()
