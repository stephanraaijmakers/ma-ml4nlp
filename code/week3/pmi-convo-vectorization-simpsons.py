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

# Reuses some of https://www.kaggle.com/code/gabrielaltay/word-vectors-from-pmi-matrix/notebook


def process_data():
#    simpsons_corpus = Corpus(download('simpsons-corpus'))
    print("Loading data...")

    simpsons_corpus=Corpus(filename='/home/stephan/.convokit/saved-corpora/simpsons-corpus')
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


def get_priors(texts):
    priors=Counter()
    n=0
    for text in texts:
        for word in text.split(" "):
            priors[word]+=1
            n+=1
    return priors


def get_joints(texts, left, right, tok2indx):
    joints=Counter()
    n=0
    for text in texts:
        tokens = [tok2indx[tok] for tok in text.split(" ")]
        for ii_word, word in enumerate(tokens):
            ii_context_min = max(0, ii_word - left)
            ii_context_max = min(len(text) - 1, ii_word + right)
            ii_contexts = [ii for ii in range(ii_context_min, ii_context_max+1) if ii != ii_word]
            for ii_context in ii_contexts:
                if ii_context>len(tokens)-1:
                    break
                skipgram = (tokens[ii_word], tokens[ii_context])
                joints[skipgram] += 1    
    return joints


def create_indexes(priors):
    tok2indx = {tok: indx for indx, tok in enumerate(priors.keys())}
    indx2tok = {indx: tok for tok,indx in tok2indx.items()}
    return tok2indx, indx2tok

def most_similar(word, mat, topn, tok2indx,indx2tok):
    indx = tok2indx[word]
    if isinstance(mat, sparse.csr_matrix):
        v1 = mat.getrow(indx)
    else:
        v1 = mat[indx:indx+1, :]
    sims = cosine_similarity(mat, v1).flatten()
    sindxs = np.argsort(-sims)
    sim_word_scores = [(indx2tok[sindx], sims[sindx]) for sindx in sindxs[0:topn]]
    return sim_word_scores

def create_ppmi(priors, joints):
    row_indxs = []
    col_indxs = []
    dat_values = []
    ii = 0
    for (tok1, tok2), sg_count in joints.items():
        ii += 1
        row_indxs.append(tok1)
        col_indxs.append(tok2)
        dat_values.append(sg_count)
    wwcnt_mat = sparse.csr_matrix((dat_values, (row_indxs, col_indxs)))
    num_skipgrams = wwcnt_mat.sum()
    assert(sum(joints.values())==num_skipgrams)
    row_indxs = []
    col_indxs = []
    ppmi_dat_values = []   # positive pointwise mutial information
    sum_over_words = np.array(wwcnt_mat.sum(axis=0)).flatten()
    sum_over_contexts = np.array(wwcnt_mat.sum(axis=1)).flatten()
    alpha = 0.75
    sum_over_words_alpha = sum_over_words**alpha
    nca_denom = np.sum(sum_over_words_alpha)
    ii = 0
    for (tok_word, tok_context), sg_count in tqdm(joints.items()):
        ii += 1
        nwc = sg_count
        Pwc = nwc / num_skipgrams
        nw = sum_over_contexts[tok_word]
        Pw = nw / num_skipgrams
        nc = sum_over_words[tok_context]
        Pc = nc / num_skipgrams
        nca = sum_over_words_alpha[tok_context]
        Pca = nca / nca_denom
        pmi = np.log2(Pwc/(Pw*Pc))  
        ppmi = pmi # max(pmi, 0)
        row_indxs.append(tok_word)
        col_indxs.append(tok_context)
        ppmi_dat_values.append(ppmi)
        ppmi_mat = sparse.csr_matrix((ppmi_dat_values, (row_indxs, col_indxs)))                                                                                                                                              
    return ppmi_mat
    

def main():
    male, female=process_data()
    out=open("simpsons-texts.txt","w")
    for t in male:
        out.write("Male:"+t+"\n")
    for t in female:
        out.write("Female:"+t+"\n")
    out.close()
    print("See simpsons-texts.txt")
    priors=get_priors(female) # male
    print("Priors done.")
    tok2indx,indx2tok=create_indexes(priors)
    print("Indexes done.")
    joints=get_joints(female,2,2, tok2indx) # male
    print("Joints done. Now computing PMI.")
    mat=create_ppmi(priors, joints)
    print("PPMI done.")
    common=priors.most_common(10)
    for (word, freq) in common:
        print(word, [w for w in most_similar(word,mat,5,tok2indx, indx2tok) if w[0]!=word]) # 5,10,20...

    inp=""
    while inp!="#":
        inp=input("Type a word:")
        if inp=="#":
            break
        if inp not in tok2indx:
            print("Word not in lexicon.")
            continue
        print(inp, [w for w in most_similar(inp,mat,25,tok2indx, indx2tok) if w[0]!=inp])    



if __name__=="__main__":
    main()

    
