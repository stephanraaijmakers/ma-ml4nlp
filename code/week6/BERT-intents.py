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

from tqdm import tqdm


from transformers import AutoModelWithLMHead, AutoTokenizer

#https://huggingface.co/mrm8488/t5-base-finetuned-e2m-intent
tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-e2m-intent")
intent_model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-e2m-intent")


def get_intent(event, max_length=16):
  input_text = "%s </s>" % event
  features = tokenizer([input_text], return_tensors='pt')

  output = intent_model.generate(input_ids=features['input_ids'], 
               attention_mask=features['attention_mask'],
               max_length=max_length)

  intent=tokenizer.decode(output[0])
  intent=re.sub("<[^\>]+>","",intent)
  intent=re.sub(" ","_",intent)
  return(intent)
 


def get_intents(dialogues,fp): 
    for dialogue in tqdm(dialogues):      
        dialogue=dialogue[0] # unpack
        fp.write("----------------------\n\n")
        for sentence in dialogue:
            intent=get_intent(sentence)
            fp.write("%s\t%s\n"%(sentence, intent))
        
            
def read_dialogue_data(fn): 
    dialogues=[]
    #labels=[]
    fp=open(fn,"r")
    for line in fp:
        fields=line.rstrip().split("\t")
        if fields:
            dialogues.append([fields]) # fields[1:] if labeled
            #labels.append(fields[0])
    fp.close()
    return dialogues #, labels

            

def main(fn):
    dialogues=read_dialogue_data(fn)
    fp=open("intents.txt","w")
    get_intents(dialogues,fp)
    fp.close()
    print("See intents.txt")  
    

if __name__=="__main__":
    main(sys.argv[1])
