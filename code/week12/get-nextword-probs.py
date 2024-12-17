from transformers import AutoModelForCausalLM , AutoTokenizer
import sys
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from langchain import HuggingFaceHub



class LMHeadModel:

    def __init__(self, model_name):
        # Initialize the model and the tokenizer.
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def get_tokenizer(self):
        return self.tokenizer
    
    def get_predictions(self, sentence):
        # Encode the sentence using the tokenizer and return the model predictions.
        inputs = self.tokenizer.encode(sentence, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(inputs)
            predictions = outputs[0]
        return predictions
    
    def get_next_word_probabilities(self, sentence, top_k_words):
        predictions = self.get_predictions(sentence)
        next_token_candidates_tensor =  predictions[0, -1, :]
        topk_candidates_indexes = torch.topk(
            next_token_candidates_tensor, top_k_words).indices.tolist()
        topk_candidates_tokens = \
            [self.tokenizer.decode([idx]).strip() for idx in topk_candidates_indexes]
        topk_candidates_indexes=[idx for idx in topk_candidates_indexes]
        all_candidates_probabilities = torch.nn.functional.softmax(
            next_token_candidates_tensor, dim=-1)
        topk_candidates_probabilities = \
            all_candidates_probabilities[topk_candidates_indexes].tolist()
        
        return zip(topk_candidates_tokens, topk_candidates_probabilities)
        
    

def generate_probs(lines, lmm, top_k, output_file):
    print("Generating probabilities...")
    Probs=[]
    Sentences=[]
 
    for i in tqdm(range(len(lines))):
        line=lines[i]
        sentence=""
        for time in range(0, len(line)):
            word=line[time] 
            sentence+=" "+word
            probs=lmm.get_next_word_probabilities(sentence, top_k)
            Probs.append(probs)  
            Sentences.append(sentence)

    outp=open(output_file,"w")

    for (sentence, probs) in zip(Sentences, Probs):
        outp.write("Sentence:%s\n"%(sentence))
        for (word, prob) in probs:
            outp.write("%s:%f\n"%(word, prob))
        outp.write("\n\n")

    outp.close()
    print("See %s for output."%(output_file))
    return
     

def main(input_file, top_k, output_file):
    with open(input_file,"r") as f:
        lines = [z for z in [x.rstrip().split(" ") for x in f.readlines()]]
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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" # Your HuggingFace READ key here.

    #llm = HuggingFaceHub(repo_id="meta-llama/Meta-Llama-3-8B-Instruct", model_kwargs={"temperature":0.1,"max_length":128})
    llm = LMHeadModel("meta-llama/Meta-Llama-3-8B-Instruct")
    #lmm = LMHeadModel("NousResearch/Llama-2-7b-hf") # No Huggingface key necessary for this one
    generate_probs(lines, llm, top_k, output_file)
    
# Register at HuggingFace, and install the huggingface-cli (command line interface).
# Then: prior to running this script: huggingface-cli login 
# Enter your secret READ token (see Huggingface, under your profile: Access tokens)
# Then run this script. NB: it will download about 20GB model stuff. 

if __name__=="__main__":
    if len(sys.argv)!=4:
        print("Usage: python get-nextword-probs.py <sentence file: one sentence per line> <desired top k words probabilities (number)> <output file name>")
        print("Example: python3.9 get-nextword-probs.py sentences.txt 10 nextword-probs.txt")
        exit(0)
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3]) # file with one sentence per line, top k word probabilities (number), output file name
