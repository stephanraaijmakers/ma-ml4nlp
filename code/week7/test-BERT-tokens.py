import pandas as pd
import numpy as np
import torch
import os
from collections import OrderedDict
import csv
import sys

from transformers import BertModel, BertTokenizer

def main(fn):
    model = BertModel.from_pretrained('bert-base-uncased',
           output_hidden_states = True,)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Read sentences from file (one sentence per line.)
    with open(fn, "r") as fp:
        sentences=fp.readlines()

    context_embeddings = []
    context_tokens = []
    for sentence in sentences:
        tokenized_text, tokens_tensor, segments_tensors = tokenize_text(sentence, tokenizer)
        list_token_embeddings = get_bert_embeddings(tokens_tensor, segments_tensors, model)
        tokens = OrderedDict()
        # loop over tokens in sensitive sentence
        for token in tokenized_text[1:-1]:
        # keep track of position of word and whether it occurs multiple times
            if token in tokens:
                tokens[token] += 1
            else:
                tokens[token] = 1
            token_indices = [i for i, t in enumerate(tokenized_text) if t == token]
            current_index = token_indices[tokens[token]-1]
            token_vec = list_token_embeddings[current_index]
            context_tokens.append(token)
            context_embeddings.append(token_vec)

    filepath =  os.path.join('./')

    fname = 'metadata_bert.tsv'
    with open(os.path.join(filepath, fname), 'w+') as file_metadata:
        for i, token in enumerate(context_tokens):
          file_metadata.write(token + '\n')
    
    fname = 'embeddings_bert.tsv'
    with open(os.path.join(filepath, fname), 'w+') as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t')
        for embedding in context_embeddings:
            writer.writerow(embedding.numpy())
    print("Upload metadata_bert.tsv and embeddings_bert.tsv to ")


def tokenize_text(text, tokenizer):
    marked_text = "[CLS] " + text + " [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1]*len(indexed_tokens)
    # convert inputs to tensors
    token_tensors = torch.tensor([indexed_tokens])
    segment_tensors = torch.tensor([segments_ids])
    return tokenized_text, token_tensors, segment_tensors


def get_bert_embeddings(tokens_tensor, segments_tensor, model):
    """
    Obtains BERT embeddings for tokens.
    """
    # gradient calculation id disabled
    with torch.no_grad():
      # obtain hidden states
      outputs = model(tokens_tensor, segments_tensor)
      hidden_states = outputs[2]
    # concatenate the tensors for all layers
    # use "stack" to create new dimension in tensor
    token_embeddings = torch.stack(hidden_states, dim=0)
    # remove dimension 1, the "batches"
    token_embeddings = torch.squeeze(token_embeddings, dim=1)
    # swap dimensions 0 and 1 so we can loop over tokens
    token_embeddings = token_embeddings.permute(1,0,2)
    # intialized list to store embeddings
    token_vecs_sum = []
    # "token_embeddings" is a [Y x 12 x 768] tensor
    # where Y is the number of tokens in the sentence
    # loop over tokens in sentence
    for token in token_embeddings:
    # "token" is a [12 x 768] tensor
    # sum the vectors from the last four layers
        sum_vec = torch.sum(token[-4:], dim=0)
        token_vecs_sum.append(sum_vec)
    return token_vecs_sum


if __name__=="__main__":
    main(sys.argv[1])


