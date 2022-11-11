import os
import torch
from transformers import BertTokenizer, BertForMaskedLM


# Guess what's behind the mask
def unmask_sent(text, tokenizer, model, top_k=5):

  text = "[CLS] %s [SEP]"%text
  tokenized_text = tokenizer.tokenize(text)
  masked_index = tokenized_text.index("[MASK]")
  indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
  tokens_tensor = torch.tensor([indexed_tokens])

  with torch.no_grad():
    outputs = model(tokens_tensor)
    predictions = outputs[0]  
  probs = torch.nn.functional.softmax(predictions[0, masked_index], dim=-1)
  top_k_weights, top_k_indices = torch.topk(probs, top_k, sorted=True)
    
  for i, pred_idx in enumerate(top_k_indices):
    predicted_token = tokenizer.convert_ids_to_tokens([pred_idx])[0]
    if i==0:
      best_guess=predicted_token
    token_weight = top_k_weights[i]
    print("[MASK]: '%s'"%predicted_token, " | weight:", float(token_weight))
  return best_guess  # always for the first MASK
  


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased').eval()

enter_input_text=""

while enter_input_text!="#":
  print("\n#################### Unmasking with BERT #################\n")
  enter_input_text=input('Enter a sentence, use [MASK] for masking words (type # to stop): ').rstrip()
  if enter_input_text.split(" ")[-1]=="[MASK]":
    enter_input_text+=" [MASK]" # preventing punctuation predictions for final mask
  while "[MASK]" in enter_input_text:
    unmasking=unmask_sent(enter_input_text, tokenizer, model)
    words=enter_input_text.split(" ")
    for i in range(len(words)):
      if words[i]=="[MASK]":
        words[i]=unmasking
        break
    enter_input_text=' '.join(words)
  if enter_input_text!="#":
    print("\nUnmasked:", enter_input_text)






