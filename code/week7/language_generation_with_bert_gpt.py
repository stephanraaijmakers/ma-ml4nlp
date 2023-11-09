import os
import torch
from torch.nn import functional as F
import string
from transformers import pipeline, BertTokenizer, BertForMaskedLM, AutoModelForCausalLM, OpenAIGPTModel,AutoModelWithLMHead, AutoTokenizer, top_k_top_p_filtering, logging, AutoModelForSeq2SeqLM, set_seed


logging.set_verbosity_error()
# declare variables
no_words_to_be_predicted = globals()
select_model = globals()
prompt = globals()

# set model configuration
def set_model_config(**kwargs):
  no_words_to_be_predicted = list(kwargs.values())[0] # integer values
  select_model = list(kwargs.values())[1] # possible values = 'bert' or 'gpt'
  prompt = list(kwargs.values())[2] #only string

  return no_words_to_be_predicted, select_model, prompt

# load model and tokenizer
def load_model(model_name):
  try:
    if model_name.lower() == "bert":
      bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
      bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased').eval()
      return bert_tokenizer,bert_model
    elif model_name.lower() == "gpt":
      gpt_tokenizer = AutoTokenizer.from_pretrained("gpt2")
      gpt_model = AutoModelWithLMHead.from_pretrained("gpt2")
      return gpt_tokenizer,gpt_model
    elif model_name.lower() == "gpt3":
      gpt_tokenizer = AutoTokenizer.from_pretrained("openai-gpt")
      gpt_model = OpenAIGPTModel.from_pretrained("openai-gpt")
      return gpt_tokenizer,gpt_model
  except Exception as e:
    pass

# bert encode
def encode_bert(tokenizer, text_sentence, add_special_tokens=True):
  text_sentence = text_sentence.replace('<mask>', tokenizer.mask_token)
  # if <mask> is the last token, append a "." so that models dont predict punctuation.
  if tokenizer.mask_token == text_sentence.split()[-1]:
    text_sentence += ' .'
    input_ids = torch.tensor([tokenizer.encode(text_sentence, add_special_tokens=add_special_tokens)])
    mask_idx = torch.where(input_ids == tokenizer.mask_token_id)[1].tolist()[0]
  return input_ids, mask_idx

# bert decode
def decode_bert(tokenizer, pred_idx, top_clean):
  ignore_tokens = string.punctuation + '[PAD]'
  tokens = []
  for w in pred_idx:
    token = ''.join(tokenizer.decode(w).split())
    if token not in ignore_tokens:
      tokens.append(token.replace('##', ''))
  return '\n'.join(tokens[:top_clean])

# gpt encode
def encode_gpt(tokenizer, text_sentence, add_special_tokens=False):
  input_ids = tokenizer.encode(text_sentence, return_tensors="pt")
  return input_ids

# gpt decode
def decode_gpt(tokenizer, input_ids, pred, top_clean):
  filtered_next_token_logits = top_k_top_p_filtering(pred, top_k=top_clean, top_p=1.0)

  # sample
  probs = F.softmax(filtered_next_token_logits, dim=-1)
  next_token = torch.multinomial(probs, num_samples=top_clean)
  generated = torch.cat([input_ids, next_token], dim=-1)  
  resulting_string = tokenizer.decode(generated.tolist()[0])
  return resulting_string

def get_all_predictions(text_sentence,  model_name, top_n, top_clean=5):
  if model_name.lower() == "bert":
    # ========================= BERT =================================
    input_ids, mask_idx = encode_bert(bert_tokenizer, text_sentence)
    with torch.no_grad():
      predict = bert_model(input_ids)[0]
    bert = decode_bert(bert_tokenizer, predict[0, mask_idx, :].topk(top_n).indices.tolist(), top_clean)
    return {'bert': bert}

  #elif model_name.lower() == "gpt2":
    # ========================= GPT =================================
    #input_ids = encode_gpt(gpt_tokenizer, text_sentence)
    #with torch.no_grad():
    #  predict = gpt_model(input_ids)[0][:, -1, :]
    #gpt = decode_gpt(gpt_tokenizer, input_ids, predict, top_clean)
    #return {'gpt': gpt}
  


def get_prediction_end_of_sentence(prompt, model_name, top_n):
  try:
    if model_name.lower() == "bert":
      prompt += ' <mask>'
      #print(prompt)
      res = get_all_predictions(prompt, model_name, top_n) 
      return res
    else:
      #print(prompt)
      res = get_all_predictions(prompt, model_name, top_n)
      return res

  except Exception as error:
    pass


#no_words_to_be_predicted, select_model, prompt = set_model_config(no_words_to_be_predicted=5, select_model = "bert", prompt = "this is a great")

prompt="go"

print("#################### Text generation with BERT/GPT2 #################\n")
print("Loading models...")
bert_tokenizer, bert_model  = load_model("bert")

checkpoint_gpt2 = "gpt2-large"
tokenizer_gpt2 = AutoTokenizer.from_pretrained(checkpoint_gpt2)
model_gpt2 = AutoModelForCausalLM.from_pretrained(checkpoint_gpt2)

print("Done.")

select_model=input('Choose a model (bert, gpt2): ')

while True:
  if select_model.lower() == "bert":
    prompt=input('Input: ').rstrip()
    if prompt=="switch":
        select_model=input('Choose a model (bert, gpt2): ')
        continue
    if prompt=="stop":
      break
    nb_completions=int(input('How many words should I generate? '))
    answer_bert = prompt.split(" ")
    for k in range(nb_completions):
      candidates=[]
      res = get_prediction_end_of_sentence(prompt, select_model, nb_completions)
      for i in res['bert'].split("\n"):
        candidates.append(i)
      prompt+=" "+candidates[0]
      answer_bert.append(candidates[0])
    print("Generated text:",' '.join(answer_bert))

  elif select_model.lower() == "gpt2":
    prompt=input('Input: ').rstrip()
    if prompt=="switch":
        select_model=input('Choose a model (bert, gpt2): ')
        continue
    if prompt=="stop":
      break
    nb_completions=int(input('How many words should I generate? '))
    inputs = tokenizer_gpt2(prompt, return_tensors="pt")
    outputs = model_gpt2.generate(**inputs, penalty_alpha=0.6, top_k=4, max_new_tokens=nb_completions+1)
    answer=tokenizer_gpt2.batch_decode(outputs, skip_special_tokens=True)
    print("Generated text:",answer)



#from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, set_seed
#set_seed(0)  # For reproducibility
#prompt = "translate English to German: The house is wonderful."
#checkpoint = "t5-small"
#tokenizer = AutoTokenizer.from_pretrained(checkpoint)
#inputs = tokenizer(prompt, return_tensors="pt")
#model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
#outputs = model.generate(**inputs, num_beams=5, do_sample=True)
#tokenizer.decode(outputs[0], skip_special_tokens=True)
# => 'Das Haus ist wunderbar.'

