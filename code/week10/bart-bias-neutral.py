from transformers import BartTokenizer, BartForConditionalGeneration 
import re

tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
model = BartForConditionalGeneration.from_pretrained('cffl/bart-base-styletransfer-subjective-to-neutral')


inp=""
print("Type a sentence, # to stop.")
while inp!="#":
    inp=input("\nSentence>").rstrip()
    if inp=="#":
        break
        
    inputs = tokenizer(inp, return_tensors="pt")
    outputs = model(**inputs)
    generated_ids = model.generate(inputs["input_ids"], max_new_tokens=100)
    print("Unbiased>"+re.sub("<\/?s>","",tokenizer.decode(generated_ids[0])))
    print()
          
    
