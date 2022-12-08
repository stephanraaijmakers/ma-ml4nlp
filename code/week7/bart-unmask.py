from transformers import BartTokenizer, BartForConditionalGeneration

tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")

txt = "My <mask> <mask> but they eat <mask> too many carbs."
input_ids = tokenizer([txt], return_tensors="pt")["input_ids"]
logits = model(input_ids).logits

batch = tokenizer(txt, return_tensors="pt")
generated_ids = model.generate(batch["input_ids"])
print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))


