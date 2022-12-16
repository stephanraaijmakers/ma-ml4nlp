from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import sys


def main(fn):
    with open(fn,"r") as fp:
        paragraphs=[line.rstrip() for line in fp.readlines()]
    fp.close()

    output_file=open("kis-output.txt","w")

    tokenizer = AutoTokenizer.from_pretrained("philippelaban/keep_it_simple")
    kis_model = AutoModelForCausalLM.from_pretrained("philippelaban/keep_it_simple")

    for paragraph in paragraphs:
        start_id = tokenizer.bos_token_id
        tokenized_paragraph = [(tokenizer.encode(text=paragraph) + [start_id])]
        input_ids = torch.LongTensor(tokenized_paragraph)
        output_ids = kis_model.generate(input_ids, max_length=150, num_beams=4, do_sample=True, num_return_sequences=8)
        output_ids = output_ids[:, input_ids.shape[1]:]
        outputs = tokenizer.batch_decode(output_ids)
        outputs = [output.replace(tokenizer.eos_token, "") for output in outputs]
        print("\nSimplification for: ", paragraph, file=output_file)
        print("================================",file=output_file)

        outputs=list(set(outputs))
        for output in outputs:
            print("\n----", file=output_file)
            print(output,file=output_file)

        print("Done, see kis-output.txt")
        
if __name__=="__main__":
    main(sys.argv[1])
