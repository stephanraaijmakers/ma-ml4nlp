from transformers import AutoModel, AutoTokenizer


def main():
    bert = AutoModel.from_pretrained('bert-base-uncased')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    prompt1=""

    print()
    while prompt1!="stop":
        prompt1=input('Input prompt 1: ("stop" to stop) >>> ').rstrip()
        if prompt1=="stop":
            break
        prompt2=input('Input prompt 2: >>> ').rstrip()

        prompt1_inputs = tokenizer(prompt1, return_tensors='pt')
        prompt2_inputs = tokenizer(prompt2, return_tensors='pt')
        prompt1_outputs = bert(**prompt1_inputs)
        prompt2_outputs = bert(**prompt2_inputs)
        vec1 = prompt1_outputs[1]
        vec2 = prompt2_outputs[1]
        print("First 10 dimension of entire sequence vector:",vec1[0][:10])
        print("First 10 dimension of entire sequence vector:",vec2[0][:10])


if __name__=="__main__":
    main()
    
