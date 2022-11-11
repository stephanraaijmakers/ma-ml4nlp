import sys
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def main(s):
    print(tokenizer.tokenize(s))

if __name__=="__main__":
    main(sys.argv[1])


    
