import sacrebleu
import re
import sys
import numpy as np

def main(fn):
    D={}
    scores=[]
    with open(fn,"r") as fp:
        for line in fp:
            m=re.match("^([^\t]+)\t(.+)$",line.rstrip())
            if m:
                D[m.group(1)]=m.group(2)
        fp.close()
                  
    for ref in D:
        score=sacrebleu.raw_corpus_bleu(ref, [D[ref]], 0.0).score / 100
        scores.append(score)

    print("Mean BLEU:", np.mean(np.array(scores)))
    exit(0)


if __name__=="__main__":
    main(sys.argv[1])

                  
        
