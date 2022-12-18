import sys
import csv
from evaluate import load
sari = load("sari")



def read_TSV_file(fn):
    sources=[]
    predictions=[]
    references=[]
    
    with open(fn) as fp:
        lines = csv.reader(fp, delimiter="\t", quotechar='"')
        for line in lines:
            sources.append(line[0])
            predictions.append(line[1])
            references.append(line[2])
    fp.close()

    return (sources, predictions, references)


def main(sources, predictions, references):
    sari_out=open("sari-results.txt","w")
    sari_score=0

    n=0
    for (source, prediction, reference) in zip(sources, predictions, references):
        n+=1
        score = sari.compute(sources=[source], predictions=[prediction], references=[[reference]])['sari'] # this assumes; just one reference (ground truth) per case
        sari_score+=score
        print("Source:%s\tPrediction:%s\tReference:%s\tSARI:%f"%(source,prediction,reference,score), file=sari_out)
            
    print("Average SARI score:",sari_score/n)
    print("See sari-results.txt for output.")

    
if __name__=="__main__":
    (sources, predictions, references)=read_TSV_file(sys.argv[1])
    main(sources, predictions, references)
    exit(0)
