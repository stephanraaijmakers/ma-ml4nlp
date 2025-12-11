import sys
import re
from statsmodels.stats.contingency_tables import mcnemar

# Input: a file with on every line one of M, N, MN, 00 (+ newline)
# M: classifier 1 correct, classifier 2 wrong
# N: classifier 1 wrong, classifier 2 right
# MN: both correct
# 00: both wrong

# Run: !python mcnemar.py <file>



def main(fn):
    fp=open(fn,"r")
    N=0
    P_M=0.0
    P_N=0.0
    P_MN=0.0
    P_None=0.0
    for line in fp:
        line=line.rstrip()
        m=re.match("^([MN0]+)\s*$",line)
        if m:
            N+=1
            tag=m.group(1)
            if tag=="M":
                P_M+=1
            elif tag=='N':
                P_N+=1
            elif tag=="MN":
                P_MN+=1
            elif tag=="00": # both wrong
                P_None+=1

    #print(P_MN,P_M,P_N,P_None)

    # define contingency table
    table = [[P_MN,P_M],
         [P_N,P_None]]
    # calculate mcnemar test
    result = mcnemar(table, exact=True)
    # summarize the finding
    print('statistic=%.3f, p-value=%f' % (result.statistic, result.pvalue))
    # interpret the p-value
    alpha = 0.05
    if result.pvalue > alpha:
        print('Same proportions of errors (fail to reject H0; p>.05)')
    else:
        print('Different proportions of errors (reject H0; p<.05)')
        if P_M>P_N:
            print("Metadata better than no metadata")
        else:
            print("No metadata better than metadata")
    fp.close()
    exit(0)

if __name__=="__main__":
    main(sys.argv[1])
