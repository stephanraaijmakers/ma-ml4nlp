from statsmodels.stats.contingency_tables import mcnemar
import random

def make_contingency_table(results_1, results_2, ground_truth, alpha=.05):
    assert len(results_1)==len(results_2)==len(ground_truth),"Error: incompatible sizes."
    Results={}
    Results[(1,1)]=0 # classifier 1 right, classifier 2 wrong (etc)
    Results[(0,1)]=0
    Results[(1,0)]=0
    Results[(0,0)]=0
    performance_1=0
    performance_2=0

    nb_observations=len(results_1)
    for i in range(nb_observations):
        score_1=0
        score_2=0
        if results_1[i]==ground_truth[i]:
            score_1=1
            performance_1+=1
        if results_2[i]==ground_truth[i]:
            score_2=1
            performance_2+=1
        Results[(score_1,score_2)]+=1
    # see https://en.wikipedia.org/wiki/McNemar%27s_test
    table=[[Results[(1,1)],Results[(1,0)]],
           [Results[(0,1)],Results[(0,0)]]]
    m = mcnemar(table, exact=False) # Use False for large samples, and True for small samples (<100)
    # See https://www.statsmodels.org/stable/generated/statsmodels.stats.contingency_tables.mcnemar.html

    print('McNemar statistic=%.6f, p-value=%.3f' % (m.statistic, m.pvalue))
    
    if m.pvalue > alpha:
        print("Failure to reject H0. No significant difference between classifier A and classifier B.")
    else:
        print("H0 rejected. Significant difference between classifier A and classifier B.")
        if performance_1>performance_2:
            print("A is better than B")
        else:
            print("B is better than A")

classifier_1=random.sample(range(0,100),100)
classifier_2=random.sample(range(0,100),100)
gt=list(range(100))
make_contingency_table(classifier_1,classifier_2,gt)
