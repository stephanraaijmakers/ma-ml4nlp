import sys
import re
import csv	

SYSTEM_PROMPT = """You are judging sentence grammaticality.                                                                                          
                                                                                                                                                        
- If the sentence is grammatical as written, output: yes                                                                                             
- If it is ungrammatical output: no

Example:
Sentence: She goes to museum
Output: no

Sentence: She goes to the museum
Output: yes

Here are the sentences:

"""


def main(fn, n_questions, max_prompts):
  n=0
  q_id=0
  with open(fn, newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    rows =[row[2] for row in reader]
 
  while (n<max_prompts):
    q='\n'.join(['Sentence: '+s+'\nOutput:' for s in rows[q_id:q_id+n_questions]])
    print("\n"+SYSTEM_PROMPT+"\n"+q)
    q_id+=n_questions
    n+=1    	


if __name__=="__main__":
  main(sys.argv[1], int(sys.argv[2]),int(sys.argv[3]))

