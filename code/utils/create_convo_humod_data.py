import sys
import re



def main(fn_young, fn_old):
    fp_y=open(fn_young,"r")
    fp_o=open(fn_old,"r")
    fp_speakers=open("humod_speakers.txt","w")
    fp_lines=open("humod_lines.txt","w")
    fp_conversations=open("humod_conversations.txt","w")

    n=0
    fp_speakers.write("u0\tbot\tunknown\n")
    utt_id=0
    for line in fp_y:
        m=re.match("^([^\t]+)\t(.+)",line.rstrip())
        if m:
            n+=1
            utt_id+=1
            question_id=utt_id
            fp_speakers.write("u%d\thuman_%d\tyoung\n"%(n,n))
            bot_utt_str="L%d"%(utt_id)
            fp_lines.write("%s\tu0\t%s\n"%(bot_utt_str, m.group(1))) 
            utt_id+=1
            human_utt_str="L%d"%(utt_id)
            fp_lines.write("%s\tu%d\t%s\n"%(human_utt_str,n,m.group(2)))
            con="[%s]"%(','.join(["\""+bot_utt_str+"\"", "\""+human_utt_str+"\""]))
            fp_conversations.write("u0\tu%d\t%s\n"%(n,con))
    fp_y.close

    for line in fp_o:
        m=re.match("^([^\t]+)\t(.+)",line.rstrip())
        if m:
            n+=1
            utt_id+=1
            question_id=utt_id
            fp_speakers.write("u%d\thuman_%d\told\n"%(n,n))
            bot_utt_str="L%d"%(utt_id)
            fp_lines.write("%s\tu0\t%s\n"%(bot_utt_str, m.group(1)))
            utt_id+=1
            human_utt_str="L%d"%(utt_id)
            fp_lines.write("%s\tu%d\t%s\n"%(human_utt_str, n,m.group(2)))
            print(n)
            con="[%s]"%(','.join(["\""+bot_utt_str+"\"", "\""+human_utt_str+"\""]))
            fp_conversations.write("u0\tu%d\t%s\n"%(n,con))
    fp_o.close

    fp_speakers.close()
    fp_lines.close()
    #fp_utterances.close()
    fp_conversations.close()

    
            
if __name__=="__main__":
    main(sys.argv[1],sys.argv[2]) # young, old
        
    
