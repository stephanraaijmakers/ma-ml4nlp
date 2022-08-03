import sys
import re

from collections import OrderedDict



def main(fn_char, fn_conv):
    fp_char=open(fn_char,"r") # simpsons_characters.csv
    fp_conv=open(fn_conv,"r") # simpsons_dataset.csv
    
    fp_speakers=open("simpsons_speakers.txt","w")
    fp_lines=open("simpsons_lines.txt","w")
    fp_conversations=open("simpsons_conversations.txt","w")

    UID={}
    for line in fp_char:
        m=re.match("^(\d+),([^,]+),[^,]+,([mf]*)",line.rstrip().lstrip())
        if m:
            id=m.group(1)
            name=m.group(2).replace(" ","_")
            UID[name]="u%s"%(id)
            
            gender=m.group(3)
            if gender=="":
                gender="?"
            fp_speakers.write("u%s\t%s\n"%(id, gender))
    fp_char.close()


    utt_id=0
    speakerD=OrderedDict()
    conv=[]    
    for line in fp_conv:

        rootSpeaker=""
        m=re.match("^([^,]+),(.+)",line.rstrip())

        if m:
            speaker=m.group(1).replace(" ","_")
            if speaker not in UID:
                continue

            currentSpeaker=UID[speaker]
            utterance=m.group(2)
            utt_label="L%d"%(utt_id)
            fp_lines.write("%s\t%s\t%s\n"%(utt_label,currentSpeaker, utterance))

            if len(speakerD)==2:
                if currentSpeaker not in speakerD:
                    speakers=list(speakerD.keys())
                    #con="[%s]"%(','.join(["\""+bot_utt_str+"\"", "\""+human_utt_str+"\""]))
                    fp_conversations.write("%s\t%s\t%s\n"%(speakers[0], speakers[1],str(conv)))
                    speakerD.clear()
                    speakerD[currentSpeaker]=1
                    conv=[utt_label]
                else:
                    conv.append(utt_label)
            else:
                speakerD[currentSpeaker]=1
                conv.append(utt_label)
                
            utt_id+=1
    fp_conv.close()
    
    fp_lines.close()
    fp_conversations.close()

    
            
if __name__=="__main__":
    main(sys.argv[1],sys.argv[2]) # characters, conversations
        
    
