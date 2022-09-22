from convokit import Corpus, Speaker, Utterance
from collections import defaultdict
from convokit import meta_index
import os.path
from tqdm import tqdm
import sys
import re
import ast

def process_data_dailydialog(fn_speakers, fn_utterances, fn_conversations):
  
    with open(fn_speakers,"r", encoding='utf-8', errors='ignore') as f:
        speaker_data = f.readlines()

    speaker_meta = {}
    for speaker in speaker_data:
        speaker_info = [info.strip() for info in speaker.split("\t")]
        # id<TAB>name<TAB>age (young/old)
        speaker_meta[speaker_info[0]] = {
            "name": speaker_info[0],
            "gender":speaker_info[1]
            }
    corpus_speakers = {k: Speaker(id = k, meta = v) for k,v in speaker_meta.items()}
    print("Number of speakers in the data = {}".format(len(corpus_speakers)))
    


    with open(fn_utterances, "r", encoding='utf-8', errors='ignore') as f:
        utterance_data = f.readlines()

    utterance_corpus = {}

    count = 0
    for utterance in tqdm(utterance_data):
        utterance_info = [info.strip() for info in utterance.split("\t")]
        if len(utterance_info) < 3:
            print(utterance_info)
        try:
            line_id, speaker, text, sentiment = utterance_info[0], utterance_info[1], utterance_info[2], utterance_info[3]
        except:
            print("ERROR:",utterance_info)
            continue
        
        meta = {"sentiment": sentiment}
        #print(meta)

        utterance_corpus[line_id] = Utterance(id=line_id, speaker=corpus_speakers[speaker], text=text, meta=meta)

    print("Total number of utterances = {}".format(len(utterance_corpus)))

    with open(fn_conversations, "r", encoding='utf-8', errors='ignore') as f:
        convo_data = f.readlines()


    for info in tqdm(convo_data):
        speaker1, speaker2, convo = [info.strip() for info in info.split("\t")]
        
        
        try:
            convo_seq = ast.literal_eval(convo)
        except:
            print("Bad:", convo)
            continue
    
        # update utterance
        conversation_id = convo_seq[0]
    
        # convo_seq is a list of utterances ids, arranged in conversational order
        for i, line in enumerate(convo_seq):
            # sanity checking: speaker giving the utterance is indeed in the pair of characters provided
            if utterance_corpus[line].speaker.id not in [speaker1, speaker2]:
                print("speaker mismatch in line {0}".format(i))
                print(utterance_corpus[line])
                print(utterance_corpus[line].speaker.id)
                print([speaker1,speaker2])
            utterance_corpus[line].conversation_id = conversation_id       
            if i == 0:
                utterance_corpus[line].reply_to = None
            else:
                utterance_corpus[line].reply_to = convo_seq[i-1]


    utterance_list = utterance_corpus.values()

    corpus = Corpus(utterances=utterance_list)
    corpus.dump("/home/stephan/.convokit/saved-corpora/dailydialog-corpus")
    convo_ids = corpus.get_conversation_ids()
    for i, convo_idx in enumerate(convo_ids[0:5]):
        print("sample conversation {}:".format(i))
        print(corpus.get_conversation(convo_idx).get_utterance_ids())

    meta_index(filename = os.path.join(os.path.expanduser("~"), ".convokit/saved-corpora/dailydialog-corpus"))
    

if __name__=="__main__":
    process_data_dailydialog(sys.argv[1], sys.argv[2],sys.argv[3])


        
