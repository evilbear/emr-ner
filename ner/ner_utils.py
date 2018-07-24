import os, argparse, codecs, logging
import numpy as np

def str2bool(v):
    # copy from StackOverflow
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_logger(filename):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)
    return logger

def get_vocab(filename):
    vocab = dict()
    with codecs.open(filename) as f:
        for idx, word in enumerate(f):
            word = word.strip('\n')
            vocab[word] = idx
    return vocab 

def get_corpus(filename, vocab_words, vocab_tags):
    with codecs.open(filename) as f:
        words, tags, corpus, words_temp, tags_temp = [], [], [], [], []
        for line in f:
            line = line.strip('\n')
            if len(line) == 0:
                corpus.append([words, tags])
                words, tags = [], []
            else:
                ls = line.split(' ')
                word, tag = ls[0], ls[-1]
                if word.isdigit():
                    word = vocab_words['$NUM$']
                elif word == '':
                    word = vocab_words[" "]
                elif word in vocab_words:
                    word = vocab_words[word]
                else:
                    word = vocab_words['$UNK$']
                tag = vocab_tags[tag]
                words_temp += [word]
                tags_temp += [tag]
                if ls[0] == '。':
                    words.append(words_temp)
                    tags.append(tags_temp)
                    words_temp, tags_temp = [], []
    return corpus

def pad_sequences(seqs):
    max_length = max(map(lambda x: len(x), seqs))
    sequence_padded, sequence_length = [], []
    for seq in seqs:
        seq = list(seq)
        seq_ = seq[:max_length] + [0]*max(max_length - len(seq), 0)
        sequence_padded.append(seq_)
        sequence_length.append(min(len(seq), max_length))
    return sequence_padded, sequence_length

def get_tag(str1):
    if str1 == 1:
        return('B-AS')
    elif str1 == 2:
        return('I-AS')
    elif str1 == 3:
        return('B-SYM')
    elif str1 == 4:
        return('I-SYM')
    elif str1 == 5:
        return('B-ISYM')
    elif str1 == 6:
        return('I-ISYM')
    elif str1 == 7:
        return('B-DRUG')
    elif str1 == 8:
        return('I-DRUG')
    elif str1 == 9:
        return('B-SUR')
    elif str1 == 10:
        return('I-SUR')
    else:
        return('O')

def get_doc2vec(filename):
    with codecs.open(filename) as f:
        corpus, words = [], []
        for line in f:
            line = line.strip('\n')
            if len(line) == 0:
                corpus.append(words)
                words = []
            else:
                word = line.split(' ')[0]
                if word == '':
                    word = ' '
                words += [word]
    return corpus

def get_lm_embeddings(mode, parm, step):
    embedding_save_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/data/lm/'+ mode+'_'+ parm+'/'+ mode+'_'+str(step)+'.npz'
    embedding_txt_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/data/lm/'+ mode+'/'+mode+'_'+str(step)+'.txt'
    line_list = []
    with open(embedding_txt_path, 'r') as f:
        for i in f:
            i = i.strip().split(' ')
            line_list += [len(i)]
    with np.load(embedding_save_path) as data:
        embeddings = data["embeddings"]
    embeddings = embeddings.tolist()
    lm_embeddings = []
    max_length = max(line_list)
    num = 0
    for i in line_list:
        pad_len = max(max_length - i, 0)
        lm_embeddings += [embeddings[num:num+i] + [[0.]*300]*pad_len]
        num += i + 1
    lm_embeddings = np.array(lm_embeddings, dtype='float32')
    return lm_embeddings