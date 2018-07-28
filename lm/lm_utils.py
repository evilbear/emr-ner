#encoding:utf-8
import numpy as np
import logging

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
    with open(filename) as f:
        for idx, word in enumerate(f):
            word = word.strip('\n')
            vocab[word] = idx
    return vocab     

class PreData(object):
    def __init__(self, vocab, train_path=None, dev_path=None, test_path=None):
        self.vocab = vocab
        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        
    def getVocabSize(self):
        return len(self.vocab)
    
    def yieldSpliceBatch(self, tag, batch_size, step_size, reverse):
        eos = self.vocab["$EOS$"]
        if tag == 'Train':
            _file = self.train_path
        elif tag == 'Dev':
            _file = self.dev_path
        else:
            _file = self.test_path
            
        data = []
        for line in open(_file):
            tokens = line.strip().split(' ')
            data += tokens + [eos]

        #反向LM
        if reverse:
            data.pop()
            data.reverse()
            data.append(eos)

        data_len = len(data)
        batch_len = data_len // batch_size
        batch_num = (batch_len -1) // step_size
        if batch_num == 0:
            raise ValueError("batch_num == 0, decrease batch_size or step_size")
        
        word_data = np.zeros([batch_size, batch_len], dtype=np.int32)
        for i in range(batch_size):
            index = i * batch_len
            word_data[i] = data[index : index + batch_len]
        for batch_id in range(batch_num):
            index = step_size * batch_id
            x = word_data[:, index : index + step_size]
            y = word_data[:, index + 1 : index + step_size + 1]
            n = batch_size * step_size
            yield(batch_id, batch_num, x, y, n)
            
       