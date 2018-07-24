#encoding:utf-8
import chardet, os, codecs, argparse, gensim
import numpy as np
from ner_model import BLSTM_CRF
from ner_utils import str2bool, get_logger, get_vocab, get_corpus, get_doc2vec

# hyperparameters
parser = argparse.ArgumentParser(description='CNER')
parser.add_argument('--train_data', type=str, default='train1.txt', help='train data source')
parser.add_argument('--dev_data', type=str, default='dev.txt', help='dev data source')
parser.add_argument('--test_data', type=str, default='testtest.txt', help='test data source')
parser.add_argument('--output_path', type=str, default='/result/', help='output path')
parser.add_argument('--epoch', type=int, default=100, help='#epoch of training')
parser.add_argument('--hidden_dim', type=int, default=300, help='#dim of hidden state')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--use_model', type=str, default='baseline', help='baseline, doc2vec_concat, doc2vec_att, lms_concat, lms_att')
parser.add_argument('--mode', type=str, default='train', help='train or test')
args = parser.parse_args()

#get word embeddings
embedding_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/data/vectors.npz'
with np.load(embedding_path) as data:
    embeddings = data["embeddings"]
embeddings = np.array(embeddings, dtype='float32')

#get words tags chars vocab
words_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/data/words.txt'
tags_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/data/tags.txt'
vocab_words = get_vocab(words_path)
vocab_tags = get_vocab(tags_path)

# paths setting
output_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + args.output_path
if not os.path.exists(output_path): os.makedirs(output_path)
log_path = output_path + 'log.txt'
logger = get_logger(log_path)
logger.info(str(args))

#Model controlled by parameters
use_doc2vec, use_lm, use_att = False, False, False
if args.use_model == 'doc2vec_concat':
    use_doc2vec = True
elif args.use_model == 'doc2vec_att':
    use_doc2vec, use_att = True, True
elif args.use_model == 'lms_concat':
    use_lm = True
elif args.use_model == 'lms_att':
    use_lm, use_att = True, True
else:
    pass


#training model
if args.mode == 'train':
    train_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/data/' + args.train_data
    dev_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/data/' + args.dev_data
    train_data = get_corpus(train_path, vocab_words, vocab_tags)
    dev_data = get_corpus(dev_path, vocab_words, vocab_tags)
    if use_doc2vec == True:
        train_doc2vec_data = get_doc2vec(train_path)
        dev_doc2vec_data = get_doc2vec(dev_path)
        doc2vec_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/doc2vec/all/model'
        doc2vec_model = gensim.models.Doc2Vec.load(doc2vec_path)
        doc2vec_model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
        train_doc2vec, dev_doc2vec = [], []
        for i in train_doc2vec_data:
            train_doc2vec.append(doc2vec_model.infer_vector(i, alpha=0.01, steps=1000))
        for j in dev_doc2vec_data:
            dev_doc2vec.append(doc2vec_model.infer_vector(j, alpha=0.01, steps=1000))
    model = BLSTM_CRF(epoch_num=args.epoch, hidden_dim=args.hidden_dim,
                    embeddings=embeddings, lr=args.lr, vocab_words=vocab_words, vocab_tags=vocab_tags,
                    output_path=output_path, logger=logger, use_doc2vec=use_doc2vec, use_lm=use_lm, use_att=use_att)
    model.build_graph()
    if use_doc2vec == True:
        model.train(train_data, dev_data, train_doc2vec, dev_doc2vec)
    else:
        model.train(train_data, dev_data)
elif args.mode == 'test':
    test_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/data/' + args.test_data
    test_data = get_corpus(test_path, vocab_words, vocab_tags)
    if use_doc2vec == True:
        test_doc2vec_data = get_doc2vec(test_path)
        doc2vec_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/doc2vec/all/model'
        doc2vec_model = gensim.models.Doc2Vec.load(doc2vec_path)
        doc2vec_model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
        test_doc2vec = []
        for i in test_doc2vec_data:
            test_doc2vec.append(doc2vec_model.infer_vector(i, alpha=0.01, steps=1000))
    model = BLSTM_CRF(epoch_num=args.epoch, hidden_dim=args.hidden_dim,
                    embeddings=embeddings, lr=args.lr, vocab_words=vocab_words, vocab_tags=vocab_tags,
                    output_path=output_path, logger=logger, use_doc2vec=use_doc2vec, use_lm=use_lm, use_att=use_att)
    model.build_graph()
    model.restore_session(output_path)
    if use_doc2vec == True:
        model.test(test_data, test_doc2vec)
    else:
        model.test(test_data)
else:
    pass