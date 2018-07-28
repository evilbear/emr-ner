#encoding:utf-8
import os, argparse, datetime
import numpy as np
from lm_model import Config, LM, LearningRateUpdater, train, test
import tensorflow as tf
from lm_utils import str2bool, get_logger, get_vocab, PreData
from pre_data import con_data

# hyperparameters
parser = argparse.ArgumentParser(description='CCKS2018 LM task')
parser.add_argument('--pre_data', type=str2bool, default=False, help='whether pre data')
parser.add_argument('--train_data', type=str, default='/lm/lm_train.txt', help='train data source')
parser.add_argument('--dev_data', type=str, default='/lm/lm_dev.txt', help='dev data source')
parser.add_argument('--use_model', type=str, default='fw_model', help='fw_model or bw_model')
parser.add_argument('--mode', type=str, default='train', help='train or test')
args = parser.parse_args()

#Positioning parameters
train_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/data' + args.train_data
dev_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/data' + args.dev_data
model_path = os.path.dirname(os.path.abspath(__file__)) + '/' + args.use_model + '/'
if not os.path.exists(model_path): os.makedirs(model_path)
log_path = model_path + 'log.txt'
embedding_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/data/vectors.npz'
words_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/data/words.txt'
with np.load(embedding_path) as data:
    embeddings = data["embeddings"]
embeddings = np.array(embeddings, dtype='float32')
vocab_words = get_vocab(words_path)
vocab_size = len(vocab_words)
logger = get_logger(log_path)
logger.info(str(args))

if args.pre_data == True:
    con_data(vocab_words)

if args.use_model == 'fw_model':
    reverse = False
else:
    reverse = True

if args.mode == 'train':
    pre_train = PreData(vocab=vocab_words, train_path=train_path, dev_path=dev_path)
    config = Config()
    lr_updater = LearningRateUpdater(config.learning_rate, config.decay, config.decay_when)
    
    graph = tf.Graph()
    with graph.as_default():
        trainm = LM(config=config, mode="Train", vocab_size=vocab_size, embeddings=embeddings, reuse=False)
        devm = LM(config=config, mode="Dev", vocab_size=vocab_size, embeddings=embeddings, reuse=True)
        
    with tf.Session(graph=graph) as session:
        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        for epoch in range(config.epoch_num):
            logger.info("Epoch {}, start time: {}".format(epoch + 1, datetime.datetime.now().strftime("%H:%M")))
            trainm.update_lr(session, lr_updater.get_lr())
            logger.info("Epoch {}, learning rate: {}".format(epoch + 1, lr_updater.get_lr()))
            print("Start training {}".format(datetime.datetime.now().strftime("%H:%M")))
            cost, word_cnt, ppl = train(session, trainm, pre_train, logger, reverse)
            logger.info("Epoch %d Train perplexity %.3f words %d" % (epoch + 1, ppl, word_cnt))
            print("End training {}".format(datetime.datetime.now().strftime("%H:%M")))
            print("Start deving")
            cost, word_cnt, ppl = train(session, devm, pre_train, logger, reverse)
            logger.info("Epoch %d Dev perplexity %.3f words %d" % (epoch + 1, ppl, word_cnt))
            logger.info("Epoch {}, end time: {}".format(epoch + 1, datetime.datetime.now().strftime("%H:%M")))
            lr_updater.update(ppl)
        saver.save(session, model_path+'save.ckpt')
            
elif args.mode == 'test':
    ner_train_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/data/lm/train/'
    ner_dev_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/data/lm/dev/'
    ner_test_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/data/lm/test/'
    if args.use_model == 'fw_model':
        temp_ner_train_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/data/lm/train_fw/'
        temp_ner_dev_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/data/lm/dev_fw/'
        temp_ner_test_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/data/lm/test_fw/'
    else:
        temp_ner_train_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/data/lm/train_bw/'
        temp_ner_dev_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/data/lm/dev_bw/'
        temp_ner_test_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/data/lm/test_bw/'

    #deal with ner train data
    #/data/lm/train/目录下有一个txt文件不在循环中
    num = len(os.listdir(ner_train_path))-1
    for i in range(num):
        ner_train_path_ = ner_train_path + 'train_'+str(i)+'.txt'
        save_ner_train_path = temp_ner_train_path + 'train_'+str(i)+'.npz'
        pre_test = PreData(vocab=vocab_words, test_path=ner_train_path_)
        config = Config()
        lr_updater = LearningRateUpdater(config.learning_rate, config.decay, config.decay_when)
        graph = tf.Graph()
        with graph.as_default():
            testm = LM(config=config, mode="Test", vocab_size=vocab_size, embeddings=embeddings, reuse=tf.AUTO_REUSE)
        with tf.Session(graph=graph) as session:
            saver = tf.train.Saver()
            saver.restore(session, model_path+'save.ckpt')
            cost, word_cnt, ppl = test(session, testm, pre_test, logger, save_ner_train_path, reverse)
            logger.info("Test perplexity %.3f words %d" % (ppl, word_cnt))

    #deal with ner dev data
    #/data/lm/dev/目录下有一个txt文件不在循环中
    num = len(os.listdir(ner_dev_path))-1
    for i in range(num):
        ner_dev_path_ = ner_dev_path + 'dev_'+str(i)+'.txt'
        save_ner_dev_path = temp_ner_dev_path + 'dev_'+str(i)+'.npz'
        pre_test = PreData(vocab=vocab_words, test_path=ner_dev_path_)
        config = Config()
        lr_updater = LearningRateUpdater(config.learning_rate, config.decay, config.decay_when)
        graph = tf.Graph()
        with graph.as_default():
            testm = LM(config=config, mode="Test", vocab_size=vocab_size, embeddings=embeddings, reuse=tf.AUTO_REUSE)
        with tf.Session(graph=graph) as session:
            saver = tf.train.Saver()
            saver.restore(session, model_path+'save.ckpt')
            cost, word_cnt, ppl = test(session, testm, pre_test, logger, save_ner_dev_path, reverse)
            logger.info("Test perplexity %.3f words %d" % (ppl, word_cnt))

    #deal with ner test data
    #/data/lm/test/目录下有一个txt文件不在循环中
    num = len(os.listdir(ner_test_path))-1
    for i in range(num):
        ner_test_path_ = ner_test_path + 'test_'+str(i)+'.txt'
        save_ner_test_path = temp_ner_test_path + 'test_'+str(i)+'.npz'
        pre_test = PreData(vocab=vocab_words, test_path=ner_test_path_)
        config = Config()
        lr_updater = LearningRateUpdater(config.learning_rate, config.decay, config.decay_when)
        graph = tf.Graph()
        with graph.as_default():
            testm = LM(config=config, mode="Test", vocab_size=vocab_size, embeddings=embeddings, reuse=tf.AUTO_REUSE)
        with tf.Session(graph=graph) as session:
            saver = tf.train.Saver()
            saver.restore(session, model_path+'save.ckpt')
            cost, word_cnt, ppl = test(session, testm, pre_test, logger, save_ner_test_path, reverse)
            logger.info("Test perplexity %.3f words %d" % (ppl, word_cnt))































