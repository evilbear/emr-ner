#encoding:utf-8
import tensorflow as tf
import numpy as np
import os, time

class Config(object):
    epoch_num = 20
    train_batch_size = 64
    train_step_size = 20
    valid_batch_size = 64
    valid_step_size = 20
    test_batch_size = 1
    test_step_size = 1 
    lstm_layers = 2
    lstm_size = 300
    lstm_forget_bias = 0.0
    max_grad_norm = 0.25
    init_scale = 0.05
    learning_rate = 0.2
    decay = 0.5
    decay_when = 1.0
    dropout_prob = 0.5
    
class LM(object):
    def __init__(self, config, mode, vocab_size, embeddings, reuse=None):
        self.config = config
        self.mode = mode
        if mode == "Train":
            self.is_training = True
            self.batch_size = self.config.train_batch_size
            self.step_size = self.config.train_step_size
        elif mode == "Dev":
            self.is_training = False
            self.batch_size = self.config.valid_batch_size
            self.step_size = self.config.valid_step_size
        else:
            self.is_training = False
            self.batch_size = self.config.test_batch_size
            self.step_size = self.config.test_step_size 
            
        vocab_size = vocab_size
        lstm_size = config.lstm_size
        lstm_layers = config.lstm_layers
        lstm_forget_bias = config.lstm_forget_bias
        batch_size = self.batch_size
        step_size = self.step_size
        with tf.variable_scope("LM", reuse=reuse):
            self.inputs  = tf.placeholder(tf.int32, [batch_size, step_size])
            self.targets = tf.placeholder(tf.int32, [batch_size, step_size])
            self.initial_state = tf.placeholder(tf.float32, [batch_size, lstm_size * 2 * lstm_layers])
            
            word_embedding = tf.Variable(embeddings,dtype=tf.float32,trainable=True,name="word_embedding")
            inputs = tf.nn.embedding_lookup(word_embedding, self.inputs)
            if self.is_training and self.config.dropout_prob > 0:
                inputs = tf.nn.dropout(inputs, config.dropout_prob)
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_size, forget_bias=lstm_forget_bias, state_is_tuple=False)
            if self.is_training and config.dropout_prob > 0:
                lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, config.dropout_prob)
            cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * lstm_layers, state_is_tuple=False)
            inputs = tf.unstack(inputs, axis=1)
            outputs, self.final_state = tf.contrib.rnn.static_rnn(cell, inputs, initial_state=self.initial_state)
            self.output = tf.reshape(tf.concat(outputs, 1), [-1, lstm_size])
            labels = tf.reshape(self.targets, [-1])
            stdv = np.sqrt(1. / vocab_size)
            initializer = tf.random_uniform_initializer(-stdv * 0.8, stdv * 0.8)
            softmax_w = tf.get_variable("softmax_w", [lstm_size, vocab_size], initializer=initializer)
            softmax_b = tf.get_variable("softmax_b", [vocab_size], initializer=tf.constant_initializer(0.0, dtype=tf.float32))
            logits = tf.matmul(self.output, softmax_w) + softmax_b
            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
            training_losses = [self.loss]
            self.cost = tf.reduce_sum(self.loss)
            if self.is_training:
                self.lr = tf.Variable(0.0, trainable=False)
                optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
                tvars = tf.trainable_variables()
                grads = tf.gradients([tf.reduce_sum(loss) / batch_size for loss in training_losses], tvars)
                grads = [tf.clip_by_norm(grad, config.max_grad_norm) if grad is not None else grad for grad in grads]
                self.eval_op = optimizer.apply_gradients(zip(grads, tvars))
            else:
                self.eval_op = tf.no_op()
            
    def update_lr(self, session, learning_rate):
        if self.is_training:
            session.run(tf.assign(self.lr, learning_rate))
    
    def get_initial_state(self):
        return np.zeros([self.batch_size, self.config.lstm_size * 2 * self.config.lstm_layers], dtype=np.float32)
    
    
class LearningRateUpdater(object):
    def __init__(self, init_lr, decay_rate, decay_when):
        self._init_lr = init_lr
        self._decay_rate = decay_rate
        self._decay_when = decay_when
        self._current_lr = init_lr
        self._last_ppl = -1
        
    def get_lr(self):
        return self._current_lr
    
    def update(self, cur_ppl):
        if self._last_ppl > 0 and self._last_ppl - cur_ppl < self._decay_when:
            current_lr = self._current_lr * self._decay_rate
            self._current_lr = current_lr
        self._last_ppl = cur_ppl
        
def train(session, model, predata, logger, reverse=False, verbose=True):
    state = model.get_initial_state()
    total_cost = 0
    total_word_cnt = 0
    start_time = time.time()
    for batch in predata.yieldSpliceBatch(model.mode, model.batch_size, model.step_size, reverse):
        batch_id, batch_num, x, y, word_cnt = batch
        feed = {model.inputs: x, model.targets:y, model.initial_state: state}
        cost, state, _= session.run([model.cost, model.final_state, model.eval_op], feed)
        total_cost += cost
        total_word_cnt += word_cnt
        if verbose and (batch_id % max(10, batch_num//10)) == 0:
            ppl = np.exp(total_cost / total_word_cnt)
            wps = total_word_cnt / (time.time() - start_time)
            logger.info("  [%5d/%d]ppl: %.3f speed: %.0f wps costs %.3f words %d" % (batch_id, batch_num, ppl, wps, total_cost, total_word_cnt))
    return total_cost, total_word_cnt, np.exp(total_cost / total_word_cnt)
    
def test(session, model, predata, logger, save_path, reverse= False, verbose=True):
    state = model.get_initial_state()
    get_outputs=[]
    total_cost = 0
    total_word_cnt = 0
    start_time = time.time()
    for batch in predata.yieldSpliceBatch(model.mode, model.batch_size, model.step_size, reverse):
        batch_id, batch_num, x, y, word_cnt = batch
        feed = {model.inputs: x, model.targets:y, model.initial_state: state}
        cost, state, _ , get_output= session.run([model.cost, model.final_state, model.eval_op, model.output], feed)
        get_output = get_output.tolist()
        get_outputs += get_output
        total_cost += cost
        total_word_cnt += word_cnt
        if verbose and (batch_id % max(10, batch_num//10)) == 0:
            ppl = np.exp(total_cost / total_word_cnt)
            wps = total_word_cnt / (time.time() - start_time)
            logger.info("  [%5d/%d]ppl: %.3f speed: %.0f wps costs %.3f words %d" % (batch_id, batch_num, ppl, wps, total_cost, total_word_cnt))
    #反向LM
    if reverse:
        get_outputs.reverse()
    get_outputs = np.array(get_outputs)
    np.savez_compressed(save_path, embeddings=get_outputs)
    return total_cost, total_word_cnt, np.exp(total_cost / total_word_cnt)


