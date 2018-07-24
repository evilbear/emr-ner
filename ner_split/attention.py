import tensorflow as tf

def Mask(inputs, seq_len, state=False, mode='mul'):
    if seq_len == None:
        return inputs
    else:
        mask = tf.cast(tf.sequence_mask(seq_len), tf.float32)
        # mask = tf.reshape(mask, (-1, tf.shape(mask)[0]*tf.shape(mask)[1]))
        if state:
            doc2vec_state = tf.constant([[1.]], tf.float32)
            doc2vec_state = tf.tile(doc2vec_state, [tf.shape(seq_len)[0], 1])
            mask = tf.concat([mask,doc2vec_state],-1)
        for _ in range(len(inputs.shape)-2):
            mask = tf.expand_dims(mask, 2)
        if mode == 'mul':
            return inputs * mask
        if mode == 'add':
            return inputs - (1 - mask) * 1e12

def Dense(inputs, output_size, bias=True, state=False, seq_len=None):
    input_size = 600
    W = tf.Variable(tf.random_uniform([input_size, output_size], -0.05, 0.05))
    if bias:
        b = tf.Variable(tf.random_uniform([output_size], -0.05, 0.05))
    else:
        b = 0
    outputs = tf.matmul(tf.reshape(inputs, (-1, input_size)), W) + b
    outputs = tf.reshape(outputs, tf.concat([tf.shape(inputs)[:-1], [output_size]], 0))
    if seq_len != None:
        outputs = Mask(outputs, seq_len, state, 'mul')
    return outputs

def self_attention(inputs, size, seq_len=None, doc2vec=None, state=False):
    inputs_shape = tf.shape(inputs)
    if state:
        doc2vec = tf.expand_dims(doc2vec, 1)
        inputs = tf.concat([inputs,doc2vec],1)
    combined = Dense(inputs, 3*size, False, state)
    Q, K, V = tf.split(combined, [size, size, size], axis=-1)
    A = tf.matmul(Q, K, transpose_b=True) / tf.sqrt(float(size))
    A = tf.transpose(A, [0, 2, 1])
    A = Mask(A, seq_len, state, mode='add')
    A = tf.transpose(A, [0, 2, 1])
    A = tf.nn.softmax(A)
    O = tf.matmul(A, V)
    O = Mask(O, seq_len, state, 'mul')
    if state:
        O = O[:,:-1,:]
    O = tf.reshape(O, (-1, inputs_shape[1], size))
    return O

def multi_attention(inputs, heads, size_per_head, seq_len=None, doc2vec=None, state=False):
    inputs_shape = tf.shape(inputs)
    if state:
        doc2vec = tf.expand_dims(doc2vec, 1)
        inputs = tf.concat([inputs,doc2vec],1)
    combined = Dense(inputs, 3*heads*size_per_head, False, state)
    Q, K, V = tf.split(combined, [heads*size_per_head, heads*size_per_head, heads*size_per_head], axis=-1)
    Q = tf.reshape(Q, (-1, tf.shape(Q)[1], heads, size_per_head))
    Q = tf.transpose(Q, [0, 2, 1, 3])
    K = tf.reshape(K, (-1, tf.shape(K)[1], heads, size_per_head))
    K = tf.transpose(K, [0, 2, 1, 3])
    V = tf.reshape(V, (-1, tf.shape(V)[1], heads, size_per_head))
    V = tf.transpose(V, [0, 2, 1, 3])
    A = tf.matmul(Q, K, transpose_b=True) / tf.sqrt(float(size_per_head))
    A = tf.transpose(A, [0, 3, 2, 1])
    A = Mask(A, seq_len, state, mode='add')
    A = tf.transpose(A, [0, 3, 2, 1])
    A = tf.nn.softmax(A)
    O = tf.matmul(A, V)
    O = tf.transpose(O, [0, 2, 1, 3])
    O = tf.reshape(O, (-1, tf.shape(O)[1], heads * size_per_head))
    O = Mask(O, seq_len, state, 'mul')
    if state:
        O = O[:,:-1,:]
    O = tf.reshape(O, (-1, inputs_shape[1], heads * size_per_head))
    return O