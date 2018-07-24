import tensorflow as tf

def attention(input1, input2, size):

    ss = tf.shape(input1)
    input1 = tf.reshape(input1, shape=[ss[0]*ss[1], size])
    input2 = tf.reshape(input2, shape=[ss[0]*ss[1], size])
    
    W_1 = tf.Variable(tf.random_normal([size, size], stddev=0.1))
    W_2 = tf.Variable(tf.random_normal([size, size], stddev=0.1))
    W_3 = tf.Variable(tf.random_normal([size, size], stddev=0.1))
    z = tf.sigmoid(tf.matmul(tf.tanh(tf.matmul(input1, W_2)+tf.matmul(input2, W_3)), W_1))
    output = input1 * z+ input2 * (1 - z)
    output = tf.reshape(output, shape=[ss[0], ss[1], size])
    
    return output
