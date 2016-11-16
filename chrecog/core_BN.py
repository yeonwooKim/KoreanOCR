# chrecog.py
# 32 X 32 numpy array를 받아
# 확률을 출력
import tensorflow as tf
from data import en_chset, ko_chset_cho, ko_chset_jung, ko_chset_jong

#################
    
def flatten_cnn(layer):
    layer_shape = layer.get_shape().as_list()
    n_out = layer_shape[1] * layer_shape[2] * layer_shape[3]
    return tf.reshape(layer, [-1, n_out])

def build_nn(shape, X, name):
    n_before = int(X.get_shape()[1])
    W = tf.Variable(tf.truncated_normal([n_before, shape], stddev=0.1), name=name+"_W")
    b = tf.Variable(tf.constant(0.1, shape=[shape]), name=name+"_b")
    layer = tf.matmul(X, W)+b
    return layer

def build_cnn(cnn_shape, patch_shape, X, name, stride=1):
    n_before = int(X.get_shape()[3])
    W = tf.Variable(tf.truncated_normal([patch_shape[0], patch_shape[1], n_before, cnn_shape], stddev=0.1),
                   name=name+"_W")
    b = tf.Variable(tf.constant(0.1, shape=[cnn_shape]), name=name+"_b")
    layer = tf.nn.conv2d(X, W, strides=[1, stride, stride, 1], padding='SAME') + b
    return layer

def build_cnn_relu(cnn_shape, patch_shape, X, name, stride=1):
    layer = tf.nn.relu(build_cnn(cnn_shape, patch_shape, X, name, stride))
    return layer

def max2d_pool(layer):
    return tf.nn.max_pool(layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# http://r2rt.com/implementing-batch-normalization-in-tensorflow.html
def batch_norm(layer, is_training, name, decay = 0.999, does_scale=True):
    layer_shape = layer.get_shape().as_list()
    depth_dim = len(layer_shape)-1
    depth = layer_shape[-1]
    if does_scale:
        scale = tf.Variable(tf.ones(depth), name=name+"_BN_s")
    else:
        scale = None
    beta = tf.Variable(tf.zeros(depth), name=name+"_BN_b")
    mov_mean = tf.Variable(tf.zeros(depth), trainable=False, name=name+"_BN_pm")
    mov_var = tf.Variable(tf.ones(depth), trainable=False, name=name+"_BN_pv")
    
    def use_batch_with_update_mov():
        batch_mean, batch_var = tf.nn.moments(layer,range(depth_dim))
        train_mean = tf.assign(mov_mean,
                               mov_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(mov_var,
                              mov_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(layer,
                batch_mean, batch_var, beta, scale, 0.001)
        
    def use_mov():
        return tf.nn.batch_normalization(layer,
            mov_mean, mov_var, beta, scale, 0.001)
    
    return tf.cond(is_training, use_batch_with_update_mov, use_mov)
    
def build_nn_bn(shape, X, is_training, name, decay = 0.999, does_scale=True):
    n_before = int(X.get_shape()[1])
    W = tf.Variable(tf.truncated_normal([n_before, shape], stddev=0.1), name=name+"_W")
    layer = tf.matmul(X, W)
    return batch_norm(layer, is_training, name, decay, does_scale)

def build_nn_bn_relu(shape, X, is_training, name, decay = 0.999):
    layer = tf.nn.relu(build_nn_bn(shape, X, is_training, name, decay, False))
    return layer
    
def build_cnn_bn(cnn_shape, patch_shape, X, is_training, name, stride=1, decay=0.999, does_scale=True):
    n_before = int(X.get_shape()[3])
    W = tf.Variable(tf.truncated_normal([patch_shape[0], patch_shape[1], n_before, cnn_shape], stddev=0.1),
                   name=name+"_W")
    layer = tf.nn.conv2d(X, W, strides=[1, stride, stride, 1], padding='SAME')
    return batch_norm(layer, is_training, name, decay, does_scale)

def build_cnn_bn_relu(cnn_shape, patch_shape, X, is_training, name, stride=1, decay=0.999):
    layer = tf.nn.relu(build_cnn_bn(cnn_shape, patch_shape, X, is_training, name, stride, decay, False))
    return layer

def slice_label(tf_label, len_tuple):
    cur = 0
    sliced = []
    for l in len_tuple:
        sliced.append(tf.slice(tf_label, [0, cur], [-1, l]))
        cur += l
    return tuple(sliced)

################

Y_size = len(ko_chset_cho)+len(ko_chset_jung)+len(ko_chset_jong)+len(en_chset)+4 

tf.reset_default_graph()
X = tf.placeholder(tf.float32, [None, 32, 32])
is_training = tf.placeholder(tf.bool)

# Small inception model
# http://laonple.blog.me/220704822964
X_reshape = tf.reshape(X, [-1, 32, 32, 1])
cnn_1_5 = build_cnn_bn_relu(12, [5,5], X_reshape, is_training, "cnn_1_5")
cnn_1_3 = build_cnn_bn_relu(36, [3,3], X_reshape, is_training, "cnn_1_3")
cnn_1_concat = tf.concat(3, [cnn_1_5, cnn_1_3])
cnn_1_pool = max2d_pool(cnn_1_concat) # 16 * 16 * 48

cnn_2_5 = build_cnn_bn_relu(18, [5,5], cnn_1_pool, is_training, "cnn_2_5")
cnn_2_3 = build_cnn_bn_relu(48, [3,3], cnn_1_pool, is_training, "cnn_2_3")
cnn_2_1 = build_cnn_bn_relu(30, [1,1], cnn_1_pool, is_training, "cnn_2_1")
cnn_2_concat = tf.concat(3, [cnn_2_5, cnn_2_3, cnn_2_1]) # 16 * 16 * 96

cnn_2_reduce = build_cnn_bn_relu(16, [1,1], cnn_2_concat, is_training, "cnn_2_reduce")

dense_1 = build_nn_bn_relu(1024, flatten_cnn(cnn_2_reduce), is_training, "dense_1")

logit = build_nn(Y_size, dense_1, "logit")
logit_cho, logit_jung, logit_jong, logit_en = slice_label(logit,
                                         (len(ko_chset_cho)+1,
                                         len(ko_chset_jung)+1,
                                         len(ko_chset_jong)+1,
                                         len(en_chset)+1))
h_cho = tf.nn.softmax(logit_cho)
h_jung = tf.nn.softmax(logit_jung)
h_jong = tf.nn.softmax(logit_jong)
h_en = tf.nn.softmax(logit_en)

var_before_adam = tf.all_variables()

#################

def init_session(sess):
    sess.run(tf.initialize_all_variables())
    print("session initialized")
            
def save_ckpt(sess, path):
    saver = tf.train.Saver(var_before_adam)
    saver.save(sess, path)
    print("ckpt saved to %s" % path)
    
def load_ckpt(sess, path):
    saver = tf.train.Saver(var_before_adam)
    saver.restore(sess, path)
    print("ckpt loaded from %s" % path)
