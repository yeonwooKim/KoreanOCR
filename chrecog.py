# chrecog.py
# 32 X 32 numpy array를 받아
# 확률을 출력
import tensorflow as tf
import numpy as np
from hangul_utils import join_jamos_char

en_chset = []
en_chset.extend(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
en_chset.extend(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n",\
              "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"])
en_chset.extend(["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N",\
              "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"])
en_chset.extend(["(", ")", "'", "\"", ".", ",", ":", ";", "!", "?", "/", "@", "#", "$",\
              "%", "^", "&", "*", "[", "]", "{", "}", "<", ">", "~", "-"])

ko_chset_cho = ["ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]
ko_chset_jung = ["ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ", "ㅙ", "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ"]
ko_chset_jong = ["X", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]

#################
    
def flatten_cnn(layer):
    layer_shape = layer.get_shape().as_list()
    n_out = layer_shape[1] * layer_shape[2] * layer_shape[3]
    return tf.reshape(layer, [-1, n_out])

def build_nn(shape, X, name):
    n_before = int(X.get_shape()[1])
    W = tf.Variable(tf.truncated_normal([n_before, shape], stddev=0.1), name=name+"_W")
    b = tf.Variable(tf.constant(0.1, shape=[shape]), name=name+"_b")
    return tf.matmul(X, W)+b

def build_cnn(cnn_shape, patch_shape, X, name, stride=1):
    n_before = int(X.get_shape()[3])
    W = tf.Variable(tf.truncated_normal([patch_shape[0], patch_shape[1], n_before, cnn_shape], stddev=0.1),
                   name=name+"_W")
    b = tf.Variable(tf.constant(0.1, shape=[cnn_shape]), name=name+"_b")
    layer = tf.nn.relu(tf.nn.conv2d(X, W, strides=[1, stride, stride, 1], padding='SAME') + b)
    return layer

def max2d_pool(layer):
    return tf.nn.max_pool(layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def slice_label(tf_label, len_tuple):
    cur = 0
    sliced = []
    for l in len_tuple:
        sliced.append(tf.slice(tf_label, [0, cur], [-1, l]))
        cur += l
    return tuple(sliced)

################

tf.reset_default_graph()
X = tf.placeholder(tf.float32, [None, 32, 32])
keep_prob = tf.placeholder(tf.float32)

# Small inception model
# http://laonple.blog.me/220704822964
X_reshape = tf.reshape(X, [-1, 32, 32, 1])
cnn_1_5 = build_cnn(12, [5,5], X_reshape, "cnn_1_5")
cnn_1_3 = build_cnn(36, [3,3], X_reshape, "cnn_1_3")
cnn_1_concat = tf.concat(3, [cnn_1_5, cnn_1_3])
cnn_1_pool = max2d_pool(cnn_1_concat) # 16 * 16 * 48

cnn_2_5 = build_cnn(18, [5,5], cnn_1_pool, "cnn_2_5")
cnn_2_3 = build_cnn(48, [3,3], cnn_1_pool, "cnn_2_3")
cnn_2_1 = build_cnn(30, [1,1], cnn_1_pool, "cnn_2_1")
cnn_2_concat = tf.concat(3, [cnn_2_5, cnn_2_3, cnn_2_1])
cnn_2_pool = max2d_pool(cnn_2_concat) # 8 * 8 * 96

cnn_3_5_reduce = build_cnn(18, [1,1], cnn_2_pool, "cnn_3_5_reduce")
cnn_3_5 = build_cnn(36, [5,5], cnn_3_5_reduce, "cnn_3_5")
cnn_3_3_reduce = build_cnn(64, [1,1], cnn_2_pool, "cnn_3_3_reduce")
cnn_3_3 = build_cnn(96, [3,3], cnn_3_3_reduce, "cnn_3_3")
cnn_3_1 = build_cnn(60, [1,1], cnn_2_pool, "cnn_3_1")
cnn_3_concat = tf.concat(3, [cnn_3_5, cnn_3_3, cnn_3_1])
cnn_3_pool = max2d_pool(cnn_3_concat) # 4 * 4 * 192

dense_1 = tf.nn.relu(build_nn(1024, flatten_cnn(cnn_3_pool), "dense_1"))
dropout_1 = tf.nn.dropout(dense_1, keep_prob)

logit = build_nn(160, dropout_1, "logit")
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

sess = tf.Session()

#################

def init_session():
    sess.run(tf.initialize_all_variables())
    print("session initialized")
            
def save_ckpt(path):
    saver = tf.train.Saver(var_before_adam)
    saver.save(sess, path)
    print("ckpt saved")
    
def load_ckpt(path):
    saver = tf.train.Saver(var_before_adam)
    saver.restore(sess, path)
    print("ckpt loaded")

#################

class Prediction:
    def __init__(self, pred_cho, pred_jung, pred_jong, pred_en):
        self.pred_cho = pred_cho
        self.pred_jung = pred_jung
        self.pred_jong = pred_jong
        self.pred_en = pred_en
        self.invalid_cho = np.argmax(pred_cho) == len(ko_chset_cho)
        self.invalid_jung = np.argmax(pred_jung) == len(ko_chset_jung)
        self.invalid_jong = np.argmax(pred_jong) == len(ko_chset_jong)
        self.invalid_en = np.argmax(pred_en) == len(en_chset)
        self.invalid_ko = (self.invalid_cho and
                           (self.invalid_jung or self.invalid_jong)
                          ) or (self.invalid_jung and self.invalid_jong)
    
    def max_cho(self):
        return np.argmax(self.pred_cho)
    
    def max_jung(self):
        return np.argmax(self.pred_jung)
    
    def max_jong(self):
        return np.argmax(self.pred_jong)
    
    def max_en(self):
        return np.argmax(self.pred_en)

    def candidate(self):
        return get_candidate(self)
    
def get_candidate(pred):
    if pred.invalid_en and pred.invalid_ko:
        return ""
    if pred.invalid_ko:
        return en_chset[pred.max_en()]
    ## 한글 취급
    if pred.invalid_cho:
        pred.pred_cho[pred.max_cho()] = 0
    if pred.invalid_jung:
        pred.pred_jung[pred.max_jung()] = 0
    if pred.invalid_jong:
        pred.pred_jong[pred.max_jong()] = 0
    cho = ko_chset_cho[pred.max_cho()]
    jung = ko_chset_jung[pred.max_jung()]
    jong = ko_chset_jong[pred.max_jong()]
    if jong == 'X':
        jong = None
    return join_jamos_char(cho, jung, jong)

def get_pred_one(input_mat):
    pred_cho, pred_jung, pred_jong, pred_en = sess.run(
        (h_cho, h_jung, h_jong, h_en),
        feed_dict={X: [input_mat], keep_prob: 1})
    return Prediction(pred_cho[0], pred_jung[0], pred_jong[0], pred_en[0])

def get_pred_batch(input_mats):
    pred_cho, pred_jung, pred_jong, pred_en = sess.run(
        (h_cho, h_jung, h_jong, h_en),
        feed_dict={X: input_mats, keep_prob: 1})
    return [Prediction(pred_cho[i], pred_jung[i], pred_jong[i], pred_en[i]) for i in range(pred_cho.shape[0])]