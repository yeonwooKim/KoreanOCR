import tensorflow as tf

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
#Y = tf.placeholder(tf.float32, [None, 160])
keep_prob = tf.placeholder(tf.float32)

#Y_cho, Y_jung, Y_jong, Y_en = slice_label(Y,
#                                         (len(ko_chset_cho)+1,
#                                         len(ko_chset_jung)+1,
#                                         len(ko_chset_jong)+1,
#                                         len(en_chset)+1))
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

#learning_rate = tf.placeholder(tf.float32)
#cost_cho = tf.nn.softmax_cross_entropy_with_logits(logit_cho, Y_cho)
#cost_jung = tf.nn.softmax_cross_entropy_with_logits(logit_jung, Y_jung)
#cost_jong = tf.nn.softmax_cross_entropy_with_logits(logit_jong, Y_jong)
#cost_en = tf.nn.softmax_cross_entropy_with_logits(logit_en, Y_en)
#cost = cost_cho + cost_jung * 1.5 + cost_jong * 0.5 + cost_en
#cost_mean = tf.reduce_mean(cost) # mean of batch set

var_before_adam = tf.all_variables()

#train = tf.train.AdamOptimizer(learning_rate).minimize(cost)

#correct_cho = tf.equal(tf.argmax(Y_cho,1), tf.argmax(h_cho,1))
#correct_jung = tf.equal(tf.argmax(Y_jung,1), tf.argmax(h_jung,1))
#correct_jong = tf.equal(tf.argmax(Y_jong,1), tf.argmax(h_jong,1))
#correct_two = tf.logical_or(tf.logical_and(correct_cho, tf.logical_or(correct_jung, correct_jong)),
#                           tf.logical_and(correct_jung, correct_jong))
#correct_ko = tf.logical_and(tf.logical_and(correct_cho, correct_jung), correct_jong)
#correct_en = tf.equal(tf.argmax(Y_en,1), tf.argmax(h_en,1))
#correct_all = tf.logical_and(correct_ko, correct_en)
#accuracy = tf.reduce_mean(tf.cast(correct_all, tf.float32))
#accuracy_two = tf.reduce_mean(tf.cast(correct_two, tf.float32))
#accuracy_cho = tf.reduce_mean(tf.cast(correct_cho, tf.float32))
#accuracy_jung = tf.reduce_mean(tf.cast(correct_jung, tf.float32))
#accuracy_jong = tf.reduce_mean(tf.cast(correct_jong, tf.float32))
#accuracy_ko = tf.reduce_mean(tf.cast(correct_ko, tf.float32))
#accuracy_en = tf.reduce_mean(tf.cast(correct_en, tf.float32))

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