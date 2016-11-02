import tensorflow as tf
from chrecog.core import *
from concurrent.futures import ThreadPoolExecutor
from data.buffer import *
from hangul_utils import join_jamos_char
import numpy as np

Y = tf.placeholder(tf.float32, [None, Y_size])
Y_cho, Y_jung, Y_jong, Y_en = slice_label(Y,
                                         (len(ko_chset_cho)+1,
                                         len(ko_chset_jung)+1,
                                         len(ko_chset_jong)+1,
                                         len(en_chset)+1))

learning_rate = tf.placeholder(tf.float32)
cost_cho = tf.nn.softmax_cross_entropy_with_logits(logit_cho, Y_cho)
cost_jung = tf.nn.softmax_cross_entropy_with_logits(logit_jung, Y_jung)
cost_jong = tf.nn.softmax_cross_entropy_with_logits(logit_jong, Y_jong)
cost_en = tf.nn.softmax_cross_entropy_with_logits(logit_en, Y_en)
cost = cost_cho + cost_jung * 1.5 + cost_jong * 0.5 + cost_en
cost_mean = tf.reduce_mean(cost) # mean of batch set

train = tf.train.AdamOptimizer(learning_rate).minimize(cost)

correct_cho = tf.equal(tf.argmax(Y_cho,1), tf.argmax(h_cho,1))
correct_jung = tf.equal(tf.argmax(Y_jung,1), tf.argmax(h_jung,1))
correct_jong = tf.equal(tf.argmax(Y_jong,1), tf.argmax(h_jong,1))
correct_two = tf.logical_or(tf.logical_and(correct_cho, tf.logical_or(correct_jung, correct_jong)),
                           tf.logical_and(correct_jung, correct_jong))
correct_ko = tf.logical_and(tf.logical_and(correct_cho, correct_jung), correct_jong)
correct_en = tf.equal(tf.argmax(Y_en,1), tf.argmax(h_en,1))
correct_all = tf.logical_and(correct_ko, correct_en)
accuracy = tf.reduce_mean(tf.cast(correct_all, tf.float32))
accuracy_two = tf.reduce_mean(tf.cast(correct_two, tf.float32))
accuracy_cho = tf.reduce_mean(tf.cast(correct_cho, tf.float32))
accuracy_jung = tf.reduce_mean(tf.cast(correct_jung, tf.float32))
accuracy_jong = tf.reduce_mean(tf.cast(correct_jong, tf.float32))
accuracy_ko = tf.reduce_mean(tf.cast(correct_ko, tf.float32))
accuracy_en = tf.reduce_mean(tf.cast(correct_en, tf.float32))

def slice_label_np(label, len_tuple):
    cur = 0
    sliced = []
    for l in len_tuple:
        sliced.append(label[cur:cur+l])
        cur += l
    return tuple(sliced)

def label_to_char(label):
    y_cho, y_jung, y_jong, y_en = slice_label_np(label, (len(ko_chset_cho)+1,
                                                        len(ko_chset_jung)+1,
                                                        len(ko_chset_jong)+1,
                                                        len(en_chset)+1))
    
    if np.argmax(y_en) < len(en_chset):
        return en_chset[np.argmax(y_en)]
        
    cho = ko_chset_cho[np.argmax(y_cho)]
    jung = ko_chset_jung[np.argmax(y_jung)]
    jong = ko_chset_jong[np.argmax(y_jong)]

    if jong == 'X':
        jong = None
        
    return join_jamos_char(cho, jung, jong)
    

# Submission function for thread pool executor
def _load_batch(imgbuf, labelbuf, batchsize):
    return (imgbuf.read(batchsize), labelbuf.read(batchsize))

# Prepare batch on another thread concurrently
def load_batch(executor, imgbuf, labelbuf, batchsize):
    return executor.submit(_load_batch, imgbuf, labelbuf, batchsize)    

def get_accuracy_no_batch(sess, imgset, labelset):
    return sess.run((accuracy, accuracy_cho, accuracy_jung, accuracy_jong, accuracy_two, accuracy_en),
                                        feed_dict={X:imgset, Y:labelset, keep_prob:1})

def get_accuracy(sess, imgbuf, labelbuf, batch=True, executor=None):
    imgbuf.seek(0)
    labelbuf.seek(0)
    
    if batch:
        temp_tuple = 0, 0, 0, 0, 0, 0
        i = 0
        batch_f = load_batch(executor, imgbuf, labelbuf, 100)
        while(True):
            batch_x, batch_y = batch_f.result()
            batch_f = load_batch(executor, imgbuf, labelbuf, 100)
            if batch_x is None:
                break
            temp_tuple = tuple([item1 + item2 * batch_x.shape[0] for item1, item2 in
                                zip(temp_tuple,
                                    get_accuracy_no_batch(sess, batch_x, batch_y))])
            i += batch_x.shape[0]
        tacc, tacc_cho, tacc_jung, tacc_jong, tacc_two, tacc_en = tuple([item / i for item in temp_tuple])
    else:
        tacc, tacc_cho, tacc_jung, tacc_jong, tacc_two, tacc_en = get_accuracy_no_batch(sess, imgbuf.read(), labelbuf.read())
    return tacc, tacc_cho, tacc_jung, tacc_jong, tacc_two, tacc_en
    
def print_accuracy(sess, imgbuf, labelbuf, batch=True, executor=None):
    tacc, tacc_cho, tacc_jung, tacc_jong, tacc_two, tacc_en = get_accuracy(sess, imgbuf, labelbuf, batch, executor)
    print ("overall accuracy = %.3f                          " % tacc)
    print ("two of three = %.3f" % tacc_two)
    print ("cho = %.3f" % tacc_cho)
    print ("jung = %.3f" % tacc_jung)
    print ("jong = %.3f" % tacc_jong)
    print ("en = %.3f" % tacc_en)
    
def print_accuracy_std(sess, imgbuf, labelbuf, batch=True):
    if batch == True:
        with ThreadPoolExecutor(max_workers=1) as executor:
            print_accuracy(sess, imgbuf, labelbuf, True, executor)
    else:
        print_accuracy(sess, imgbuf, labelbuf, False)
        
def error_check(sess, chset, pred_tf, label_tf, imgbuf, labelbuf):
    label_len = label_tf.get_shape()[1]
    n_error = np.zeros([label_len, label_len])
    n_all = np.zeros(label_len)
    new_chset = chset + ["inv"]
    
    batchsize = 100
    
    imgbuf.seek(0)
    labelbuf.seek(0)
    with ThreadPoolExecutor(max_workers=1) as executor:
        batch_f = load_batch(executor, imgbuf, labelbuf, batchsize)
        while(True):
            batch_x, batch_y = batch_f.result()
            batch_f = load_batch(executor, imgbuf, labelbuf, batchsize)
            if batch_x is None:
                break
            h, y = sess.run((pred_tf, label_tf), feed_dict={X:batch_x, Y:batch_y, keep_prob:1})
            for j in range(batch_x.shape[0]):
                n_all[np.argmax(y[j])] += 1
                if (np.argmax(h[j]) != np.argmax(y[j])):
                    n_error[np.argmax(y[j])][np.argmax(h[j])] += 1

    print ("Error rate")
    for i, ch in enumerate(new_chset):
        most_error = np.argmax(n_error[i])
        print ("%s : %2.0f%% (%4d / %4d)" %
               (ch, float(np.sum(n_error[i])) / n_all[i] * 100, np.sum(n_error[i]), n_all[i]), end="")
        if n_error[i][most_error] > 0:
            print ("%6d errors with %s" % (n_error[i][most_error], new_chset[most_error]))
        else:
            print ("")

class Trainer:
    def __init__(self, imgtar, label):
        self.trainimg = TarBuffer(imgtar, 15000, -1)
        self.trainlabel = ArrayBuffer(label, 15000, -1)
        self.cvimg = TarBuffer(imgtar, 12000, 3000)
        self.cvlabel = ArrayBuffer(label, 12000, 3000)
        self.testimg = TarBuffer(imgtar, 0, 12000)
        self.testlabel = ArrayBuffer(label, 0, 12000)
        self.sess = tf.Session()
    
    def init_session(self):
        self.sess.run(tf.initialize_all_variables())
        
    def save(self, path):
        save_ckpt(self.sess, path)
        
    def load(self, path):
        load_ckpt(self.sess, path)
        
    def print_test_accuracy(self):
        print_accuracy_std(self.sess, self.testimg, self.testlabel)
    
    def train(self, max_epoch=4, batchsize=200, is_console=True, lr_init = 0.003):
        trainimg = self.trainimg
        trainlabel = self.trainlabel
        cvimg = self.cvimg
        cvlabel = self.cvlabel
        testimg = self.testimg
        testlabel = self.testlabel
        sess = self.sess
        
        trainsize = trainimg.size
        batch_per_epoch = int(trainsize/batchsize)
        print ("Training %d, mini-batch %d * %d" % (trainsize, batchsize, batch_per_epoch))
        epoch = 0

        i = 0
        lr = lr_init

        with ThreadPoolExecutor(max_workers=1) as executor:
            trainimg.seek(0)
            trainlabel.seek(0)
            batch_f = load_batch(executor, trainimg, trainlabel, batchsize)
            while (epoch < max_epoch):
                if i % 200 == 0 :
                    tacc = get_accuracy(sess, cvimg, cvlabel, True, executor)[0]
                    print ("%6dth epoch : cv accuracy = %.3f                  " % (epoch, tacc))

                batch_x, batch_y = batch_f.result()
                if batch_x is None:
                    print_accuracy(sess, testimg, testlabel, True, executor)
                    trainimg.seek(0)
                    trainlabel.seek(0)
                    batch_f = load_batch(executor, trainimg, trainlabel, batchsize)
                    epoch += 1
                    continue

                ## Load batch must be after None check, because None check modifies buffer index
                batch_f = load_batch(executor, trainimg, trainlabel, batchsize)

                cur_cost = sess.run((train, cost_mean),
                                    feed_dict={X:batch_x, Y:batch_y, keep_prob:0.5, learning_rate:lr})[1]
                if(is_console):
                    print ("%dth... lr = %.2e, cost = %.2e\r" % (i, lr, cur_cost), end="")
                lr = lr * (1 - 0.0003)
                i += 1

            print("train complete--------------------------------")
            print("test accuracy ---")
            print_accuracy(sess, testimg, testlabel, True, executor)
            print("train accuracy ---")
            print_accuracy(sess, trainimg, trainlabel, True, executor)