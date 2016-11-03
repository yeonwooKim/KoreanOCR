import numpy as np
import tensorflow as tf
from hangul_utils import join_jamos_char
from data import en_chset, ko_chset_cho, ko_chset_jung, ko_chset_jong
from chrecog.core import *

sess = tf.Session()

def get_session():
    return sess

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