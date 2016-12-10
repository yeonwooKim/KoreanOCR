import numpy as np
import cv2
import tensorflow as tf
from hangul_utils import join_jamos_char
from data import en_chset, ko_chset_cho, ko_chset_jung, ko_chset_jong
from detection.util import CHARTYPE
from data.buffer import *
#from chrecog.core_2IDR2 import *
from chrecog.core_BN2 import *

sess = tf.Session()
#load_ckpt(sess, "data/ckpt/161104_2IDR2.ckpt")
load_ckpt(sess, "data/ckpt/161117_BN2.ckpt")

def get_session():
    return sess

class Prediction:
    def __init__(self, pred_cho, pred_jung, pred_jong, pred_en):
        self.pred_cho = pred_cho
        self.pred_jung = pred_jung
        self.pred_jong = pred_jong
        self.pred_en = pred_en
        self.cal()

    def cal(self):
        self.max_en = np.argmax(self.pred_en)
        self.max_cho = np.argmax(self.pred_cho)
        self.max_jung = np.argmax(self.pred_jung)
        self.max_jong = np.argmax(self.pred_jong)
        self.max_en_prob = self.pred_en[self.max_en]
        self.max_cho_prob = self.pred_cho[self.max_cho]
        self.max_jung_prob = self.pred_jung[self.max_jung]
        self.max_jong_prob = self.pred_jong[self.max_jong]
        self.maxmin_ko_prob = min([self.max_cho_prob, self.max_jung_prob, self.max_jong_prob])
        self.invalid_cho = self.max_cho == len(ko_chset_cho)
        self.invalid_jung = self.max_jung == len(ko_chset_jung)
        self.invalid_jong = self.max_jong == len(ko_chset_jong)
        self.invalid_en = self.max_en == len(en_chset)
        self.invalid_ko = (self.invalid_cho and
                           (self.invalid_jung or self.invalid_jong)
                          ) or (self.invalid_jung and self.invalid_jong)
        self.sure, self.candidate = get_candidate(self)
    
def get_candidate(pred):
    if pred.invalid_en and pred.invalid_ko:
        return min(pred.max_en_prob, pred.maxmin_ko_prob), None
    if pred.invalid_ko:
        return pred.max_en_prob, en_chset[pred.max_en]
    ## 한글 취급
    if pred.invalid_cho:
        pred.pred_cho[pred.max_cho] = 0
        pred.cal()
    if pred.invalid_jung:
        pred.pred_jung[pred.max_jung] = 0
        pred.cal()
    if pred.invalid_jong:
        pred.pred_jong[pred.max_jong] = 0
        pred.cal()
    cho = ko_chset_cho[pred.max_cho]
    jung = ko_chset_jung[pred.max_jung]
    jong = ko_chset_jong[pred.max_jong]
    if jong == 'X':
        jong = None
    return pred.maxmin_ko_prob, join_jamos_char(cho, jung, jong)

def get_pred_one(input_mat):
    pred_cho, pred_jung, pred_jong, pred_en = sess.run(
        (h_cho, h_jung, h_jong, h_en),
        feed_dict={X: [input_mat], is_training: False})
    return Prediction(pred_cho[0], pred_jung[0], pred_jong[0], pred_en[0])

def get_pred_batch(input_mats):
    pred_cho, pred_jung, pred_jong, pred_en = sess.run(
        (h_cho, h_jung, h_jong, h_en),
        feed_dict={X: input_mats, is_training: False})
    return [Prediction(pred_cho[i], pred_jung[i], pred_jong[i], pred_en[i])
            for i in range(pred_cho.shape[0])]

def reshape_with_margin(img, size=32, pad=4):
    """Reshape narrow letters to 32 X 32
    without modifying ratio"""
    if img.shape[0] > img.shape[1]:
        dim = img.shape[0]
        margin = (dim - img.shape[1])//2
        margin_img = np.zeros([dim, margin])
        reshaped = np.c_[margin_img, img, margin_img]
    else:
        dim = img.shape[1]
        margin = (dim - img.shape[0])//2
        margin_img = np.zeros([margin, dim])
        reshaped = np.r_[margin_img, img, margin_img]
    reshaped = cv2.resize(reshaped, (size-pad*2, size-pad*2), interpolation = cv2.INTER_AREA)
    if pad > 0:
        padded = np.zeros([size, size])
        padded[pad:-pad, pad:-pad] = reshaped
    else:
        padded = reshaped
    return padded

def set_img(l, allc, allimg, c):
    if c.type == CHARTYPE.BLANK:
        return
    if c.pt[1] - c.pt[0] < 3:
        c.value = ""
        c.prob = 1.0
        return
    c.img = reshape_with_margin(l.img[:, c.pt[0]:c.pt[1]]) / 255
    allc.append(c)
    allimg.append(c.img)

def set_img_recur(l, allc, allimg, clist):
    for c in clist:
        set_img(l, allc, allimg, c)
        if len(c.children) > 0:
            set_img_recur(l, allc, allimg, c.children)

def get_pred(graphs):
    all_chars = []
    all_imgs = []
    all_pred = []
    for p in graphs:
        for l in p.lines:
            for c in l.chars:
                if len(c.children) > 0:
                    set_img_recur(l, all_chars, all_imgs, c.children)
                else:
                    set_img(l, all_chars, all_imgs, c)
    
    print("predicting %d cases..." % len(all_chars))
    all_imgs = np.asarray(all_imgs)
    imgbuf = ArrayBuffer(all_imgs, 0, -1)

    while (True):
        batch_x = imgbuf.read(200)
        if (batch_x is None):
            break
        all_pred.extend(get_pred_batch(batch_x))

    assert len(all_chars) == len(all_pred)

    for i in range(len(all_chars)):
        all_chars[i].pred = all_pred[i]

    return graphs