import json
import gzip
import tarfile
import numpy as np
from scipy.ndimage import imread
from hangul_utils import check_syllable, split_syllable_char, split_syllables, join_jamos
import random

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

# Read training and test images from file
def get_image_index_from_file(indexf, dataf):
    index_data = []
    with gzip.open(indexf, 'rt') as arc:
        index_data.extend(json.load(arc))
        print("index loaded")
        arc.close()

    # Read-stream mode r|*
    with tarfile.open(dataf, "r|*") as tar:
        print("tar opened")
        img_data = []
        for i, member in enumerate(index_data):
            if i%10000 == 1:
                print("%2.0f%% complete (%d / %d)" % (i / len(index_data) * 100, i, len(index_data)))
            ti = tar.next()
            if ti.name != member['path']:
                print("ERROR: order doesn't match")
                break;
            f = tar.extractfile(ti)
            img_data.append(1 - (imread(f)/255))
        img = np.array(img_data)
        del img_data
        print("image loaded")
        tar.close()
    return (index_data, img)

def get_label(index_data):
    # len + 1 for one 'invalid' label
    label_ko_cho = np.zeros([len(index_data), len(ko_chset_cho)+1])
    label_ko_jung = np.zeros([len(index_data), len(ko_chset_jung)+1])
    label_ko_jong = np.zeros([len(index_data), len(ko_chset_jong)+1])
    label_en = np.zeros([len(index_data), len(en_chset)+1])
    for i, member in enumerate(index_data):
        target = member['target'] # Target Character
        # Is Hangeul?
        if (check_syllable(target)):
            splited = split_syllable_char(target)
            label_en[i][len(en_chset)] = 1
            label_ko_cho[i][ko_chset_cho.index(splited[0])] = 1
            label_ko_jung[i][ko_chset_jung.index(splited[1])] = 1
            if len(splited) < 3:
                label_ko_jong[i][0] = 1
            else:
                label_ko_jong[i][ko_chset_jong.index(splited[2])] = 1
        else :
            label_ko_cho[i][len(ko_chset_cho)] = 1
            label_ko_jung[i][len(ko_chset_jung)] = 1
            label_ko_jong[i][len(ko_chset_jong)] = 1
            label_en[i][en_chset.index(target)] = 1
            
    # Concatenate all labels
    label = np.concatenate((label_ko_cho, label_ko_jung, label_ko_jong, label_en), axis=1)
    print("label loaded")
    return label

def getIndex(l, indexes):
    return [l[i] for i in indexes]

def shuffle(n, *lists):
    perm = np.random.permutation(n)
    lists = list(lists)
    for i in range(len(lists)):
        if hasattr(lists[i], "shape"):
            lists[i] = lists[i][perm]
        else:
            lists[i] = getIndex(lists[i], perm)
    print("shuffled")
    return tuple(lists)

def get_all():
    index_data_en, img_en = get_image_index_from_file('data/en/index.json.gz', 'data/en/data.tar.gz')
    index_data_ko, img_ko = get_image_index_from_file('data/ko/index.json.gz', 'data/ko/data.tar.gz')
    index_data = index_data_en + index_data_ko
    img = np.concatenate((img_en, img_ko))
    del img_en
    del img_ko
    label = get_label(index_data)
    return shuffle(img.shape[0], index_data, img, label)