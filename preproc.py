# preproc.py
# numpy array로 변환된 이미지를 받아
# 전처리를 수행
import numpy as np

# RGB 이미지를 흑백 이미지로 변환
def rgb2gray(rgb):
    if len(rgb.shape) is 3:
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
    else:
        print ("Not RGB!")
        return rgb

# app.py에서 호출
# 전처리가 완료된 numpy array를 반환
def process(img):
    return rgb2gray(img)