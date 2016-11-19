# preproc.py
# numpy array로 변환된 이미지를 받아
# 전처리를 수행
import numpy as np
import cv2

# app.py에서 호출
# 전처리가 완료된 numpy array를 반환
def process(img, threshold=True):
    if len(img.shape) > 2 :
        grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else :
        grayscale_img = img
        
    if not threshold:
        return grayscale_img
    
    return cv2.threshold(grayscale_img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]