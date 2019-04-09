# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 22:52:28 2019

@author: Kunal
"""

import cv2
import matplotlib.pyplot as plt

img=cv2.imread("img1.jpg",0)

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


from sklearn.decomposition import PCA
pca = PCA(0.95)
img_new = pca.fit_transform(img)

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("Output/img_pca.jpg",img_new)