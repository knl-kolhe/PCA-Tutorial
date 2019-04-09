# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 22:52:28 2019

@author: Kunal
"""

import cv2
import matplotlib.pyplot as plt

img=cv2.imread("img1.jpg",1)
img=cv2.resize(img,(1920,540))
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)


plt.imshow(img)
plt.show()