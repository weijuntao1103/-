"""
Task [I] - Demonstrating how to compute the histogram of an image using 4 methods.
(1). numpy based
(2). matplotlib based
(3). opencv based
(4). do it myself (DIY)
check the precision, the time-consuming of these four methods and print the result.
"""


import numpy as np
import matplotlib.pyplot as plt
from cv2 import cv2

###
#please coding here for solving Task [I].
# plot the histogram
#file_name = 'C:/canoe.tiff'
#img_bgr   = cv2.imread(file_name)
#img_red  = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2GRAY)
#red_levels = np.arange(0,256,1)
#pixel_counts
#N_x = np.zeros_like(red_levels, dtype=np.ﬂoat)
#for (i,level) in enumerate(red_levels):
 #   N_x[i] = np.sum(img_red==level)
#plt.plot(red_levels, N_x)
#plt.xlabel('bins = 256 red levels')
#plt.ylabel('Counted pixel numbers in each level')
#plt.title('Red Histogram')
#plt.show()

#file_name = 'C:/canoe.tiff'
#img = cv2.imread(file_name)
#hist,bins = np.histogram(img[2,:,:].ravel(),256,[0,256])
#plt.plot(hist,color='r')
#plt.show()

#file_name = 'C:/canoe.tiff'
#img = cv2.imread(file_name)
#plt.figure()
#plt.hist(img[2,:,:].flatten(), 256, (0,256))
#plt.show()


#file_name = 'C:/canoe.tiff'
#img = cv2.imread(file_name)
#hist = cv2.calcHist([img], [2], None, [256], [0.0,256.0])
#plt.plot(hist,color='r')
#plt.show()



###





"""
Task [II]Refer to the link below to do the gaussian filtering on the input image.
Observe the effect of different @sigma on filtering the same image.
Try to figure out the gaussian kernel which the ndimage has used [Solution to this trial wins bonus].
https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html
"""

###
#please coding here for solving Task[II]

#import scipy
#from scipy import ndimage
#file_name = 'C:/canoe.tiff'
#imge= cv2.imread(file_name)
#cv2.imshow('fivestart.png',imge)
#imge1=ndimage.gaussian_filter(imge,sigma=10)
#plt.imshow(imge1)
#plt.show()







"""
Task [III] Check the following link to accomplish the generating of random images.
Measure the histogram of the generated image and compare it to the according gaussian curve
in the same figure.
"""

###
#please coding here for solving Task[III]

#mean = (2, 2)
#cov =np.eye(2)
#x = np.random.multivariate_normal(mean, cov, (500, 500), 'raise')
#plt.hist(x.ravel(), bins=256, color='r')
#plt.show()


