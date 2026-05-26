import shutil
import numpy as np
import glob
import cv2

imgpaths = glob.glob('/database/iyj0121/dataset/ELRVD/SID/gt/*.png')
imgpaths.sort()
for i in range(len(imgpaths)):
    img = cv2.imread(imgpaths[i])
    cv2.imwrite('/database/iyj0121/dataset/ELRVD/SID/long_isp_png/SID_{}_clean.png'.format(i), img)  
