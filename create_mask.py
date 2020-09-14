import os
import cv2 
import numpy as np 

path_mask = ("C:\\Users\\bulzg\\Desktop\\road_detection\\mask\\")

os.chdir("C:\\Users\\bulzg\\Desktop\\road_detection\\result_lines")

def create_mask(img, name):
    mask = np.zeros_like(img)
    mask[:,:,1] = img[:,:,1]/255
    mask[:,:,1] = mask[:,:,1]*255
    cv2.imwrite(path_mask + name + ".jpg", mask)
    # print(mask)

for img_name in os.listdir():
    img = cv2.imread(img_name)
    create_mask(img, img_name.split(".")[0])

