import cv2
import numpy as np


def augment_data(filelist):
    file_step = 0
    if filelist[1] != 0:
        initial_step = filelist[1]*8
    else:
        initial_step = filelist[1]+filelist[1]*8

    img1 = cv2.imread(filelist[0])
    name = filelist[2] + "img_" + str(initial_step+file_step).zfill(4) + ".png"
    #print(name)
    cv2.imwrite(name,img1)
    file_step +=1

    name = filelist[2] + "img_" + str(initial_step+file_step).zfill(4) + ".png"
    #print(name)
    img2 = np.flipud(img1)
    cv2.imwrite(name,img2)
    file_step += 1

    name = filelist[2] + "img_" + str(initial_step+file_step).zfill(4) + ".png"
    #print(name)
    img3 = np.fliplr(img1)
    cv2.imwrite(name,img3)
    file_step += 1

    name = filelist[2] + "img_" + str(initial_step+file_step).zfill(4) + ".png"
    #print(name)
    img4 = np.fliplr(img1)
    img4 = np.flipud(img4)
    cv2.imwrite(name,img4)
    file_step += 1

    name = filelist[2] + "img_" + str(initial_step+file_step).zfill(4) + ".png"
    #print(name)
    img5 = np.rot90(img1)
    cv2.imwrite(name,img5)
    file_step += 1

    name = filelist[2] + "img_" + str(initial_step+file_step).zfill(4) + ".png"
    #print(name)
    img6 = np.rot90(img1,-1)
    cv2.imwrite(name,img6)
    file_step += 1

    name = filelist[2] + "img_" + str(initial_step+file_step).zfill(4) + ".png"
    #print(name)
    img7 = np.rot90(img1)
    img7 = np.flipud(img7)
    cv2.imwrite(name,img7)
    file_step += 1

    name = filelist[2] + "img_" + str(initial_step+file_step).zfill(4) + ".png"
    #print(name)
    img8 = np.rot90(img1,-1)
    img8 = np.flipud(img8)
    cv2.imwrite(name,img8)

