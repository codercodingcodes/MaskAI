import os
import cv2
import numpy as np


directory = r"D:\Programming\Data\2020\dataset"
master_data = []
master_label = []
test_len = 0.3
mask = 1
no_mask = 0
def translate (dir):


    width = 100
    height = 100
    image = cv2.imread(dir)
    print(dir)

    resized_img = cv2.resize(image,(width,height))
    greyscale = cv2.cvtColor(resized_img,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(greyscale, (5, 5), 0)


    return blur/255
count = 0
for folder in os.listdir(directory):
    print(folder)
    for type in os.listdir(directory+"\\"+folder):
        print(type)
        for file in os.listdir(directory+"\\"+folder+"\\"+type):
            count += 1
            image = translate(directory+"\\"+folder+"\\"+type+"\\"+file)
            master_data.append(image)
            if type == "WithMask":
                master_label.append(mask)
            elif type == 'WithoutMask':
                master_label.append(no_mask)
print(len(master_data))
print(len(master_label))

np.save("data",np.array(master_data))
np.save("label",np.array(master_label))


