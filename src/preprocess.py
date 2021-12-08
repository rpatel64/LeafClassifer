'''
Author: Matthew Jones  and  Ravi Patel
Description: This code takes leaf images from a local location
and removes the background and color, making the image black
and white. The new pictures are saved in a given local location
'''

import os
from PIL import Image 


# Path is the location of the pictures.
path = str(os.getcwd()) + str("\data\Tulip Poplar")
# data path is the location where the processed pictures will be saved
datapath = str(os.getcwd()) + str("\data\TulipPoplarProcessed")

files = os.listdir(path)
datafiles = os.listdir(datapath)
print("Number of datafiles: ", len(files))
print("Number of processed datafiles: ", len(datafiles))

# Only pre process the image if it has not been processed.
if(len(datafiles) == 0):
    for f in files:
        imgPath = path+"\\"+str(f)
        imgfname = f.split('.')[0]
        print(imgPath)
        col = Image.open(imgPath)
        gray = col.convert('L')
        # convert the image to black and white
        bw = gray.point(lambda x: 0 if 0<x<150 else 255, '1')
        # save the image
        bw.save(datapath+"\\"+str(imgfname)+"_bw.png")
