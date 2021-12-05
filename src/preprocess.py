import os
from PIL import Image 

path = str(os.getcwd()) + str("\data\Red Maple")
datapath = str(os.getcwd()) + str("\data\RedMapleProcessed")

files = os.listdir(path)
datafiles = os.listdir(datapath)
print("Number of datafiles: ", len(files))
print("Number of processed datafiles: ", len(datafiles))
if(len(datafiles) == 0):
    for f in files:
        imgPath = path+"\\"+str(f)
        imgfname = f.split('.')[0]
        print(imgPath)
        # image_file = Image.open(imgPath) # open colour image
        # image_file = image_file.convert('1') # convert image to black and white
        # image_file.save('blackwhite.png')
        # img = Image.open(imgPath).convert('LA')
        # img.save('greyscale.png')

        col = Image.open(imgPath)
        gray = col.convert('L')
        bw = gray.point(lambda x: 0 if 0<x<150 else 255, '1')
        bw.save(datapath+"\\"+str(imgfname)+"_bw.png")