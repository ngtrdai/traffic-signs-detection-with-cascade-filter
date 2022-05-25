from PIL import Image

import os, sys

path  = "./ahiih/"
dirs = os.listdir(path)

def resize_image():
    for item in dirs:
        if os.path.isfile(path + item):
            img = Image.open(path + item)
            f, e = os.path.splitext(path + item)
            imgResize = img.resize((50, 50), Image.ANTIALIAS)
            imgResize.save(f + ' resized.jpg', 'JPEG', quality=90)
            print("loading")

resize_image()