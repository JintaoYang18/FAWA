# -*- coding: UTF-8 -*-
import cv2
from PIL import Image
import numpy as np

def alphaBlending(img1,logo1,x,y,alpha1,beta1,rotation):
    # image
    pp = Image.open(img1)
    pp = pp.convert("RGBA") # jpg 2 png
    # pp = pp.resize((300, 300), Image.ANTIALIAS)
    pp.save("./tmp/tmp.png")
    lenth,width = pp.size
    lenth = min(lenth,width)
    width = min(lenth,width)
    lenth_L = int(lenth*beta1)
    width_L = int(width*beta1)

    # logo
    bb = Image.open(logo1)
    lenth_logo, width_logo = bb.size
    lenth_L0 = int(lenth_logo*beta1*1.0) # 1.0 is an experience value
    width_L0 = int(width_logo*beta1*1.0) # 1.0 is an experience value

    if rotation == 0:
        bb = bb.convert("RGBA")
        bb = bb.resize((lenth_L0, width_L0), Image.ANTIALIAS)
        bb.save("./tmp/tmp1.png")
    else:
        bb = bb.convert("RGBA")
        bb = bb.resize((lenth_L0, width_L0), Image.ANTIALIAS)
        bb = bb.rotate(rotation * 90)
        bb.save("./tmp/tmp1.png")

    img = cv2.imread("./tmp/tmp.png",-1)
    #print(img.shape)
    logo = cv2.imread("./tmp/tmp1.png",-1) #logo
    #print(logo.shape)
    y1, x1, z1 = logo.shape
    #print(img.shape,logo.shape,x,y,x1,y1)
    scr_channels = cv2.split(logo) # logo
    dstt_channels = cv2.split(img) # img
    b, g, r, a = cv2.split(logo) # logo

    # Limit boundary (299*299)
    if y+y1 >= 300:
        y = 300 -y1 -1
    if x + x1 >= 300:
        x = 300 - x1 - 1

    for i in range(3):
        #dstt_channels[i][80:160, 160:240] = dstt_channels[i][80:160, 160:240] * (255.0 - a)/ 255
        dstt_channels[i][y:(y+y1), x:(x+x1)] = (dstt_channels[i][y:(y+y1), x:(x+x1)] * (255.0-a*alpha1)/ 255 +
                                                np.array(scr_channels[i] *  (a*alpha1 ) / 255, dtype=np.uint8))
        # dstt_channels[i][0:80, 0:80] += np.array(scr_channels[i] * ( a  / 255), dtype=np.uint8)
    cv2.imwrite("./tmp/img_target.png", cv2.merge(dstt_channels))
    cc = Image.open("./tmp/img_target.png")
    cc = cc.convert("RGB")
    cc.save("./tmp/img_target.jpg")

# if __name__ == '__main__':
#     alphaBlending(ImgDir,LogoDir,40,40,Alpha,Beta,0)