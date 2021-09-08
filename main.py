# Auther : Jintao Yang

# STEP 1 :
# ==> Import packages
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from blending_FDE import alphaBlending
import scipy
from FDE_Pic import Population
import os
from time import gmtime, strftime

# STEP 2 :
# ==> Initial variables
filenames = "./tmp/ILSVRC2012_val_00004381.JPEG"
originals = Image.open(filenames)

# STEP 3 :
# ==> Loading target pre-trained model (VGG16/resnet18/...)
device_0 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu") # CUDA Device
model = models.vgg16(pretrained=False)
pthfile = './pre_trained_models/vgg16-397923af.pth'
model.load_state_dict(torch.load(pthfile))
model = model.to(device_0)

# ############ Classification Function
def classModel_1(file_dir):
    img_n = Image.open(file_dir)
    data = trans(img_n)
    data = data.unsqueeze(dim=0)
    data = data.to(device_0)
    # Starting EVAL
    model.eval()
    output = model(data)
    out_np = output.cpu().data[0]
    return out_np

trans = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
                ])

# STEP 3-a :
# ==> Prediction and plot image
"""
    def classModel(img_n):
        data = trans(img_n)
        data = data.unsqueeze(dim=0)
        data = data.to(device_0)
        model.eval()
        output = model(data)
        out_np = output.cpu().data[0]
        ind = int(np.argmax(out_np))
        print('Label:', ind, 'Confidence_Score：', out_np[ind].item())
        a5 = np.argsort(a=out_np, axis=-1)
        print(a5)
        print('TOP5 ——--------------------results：______________')
        print('Top1__Label：', a5[999].item(), 'Confidence_Score：', out_np[a5[999]].item())
        print('Top2__Label：', a5[998].item(), 'Confidence_Score：', out_np[a5[998]].item())
        print('Top3__Label：', a5[997].item(), 'Confidence_Score：', out_np[a5[997]].item())
        print('Top4__Label：', a5[996].item(), 'Confidence_Score：', out_np[a5[996]].item())
        print('Top5__Label：', a5[995].item(), 'Confidence_Score：', out_np[a5[995]].item(),'\r\n')
        return a5
    
    plt.subplot(111)
    plt.imshow(originals)
    plt.show()
    print('Clean Image')
    P_clean = classModel(originals)
"""

# STEP 4 :
# ==> Loading original logo image (source image)
ImgDir = filenames
LogoDir = "./logo/whu_logo_80_80.png"

# STEP 4-a :
# ==> Plot logo
"""
    wat_image = Image.open(ori_water_name)
    #wat_trans = wat_image.resize((80,80),Image.ANTIALIAS)
    #wat_trans.save("./wat_origin/logo/whu_logo_80_80.png")
    plt.subplot(122)
    plt.imshow(wat_image)#.astype(np.uint8))
    plt.show()
"""

# STEP 4-b :
# ==> Embedding logo
"""
Alpha = 0.30
Beta =  0.30
alphaBlending(ImgDir,LogoDir,40,40,Alpha,Beta,0)
"""

# STEP 5 :
# ==> Parameter Setting
filenames1 = "./tmp/img_target.jpg"
StartNum = 0 # start of attack image number in the .txt file list
EndNum = 2   # end of attack image number in the .txt file list

# Parameter of Differential Evolution
Max_g_rounds = 100
Max_p_sizes = 50
eta_factor = 0.8
co_factor = 0.75

# Boundary of Function Solutions x
# The order is [Alpha, Beta, Embedded X coordinate,Embedded Y coordinate, Number of rotations 90 degrees]
Min_range = [1, 1, 0, 0, 0] # variables lower bound
Max_range = [3, 3, 250, 250, 3] # variables upper bound
Dim=5 # dimension of your problem

Label_1 = 21 # random initial label number
attack_file = './new_2_clean_pics_resize300.txt' # target clean images list (dictionaries)
class_file = './class_2.txt' # target clean image corresponding labels list (numbers)
save_file = './attack_new_2.txt' # fawa_example images list (dictionaries)

ImgDir_1 = ImgDir # don't change if not clearly understand the code
LogoDir_1 = LogoDir # don't change if not clearly understand the code

# ############ Object Function
def obj_func(p):
    # [x1, x2, x3, x4, x5]
    # [Alpha, Beta, Embedded X coordinate,Embedded Y coordinate, Number of rotations 90 degrees]
    x1, x2, x3, x4, x5 = p
    # print(x1, x2, x3, x4)
    alphaBlending(ImgDir_1, LogoDir_1, int(x3), int(x4), x1 / 10.0, x2 / 10.0, x5)
    return classModel_1(filenames1)[Label_1].item()

# unequal constraint
# Limit the scope of the logo to the original image
constraint_ueq = [
    # [X[0], X[1], X[2], X[3], X[4]]
    # [Alpha, Beta, Embedded X coordinate,Embedded Y coordinate, Number of rotations 90 degrees]
    lambda x: int(x[2]) - int(298.0 - 30.0 * x[1]),
    lambda x: int(x[3]) - int(298.0 - 30.0 * x[1])
    # 298.0 is the width/height of target clean image (299*299), experience value
    # 30.0 is the original width/height of logo image, experience value
]

# ############  Function of Differential Evolution
def attack():
    # Format the predict/callback functions for the differential evolution algorithm
    p = Population(min_range=Min_range, max_range=Max_range, dim=Dim, eta=eta_factor, g_rounds=Max_g_rounds,
                   p_size=Max_p_sizes, object_func=obj_func, co=co_factor)
    best_x = p.evolution()
    v1, v2, v3, v4, v5 = best_x
    # print(v1 / 10.0, v2 / 10.0, int(v3), int(v4), int(v5))
    alphaBlending(ImgDir_1, LogoDir_1, int(v3), int(v4), v1 / 10.0, v2 / 10.0, int(v5))

# ############  Read txt file line by line
f = open(attack_file,"r")
att_img = f.read().splitlines()
f.close()

f = open(class_file,"r")
labels = f.read().splitlines()
f.close()

f = open(save_file,"r")
savefiles = f.read().splitlines()
f.close()

# STEP 5 :
# ==> Attack totally (EndNum - StartNum) images using for-loop
for i in range(StartNum,EndNum):
    Label_1 = int(labels[i])
    print(Label_1)
    ImgDir_1 = att_img[i]
    print(ImgDir_1)
    print('    Attacking No.',i,' images...')
    # Attack start
    print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    attack()
    print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    # Attack end

    # Save FAWA-Pics
    tmp1 = Image.open(filenames1)
    SaveDir = savefiles[i]
    tmp1.save(SaveDir)
    print(SaveDir, '\n')
print('----Attack 2 Images Ending----')