import csv
import glob
import os
import os.path
from subprocess import call
import numpy as np
import cv2 
from PIL import Image,ImageEnhance
def get_subdirs(dir):
    "Get a list of immediate subdirectories"
    return next(os.walk(dir))[1]

# MATRIX OF OPTICAL FLOW FOR LOOK CLASS
x = get_subdirs('train_test\\train\\look')

os.listdir('train_test\\train\\look\\'+x[0])
optic_horz=[]
optic_verc=[]
all_op_horz=[]
all_op_verc=[]
all_op=[]

# Making matrix of optical flow for look class
for i in range(len(x)):
    count = 0
    y=os.listdir('train_test/train/look/'+x[i]+'/')
    
    img = Image.open('train_test/train/look/'+x[i]+'/'+y[0])
    contrast_enhancer = ImageEnhance.Contrast(img)
    pil_enhanced_image = contrast_enhancer.enhance(2)
    enhanced_image = np.asarray(pil_enhanced_image)
    r, g, b = cv2.split(enhanced_image)
    enhanced_image = cv2.merge([b, g, r])
    frame1 = cv2.resize(enhanced_image, (180,320))
    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    n=len(y)%30
    print(x[i])
    for j in range(1,len(y)-n,round((len(y)-n)/30)):
       
        count=count+1
        
       
        img_next = Image.open('train_test/train/look/'+x[i]+'/'+y[j])
        contrast_enhancer2 = ImageEnhance.Contrast(img_next)
        pil_enhanced_image2 = contrast_enhancer2.enhance(2)
        enhanced_image2 = np.asarray(pil_enhanced_image2)
       
        frame2 = cv2.resize(enhanced_image2, (180,320))
        img_nextcv = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prvs,img_nextcv, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        horz = cv2.normalize(flow[...,0], None, 0, 255, cv2.NORM_MINMAX)
        vert = cv2.normalize(flow[...,1], None, 0, 255, cv2.NORM_MINMAX)
        horz = horz.astype('uint8')
        vert = vert.astype('uint8')
        optic_horz.append(horz)
        optic_verc.append(vert)
      
        
        prvs = img_nextcv
    
    all_op_horz.append(optic_horz)
    all_op_verc.append(optic_verc)
    temp1 = np.array(optic_horz)
    temp2 = np.array(optic_verc)
    temp3 = np.dstack((optic_horz,optic_verc))
    all_op.append(temp3)
    optic_horz=[]
    optic_verc=[]

    a= np.array(all_op_horz)
    b=np.array(all_op_verc)
    c = np.array(all_op)
    print(a.shape)
    print(b.shape)
    print(c.shape)

# Save matrix of optical flow        
np.save('newflowxy_look_320x360.npy',c)

# --------

# MATRIX OF OPTICAL FLOW FOR TAKE CLASS
x = get_subdirs('train_test\\train\\take')

os.listdir('train_test\\train\\take\\'+x[0])

optic_horz1=[]
optic_verc1=[]
all_op_horz1=[]
all_op_verc1=[]
all_op1=[]

# MAKING MATRIX OF OPTICAL FLOW
for i in range(len(x)):
    count = 0
    y=os.listdir('train_test/train/take/'+x[i]+'/')
 
    img = Image.open('train_test/train/take/'+x[i]+'/'+y[0])
    contrast_enhancer = ImageEnhance.Contrast(img)
    pil_enhanced_image = contrast_enhancer.enhance(2)
    enhanced_image = np.asarray(pil_enhanced_image)
    r, g, b = cv2.split(enhanced_image)
    enhanced_image = cv2.merge([b, g, r])
    frame1 = cv2.resize(enhanced_image, (180,320))
    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    n=len(y)%30
    print(x[i])
    for j in range(1,len(y)-n,round((len(y)-n)/30)):
        #print(round((len(y)-n)/20))
        #j=j+round((len(y)-n)/20)
        count=count+1
        
        
        img_next = Image.open('train_test/train/take/'+x[i]+'/'+y[j])
        contrast_enhancer2 = ImageEnhance.Contrast(img_next)
        pil_enhanced_image2 = contrast_enhancer2.enhance(2)
        enhanced_image2 = np.asarray(pil_enhanced_image2)
        r, g, b = cv2.split(enhanced_image2)
        enhanced_image2 = cv2.merge([b, g, r])
        frame2 = cv2.resize(enhanced_image2, (180,320))
        img_nextcv = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prvs,img_nextcv, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        horz = cv2.normalize(flow[...,0], None, 0, 255, cv2.NORM_MINMAX)
        vert = cv2.normalize(flow[...,1], None, 0, 255, cv2.NORM_MINMAX)
        horz = horz.astype('uint8')
        vert = vert.astype('uint8')
        optic_horz1.append(horz)
        optic_verc1.append(vert)
        #print(type(horz))
        prvs = img_nextcv
    
    all_op_horz1.append(optic_horz1)
    all_op_verc1.append(optic_verc1)
    temp1 = np.array(optic_horz1)
    temp2 = np.array(optic_verc1)
    temp3 = np.dstack((optic_horz1,optic_verc1))
    all_op1.append(temp3)
    optic_horz1=[]
    optic_verc1=[]
    a= np.array(all_op_horz1)
    b=np.array(all_op_verc1)
    c=np.array(all_op1)
    print(a.shape)
    print(b.shape)
    print(c.shape)

# save matrix of optical flow
np.save('newflowxy_take_320x360.npy',c)

# --------------------

# MATRIX OF OPTICAL FLOW FOR TAKEBACK

x = get_subdirs('train_test\\train\\takeback')

os.listdir('train_test\\train\\takeback\\'+x[0])
optic_horz2=[]
optic_verc2=[]
all_op_horz2=[]
all_op_verc2=[]
all_op2=[]

# Make matrix of optical flow
for i in range(len(x)):
    count = 0
    y=os.listdir('train_test/train/takeback/'+x[i]+'/')
   
    img = Image.open('train_test/train/takeback/'+x[i]+'/'+y[0])
    contrast_enhancer = ImageEnhance.Contrast(img)
    pil_enhanced_image = contrast_enhancer.enhance(2)
    enhanced_image = np.asarray(pil_enhanced_image)
    r, g, b = cv2.split(enhanced_image)
    enhanced_image = cv2.merge([b, g, r])
    frame1 = cv2.resize(enhanced_image, (180,320))
    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    n=len(y)%30
    print(x[i])
    for j in range(1,len(y)-n,round((len(y)-n)/30)):
     
        count=count+1
        
       
        img_next = Image.open('train_test/train/takeback/'+x[i]+'/'+y[j])
        contrast_enhancer2 = ImageEnhance.Contrast(img_next)
        pil_enhanced_image2 = contrast_enhancer2.enhance(2)
        enhanced_image2 = np.asarray(pil_enhanced_image2)
        r, g, b = cv2.split(enhanced_image2)
        enhanced_image2 = cv2.merge([b, g, r])
        frame2 = cv2.resize(enhanced_image2, (180,320))
        img_nextcv = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prvs,img_nextcv, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        horz = cv2.normalize(flow[...,0], None, 0, 255, cv2.NORM_MINMAX)
        vert = cv2.normalize(flow[...,1], None, 0, 255, cv2.NORM_MINMAX)
        horz = horz.astype('uint8')
        vert = vert.astype('uint8')
        optic_horz2.append(horz)
        optic_verc2.append(vert)
        #print(type(horz))
        prvs = img_nextcv
    #cb =np.array(optic_horz)
    all_op_horz2.append(optic_horz2)
    all_op_verc2.append(optic_verc2)
    temp1 = np.array(optic_horz2)
    temp2 = np.array(optic_verc2)
    temp3 = np.dstack((optic_horz2,optic_verc2))
    all_op2.append(temp3)
    optic_horz2=[]
    optic_verc2=[]
    a= np.array(all_op_horz2)
    b=np.array(all_op_verc2)
    c=np.array(all_op2)
    print(a.shape)
    print(b.shape)
        
# save matrix of optical flow
np.save('newflowxy_takeback_320x360.npy',c)































