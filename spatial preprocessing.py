for spatial preprocessing
change every folder location to your where you save yout object frame each class in temporal preprocessing.py
then change the code to find matrix of optical flow in temporal preprocessing with :

for i in range(len(x)):
    count = 0
    y=os.listdir('train_testob/train2/cocacola/'+x[i]+'/')
    n=len(y)%10
    print(x[i])
    for j in range(0,len(y)-n,round((len(y)-n)/10)):
        img = Image.open('train_testob/train2/cocacola/'+x[i]+'/'+y[j])
        img = img.resize((180,320),Image.ANTIALIAS)
        arr = np.array(img)
        arr=arr/265
        a.append(arr)
    
    spatial= np.dstack((a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7],a[8],a[9]))
    all_spatial.append(spatial)
    spatial=[]
    a=[]
    b= np.array(all_spatial)
    print(b.shape)

# the code above will try to stacked 10 frames from all generated frames in one video

# Save matrix of frames
np.save('traincocacola320x180.npy',b) 
