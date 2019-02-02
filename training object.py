import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf


def get_session(gpu_fraction=0.333):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction,
                                allow_growth=True)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


ktf.set_session(get_session())
from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

get_available_gpus()
from keras.layers.normalization import BatchNormalization

# load training data
ch = np.load('object/trainchiki320x180.npy')
coca = np.load('object/traincocacola320x180.npy')
coco = np.load('object/traincocopie320x180.npy')

java = np.load('object/trainjavana320x180.npy')
nextar = np.load('object/trainnextar320x180.npy')
noth = np.load('object/trainnothing320x180.npy')

ore = np.load('object/trainoreo320x180.npy')
perm = np.load('object/trainpermen320x180.npy')
pokka = np.load('object/trainpokka320x180.npy')

sp = np.load('object/trainsponge320x180.npy')
sprt = np.load('object/trainsprite320x180.npy')

# Initialize output (Using the manual way since i use my old computer and keep getting hang everytime i use mapping
for i in range(len(ch)):
    x.append(ch[i])
for j in range(len(coca)):
    x.append(coca[j])
for k in range(len(coco)):
    x.append(coco[k])
for i in range(len(java)):
    x.append(java[i])
for j in range(len(nextar)):
    x.append(nextar[j])
for k in range(len(noth)):
    x.append(noth[k])
    
for i in range(len(ore)):
    x.append(ore[i])
for j in range(len(perm)):
    x.append(perm[j])
for k in range(len(pokka)):
    x.append(pokka[k])

for i in range(len(sp)):
    x.append(sp[i])
for j in range(len(sprt)):
    x.append(sprt[j])
for i in range(len(ch)):
    y.append(0)
for j in range(len(coca)):
    y.append(1)
for k in range(len(coco)):
    y.append(2)
for i in range(len(java)):
    y.append(3)
for j in range(len(nextar)):
    y.append(4)
for k in range(len(noth)):
    y.append(5)
    
for i in range(len(ore)):
    y.append(6)
for j in range(len(perm)):
    y.append(7)
for k in range(len(pokka)):
    y.append(8)

for i in range(len(sp)):
    y.append(9)
for j in range(len(sprt)):
    y.append(10)


#---- done output initialized




y_cat = np_utils.to_categorical(y,11)
xt=np.array(x)

# Actually, i should load testing data so that I can know the validation score.. but since the sum of video
# of object class is not balance.. soo i think it would be pointless anyway
# if your datasets have balance sum of each class, you should load it in order to know the validation score
# like I use in  training movement.py


# model CNN (I use so many convolutional process since My GPU resource is limited and the more convolutional process the smalle the parameter will be)
from keras.layers.convolutional import ZeroPadding2D


model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=(320,180,30)))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))


model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))


model.add(Flatten())
model.add(Dense(50042, activation='relu'))
model.add(Dropout(0.8))
model.add(Dense(50042, activation='relu'))
model.add(Dropout(0.8))
model.add(Dense(11, activation='softmax'))
model.summary()


from keras.optimizers import Adam , Adamax,Adagrad,RMSprop
from keras import optimizers
sgd = SGD(lr=0.04, decay=1e-6, momentum=0.97, nesterov=True,clipnorm=0.1)
import keras
#Adam(lr=0.05, decay=1e-6)
c = RMSprop(lr=0.00006)
optimizers = Adam(lr=0.00001, decay=1e-6)
#Adamax(lr=0.004)

model.compile(loss='categorical_crossentropy',

              optimizer=c, # choose any optimizer you want to use

              metrics=['accuracy'])

#Save weight every 5 epoch..(it's optional, just comment if you don't want to use it)
mc = keras.callbacks.ModelCheckpoint('weightob21/weightstrain4-baru-{epoch:08d}.h5', 
                                     save_weights_only=True, period=5)


model.fit(xt, y_cat,callbacks=[mc],batch_size=2, nb_epoch=200,verbose=1)


score = model.evaluate(xt,y_cat)
print ("accuracy: =" ,score[1])

# Save model and weight
json_model1=model.to_json()
model.save_weights('temporal_weightob2.h5')

with open("temporal_modelob2.json", "w") as json_file:
    json_file.write(json_model1)










