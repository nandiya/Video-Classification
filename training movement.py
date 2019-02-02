from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
import tensorflow as tf
config = tf.ConfigProto( device_count = {'GPU': 0, 'CPU':3} ) 
sess = tf.Session(config=config) 
import numpy as np

from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

get_available_gpus()
import keras

keras.backend.set_session(sess)
keras.backend.get_session().run(tf.initialize_all_variables())
from keras.layers.normalization import BatchNormalization

# UPLOAD ALL THE TRAINING DATA
look = np.load('npyfile/train/newflowxy_look_320x360.npy')
take = np.load('npyfile/train/newflowxy_take_320x360.npy')
takeback = np.load('npyfile/train/newflowxy_takeback_320x360.npy')

ylook = []
for i in range(156):
    ylook.append(0)
ytake=[]
for i in range(140):
    ytake.append(1)
ytakeback=[]
for i in range(160):
    ytakeback.append(2)
x=[]

y=[]
for i in range(len(look)):
    x.append(look[i])
for j in range(len(take)):
    x.append(take[j])
for k in range(len(takeback)):
    x.append(takeback[k])


y=[]
for i in range(156):
    y.append(0)
for i in range(140):
    y.append(1)
for i in range(160):
    y.append(2)


x=np.array(x)
y=np.array(y)

y_cat = np_utils.to_categorical(y,3)
from keras.layers import LSTM, TimeDistributed, Reshape, SimpleRNN

# UPLOAD TESTING DATA
tlook = np.load('chshnpy/test2flowxy_look_320x360.npy')
ttake = np.load('chshnpy/test2flowxy_take_320x360.npy')
ttakeback = np.load('chshnpy/test2flowxy_takeback_320x360.npy')

xtest=[]

ytest=[]
for i in range(len(tlook)):
    xtest.append(tlook[i])
for j in range(len(ttake)):
    xtest.append(ttake[j])
for k in range(len(ttakeback)):
    xtest.append(ttakeback[k])
    
for i in range(65):
    ytest.append(0)
for i in range(65):
    ytest.append(1)
for i in range(65):
    ytest.append(2)


xt=np.array(xtest)
yt=np.array(ytest)
print(xt.shape)
print(yt.shape)

yt_cat = np_utils.to_categorical(yt,3)


# CNN+LSTM Model
model = Sequential()

model.add(Convolution2D(48, (7, 7), border_mode='same',data_format="channels_first",input_shape=(30,320, 360)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(96, 5, 5, border_mode='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(256, 3, 3, border_mode='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Convolution2D(512, 3, 3, border_mode='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Convolution2D(512, 3, 3, border_mode='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Reshape((1,122880)))
model.add(LSTM(1000, return_sequences = True))
model.add(Flatten())
model.add(Dense(524))
model.add(Activation('relu'))
model.add(Dropout(0.6))
model.add(Dense(522))
model.add(Activation('relu'))
model.add(Dropout(0.8))
model.add(Dense(3, activation='softmax'))
model.summary()



from keras.optimizers import Adam , Adamax,Adagrad,RMSprop
from keras import optimizers
#sgd = SGD(lr=0.000004, decay=1e-6, momentum=0.97, nesterov=True,clipnorm=0.1)

#Adam(lr=0.05, decay=1e-6)
c = RMSprop(lr=0.00008)
#optimizers = Adam(lr=0.0000003, decay=1e-6)
#Adamax(lr=0.004)

model.compile(loss='categorical_crossentropy',

              optimizer=c, # chooses any optimizer you want to use

              metrics=['accuracy'])

# Save model every 5 epcoc (it's optional, just comment it if you don't want to use it)
mc = keras.callbacks.ModelCheckpoint('weightw2/weightstrain--99{epoch:08d}.h5', 
                                     save_weights_only=True, period=5)

model.fit(x, y_cat,callbacks=[mc],batch_size=20, nb_epoch=200, verbose=1)

from sklearn.metrics import classification_report
score = model.evaluate(x,y_cat)
print ("accuracy: =" ,score[1])
print ("loss: =" ,score[0])

#Save model and weight
json_model1=model.to_json()

from keras.models import model_from_json
with open("temporal_model3.json", "w") as json_file:
    json_file.write(json_model1)




