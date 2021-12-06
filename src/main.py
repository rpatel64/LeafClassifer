# import
import validation

import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import adam_v2
#from keras.optimizers import Adam
from tensorflow.python.keras.optimizer_v2 import adam

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.model_selection import train_test_split

import tensorflow as tf
import cv2
import os
import numpy as np
import matplotlib.pylab as plt

import seaborn as sns

import pandas as pd

#labels = ['RedMapleProcessed', 'BlackOakProcessed']

# Processed
labels = ['RedMapleProcessed', 'BlackOakProcessed', 'WillowOakProcessed', 'TulipPoplarProcessed', 'BlackWalnutProcessed', 'RedMulberryProcessed', 'SweetGumProcessed', 'BlackGumProcessed']

# Unprocessed
#labels = ['Red Maple', 'Black Oak', 'Willow Oak', 'Tulip Poplar', 'Black Walnut', 'Red Mulberry', 'Sweet Gum', 'Black Gum']


numEpoch = 1
lr = 0.005

d = len(labels)

#labels = ['Black Oak']

img_size = 224
def get_data(data_dir):
    data = [] 
    #data = data.astype(dtype=object)
    for label in labels: 
        path = os.path.join(data_dir, label)
        
        print(path)
        
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img))[...,::-1] #convert BGR to RGB format
                resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    return np.array(data, dtype=object)

path = "../data"
#blackOakPath = "~/Users/matthewjones/Desktop/Fall2021/COSC523/finalProj/LeafClassifer/data/Black Oak"
#redMaplePath = "~/Users/matthewjones/Desktop/Fall2021/COSC523/finalProj/LeafClassifer/data/Red Maple"

X = get_data(path)

#train = np.zeros(122)
#'WillowOakProcessed', 'TulipPoplarProcessed', 'BlackWalnutProcessed', 'RedMulberryProcessed', 'SweetGumProcessed', 'BlackGumProcessed'

l = []


for i in X:
    if(i[1] == 0):
        l.append("Red Maple")
    elif (i[1] == 1):
        l.append("Black Oak")
    elif (i[1] == 2):
        l.append("Willow Oak")
    elif (i[1] == 3):
        l.append('Tulip Poplar')
    elif (i[1] == 4):
        l.append('Black Walnut')
    elif (i[1] == 5):
        l.append('Red Mulberry')
    elif (i[1] == 6):
        l.append('Sweet Gum')
    elif (i[1] == 7):
        l.append('Black Gum')

df = pd.DataFrame({'keys': l, 'ids': l})

c = ['b', 'g', 'r', 'c', 'm', 'y', 'darkviolet', 'peru']



pd.value_counts(df['ids']).plot.bar(color = c)
plt.xticks(rotation=45)

plt.show()

'''
for i in X:
    if(i[1] == 0):
        l[0] = l[0] +1
    elif (i[1] == 1):
        l[1] = l[1] +1
    elif (i[1] == 2):
        l[2] = l[2] +1
    elif (i[1] == 3):
        l[3] = l[3] +1
    elif (i[1] == 4):
        l[4] = l[4] +1
    elif (i[1] == 5):
        l[5] = l[5] +1
    elif (i[1] == 6):
        l[6] = l[6] +1
    elif (i[1] == 7):
        l[7] = l[7] +1
'''


'''
plt.xticks(rotation=45)
colors = ['green', 'blue', 'lime'] 
        
c = ['b', 'g', 'r', 'c', 'm', 'y', 'darkviolet', 'peru']
n_bins = d

plt.hist(l, n_bins, density = True,
         histtype ='bar', 
         color = c, 
         label = c) 

plt.legend(prop ={'size': 10})
plt.show()
'''

#plt.hist(l, rwidth=.7, color=c)
#plt.tight_layout()

#plt.xticks(rotation=40)#, ha='right')       
#plt.show()
#sns.set_style('darkgrid')
#sns.countplot(l)
'''
df = pd.DataFrame(l)
ax = sns.countplot(x="Column", data=df)

ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()
plt.show()
'''


# shows random image
plt.figure(figsize = (5,5))
plt.imshow(X[1][0])
plt.title(labels[X[0][1]])


y = np.zeros(len(X))

train, val, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

x_train = []
y_train = []
x_val = []
y_val = []

for feature, label in train:
  x_train.append(feature)
  y_train.append(label)

for feature, label in val:
  x_val.append(feature)
  y_val.append(label)

# Normalize the data
x_train = np.array(x_train) / 255
x_val = np.array(x_val) / 255

x_train.reshape(-1, img_size, img_size, 1)
y_train = np.array(y_train)

x_val.reshape(-1, img_size, img_size, 1)
y_val = np.array(y_val)

y_train = tf.one_hot(y_train, depth=d)
y_val = tf.one_hot(y_val, depth=d)

#y_train = np.asarray(y_train).astype('float32').reshape((-1,1))
#y_val = np.asarray(y_val).astype('float32').reshape((-1,1))

'''
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range = 30,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.2, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip = True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(x_train)

model = Sequential()
model.add(Conv2D(32,3,padding="same", activation="relu", input_shape=(224,224,3)))
model.add(MaxPool2D())

model.add(Conv2D(32, 3, padding="same", activation="relu"))
model.add(MaxPool2D())

model.add(Conv2D(64, 3, padding="same", activation="relu"))
model.add(MaxPool2D())
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(128,activation="relu"))
model.add(Dense(2, activation="softmax"))

model.summary()
'''



base_model = tf.keras.applications.MobileNetV2(input_shape = (224, 224, 3), include_top = False, weights = "imagenet")

base_model.trainable = False

model = tf.keras.Sequential([base_model,
                                 tf.keras.layers.GlobalAveragePooling2D(),
                                 tf.keras.layers.Dropout(0.2),
                                 tf.keras.layers.Dense(d, activation="softmax")                                     
                                ])

base_learning_rate = lr
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

#model.add(Flatten())

history = model.fit(x_train,y_train,epochs = numEpoch , validation_data = (x_val, y_val))




'''
# original 0.000001
opt = tf.keras.optimizers.Adam(lr=lr)
model.compile(optimizer = opt , loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) , metrics = ['accuracy'])



# original epoch 500
history = model.fit(x_train,y_train,epochs = numEpoch , validation_data = (x_val, y_val))

'''
#validation.validate(history, x_val, y_val)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']


epochs_range = range(numEpoch)

plt.figure(figsize=(15, 15))
plt.subplot(2, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

#predictions = model.predict_classes(x_val)
#predictions = predictions.reshape(1,-1)[0]

#print(classification_report(y_val, predictions, target_names = ['Red Maple', 'Black Oak', 'Willow Oak', 'Tulip Poplar', 'Black Walnut', 'Red Mulberry', 'Sweet Gum', 'Black Gum']))

