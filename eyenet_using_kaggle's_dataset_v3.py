

"""##Dataset

###Download data using kaggle client API
"""

from google.colab import files
files.upload()

!pip install -q kaggle

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/

!kaggle competitions download -c diabetic-retinopathy-detection

import shutil 
shutil.copy('./trainLabels.csv.zip', '/content/gdrive/My Drive/Grad Project/trainLabels.csv.zip')

"""###Extract data"""

cat train.zip* > train.zip

import shutil 
shutil.copy('./train.zip', '/content/gdrive/My Drive/Grad Project/train.zip')

import zipfile
with zipfile.ZipFile('./train.zip', 'r') as zip_ref:
    zip_ref.extractall()

!mv './train' '/content/gdrive/My Drive/Grad Project/train'/

import os
import pandas as pd

imgsList = os.listdir('./train')

print('Train size is', len(imgsList))
print(imgsList)

imgsList.sort()
imgsList.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
print(imgsList)

imgsFrame = pd.DataFrame(imgsList)
print(imgsFrame)

cat test.zip* > test.zip

import zipfile
with zipfile.ZipFile('./test.zip', 'r') as zip_ref:
    zip_ref.extractall()

"""###Data preprocessing

####Cropping and resizing
"""

'''
> The ImageFile module provides support functions for the image open and save functions.
> skimage.transform import resize: Resize image to match a certain size.
'''
import os
import sys
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from skimage import io
from skimage.transform import resize
import numpy as np


def create_directory(directory):
    '''
    Creates a new folder in the specified directory if the folder doesn't exist.
    INPUT
        directory: Folder to be created, called as "folder/".
    OUTPUT
        New folder in the current directory.
    '''
    if not os.path.exists(directory):
        os.makedirs(directory)


def crop_and_resize_images(path, new_path, cropx, cropy, img_size=256):
    '''
    Crops, resizes, and stores all images from a directory in a new directory.
    INPUT
        path: Path where the current, unscaled images are contained.
        new_path: Path to save the resized images.
        img_size: New size for the rescaled images.
    OUTPUT
        All images cropped, resized, and saved from the old folder to the new folder.
    '''
    
    '''
    > .DS_Store is a file that stores custom attributes of its containing folder, such as the position 
    of icons or the choice of a background image
    > // is a floor division operator
    > Size of the generated output image (rows, cols[, …][, dim]). If dim is not provided, the number of channels
      is preserved. In case the number of input channels does not equal the number of output channels a n-dimensional
      interpolation is applied.
    '''
    
    create_directory(new_path)
    dirs = [l for l in os.listdir(path) if l != '.DS_Store']                    
    total = 0

    for item in dirs:
        img = io.imread(path+item)
        y,x,channel = img.shape
        startx = x//2-(cropx//2)
        starty = y//2-(cropy//2)
        img = img[starty:starty+cropy,startx:startx+cropx]
        img = resize(img, (256,256))
        io.imsave(str(new_path + item), img)
        total += 1
        print("Saving: ", item, total)

crop_and_resize_images(path='./train/train/', new_path='./train-resized-256/', cropx=1800, cropy=1800, img_size=256)
crop_and_resize_images(path='./test/test/', new_path='./test-resized-256/', cropx=1800, cropy=1800, img_size=256)

"""####Removing black images"""

import time
import numpy as np
import pandas as pd
from PIL import Image


def find_black_images(file_path, df):
    """
    Creates a column of images that are not black (np.mean(img) != 0)
    INPUT
        file_path: file_path to the images to be analyzed.
        df: Pandas DataFrame that includes all labeled image names.
        column: column in DataFrame query is evaluated against.
    OUTPUT
        Column indicating if the photo is pitch black or not.
    """

    lst_imgs = [l for l in df['image']]
    return [1 if np.mean(np.array(Image.open(file_path + img))) == 0 else 0 for img in lst_imgs]


if __name__ == '__main__':
    start_time = time.time()
    trainLabels = pd.read_csv('../labels/trainLabels.csv')

    trainLabels['image'] = [i + '.jpeg' for i in trainLabels['image']]
    trainLabels['black'] = np.nan

    trainLabels['black'] = find_black_images('../data/train-resized-256/', trainLabels)
    trainLabels = trainLabels.loc[trainLabels['black'] == 0]
    trainLabels.to_csv('trainLabels_master.csv', index=False, header=True)

    print("Completed")
    print("--- %s seconds ---" % (time.time() - start_time))
    
    trainLabels2 = trainLabels.loc[trainLabels['black'] == 1]
    print(trainLabels2)

"""###Read labels"""

import zipfile
with zipfile.ZipFile('./trainLabels.csv.zip', 'r') as zip_ref:
    zip_ref.extractall()

import pandas as pd 
trainLabels = pd.read_csv("trainLabels.csv") 
print(trainLabels)

trainLabelsMergeDr = trainLabels
for i in range(35126): 
  if trainLabelsMergeDr.loc[(i, 'level')] >= 1:
    trainLabelsMergeDr.loc[(i, 'level')] = str(1)
  else:
    trainLabelsMergeDr.loc[(i, 'level')] = str(0)

trainLabelsMergeDr = trainLabelsMergeDr.drop("image", axis=1)   
trainLabelsMergeDr['image'] = imgsFrame[0]

print(trainLabelsMergeDr.head(10))

"""####Separate folder for each class"""

import os, shutil
dir = os.path.join("./train/0")
if not os.path.exists(dir):
  os.mkdir(dir)

dir = os.path.join("./train/1")
if not os.path.exists(dir):
  os.mkdir(dir)

for i in range(35126):
  if trainLabelsMergeDr.loc[(i, 'level')] == 0:
    shutil.move(os.path.join('./train/', trainLabelsMergeDr.loc[(i, 'image')]), "./train/0/")
  else:
    shutil.move(os.path.join('./train/', trainLabelsMergeDr.loc[(i, 'image')]), "./train/1/")

"""##Model

###Data loader
"""

from skimage.transform import resize
def crop_and_resize_images(img):
    cropx = 1800
    cropy = 1800
    img_size=256
   
    y,x,channel = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    img = img[starty:starty+cropy,startx:startx+cropx]
    img = resize(img, (256,256))
    return img

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.models import Sequential
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
import keras.backend as K

'''
kernel_size: Initial size of kernel
nb_filters: Initial number of filters
channels: Specify if the image is grayscale (1) or RGB (3)
nb_epoch: Number of epochs
batch_size: Batch size for the model
nb_classes: Number of classes for classification
'''

batch_size = 512
nb_classes = 2
nb_epoch = 30

img_rows, img_cols = 256, 256
channels = 3
nb_filters = 32
kernel_size = (8, 8)

from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(preprocessing_function = crop_and_resize_images, validation_split=0.1)

train_it = datagen.flow_from_dataframe(dataframe=trainLabelsMergeDr, directory="./train", x_col='image',
                                       y_col='level', subset="training", batch_size=32,
                                       shuffle=True, class_mode='binary', target_size=(img_rows, img_cols))

val_it = datagen.flow_from_dataframe(dataframe=trainLabelsMergeDr, directory="./train", x_col='image',
                                     y_col='level', subset="validation", batch_size=32,
                                     shuffle=True, class_mode='binary', target_size=(img_rows, img_cols))

model = Sequential()
model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1]), padding='valid', strides=1,
                   input_shape=(img_rows, img_cols, channels), activation="relu"))
model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1]), activation="relu"))
model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1]), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('sigmoid'))
model.add(Dropout(0.25))
model.add(Dense(1))
model.add(Activation('softmax'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

stop = EarlyStopping(monitor='val_acc', min_delta=0.001, patience=10, verbose=1, mode='auto')
checkpointer = ModelCheckpoint(filepath='/content/gdrive/My Drive/Grad Project/Eyenet_Arch.h5', verbose=1, save_best_only=False)
model.fit_generator(generator=train_it, validation_data=train_it, samples_per_epoch=35126, 
                    steps_per_epoch=1000, nb_epoch=2, verbose=1, callbacks=[stop, checkpointer])

#Finished epochs = 6

# load model
from keras.models import load_model
model = load_model('/content/gdrive/My Drive/Grad Project/Eyenet_Arch.h5')

stop = EarlyStopping(monitor='val_acc', min_delta=0.001, patience=10, verbose=1, mode='auto')
checkpointer = ModelCheckpoint(filepath='/content/gdrive/My Drive/Grad Project/Eyenet_Arch.h5', verbose=1, save_best_only=False)

model.fit_generator(generator=train_it, validation_data=train_it, samples_per_epoch=35126, 
                    steps_per_epoch=1000, nb_epoch=2, verbose=1, callbacks=[stop, checkpointer])

