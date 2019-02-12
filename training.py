import keras
from keras.models import load_model
# from keras.datasets import cifar10
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, concatenate, BatchNormalization, Activation
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
import os
import pickle
import numpy as np
import os
from noisy_generator import noisy
import random
import cv2
import matplotlib.pyplot as plt
from patch_generator import merge, devisor
def get_data_mat(dir):
    X = []
    y = []
    print(dir)
    files = os.listdir(dir)
    
    random.shuffle(files)
    for file in files:
        if file[-3:] == 'png' or file[-3:] == 'jpg':
            img = cv2.imread(os.path.join(dir, file))
            

            
            # cv2.imshow("ali", img)
            '''if file[:3] == 'fac':'''
            #     ****  X.append(noisy("poisson", img))   ****
            X.append(img)
    return np.array(X)
os.environ["KERAS_BACKEND"] = "tensorflow"
kerasBKED = os.environ["KERAS_BACKEND"] 
print(kerasBKED)
batch_size = 32
num_classes = 10
epochs = 100
'''
x, y = get_data_mat('set/train')
print('preparing training images...')
x_train = []
y_train = []

for i in range(len(x)):
    c = devisor(x[i])
    x_train = merge(x_train, c)
    y_train = merge(y_train, c)
    print(i + 1, "/", len(x))

x, y = get_data_mat('set/test')
print('preparing valifation images...')
x_test = []
y_test = []
print(1)

for i in range(len(x)):
    c = devisor(x[i])
    x_test = merge(x_test, c)
    y_test = merge(y_test, c)
    print(i + 1, "/", len(x))

for i in range(len(x_train)):
    cv2.imwrite('set/32x32/' + str(i) + '.png', x_train[i])
for i in range(len(x_test)):
    cv2.imwrite('set/32x32/' + str(i + len(x_train)) + '.png', x_test[i])
'''

x = get_data_mat('set/32x32')

x_train = x[:25000]
x_test = x[25000:]
print(1)
# normalize data

saveDir = "/opt/files/python/transfer/ae/"
if not os.path.isdir(saveDir):
    os.makedirs(saveDir)

x_train = np.array(x_train)
x_test = np.array(x_test)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
'''x_train /= 255
x_test /= 255'''
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
# divide x_test into validation and test
x_val = x_test
x_test = x_test
print("validation data: {0} \ntest data: {1}".format(x_val.shape, x_test.shape))
# adding noise to images
noise_factor = 0.1
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape) 
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape) 
x_val_noisy = x_val + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_val.shape) 

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)
x_val_noisy = np.clip(x_val_noisy, 0., 1.)
# definition to show original image and reconstructed image
def showOrigDec(orig, noise, num=10):
    import matplotlib.pyplot as plt
    n = num
    plt.figure(figsize=(20, 4))

    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i+1)
        plt.imshow(orig[i].reshape(32, 32, 3))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        # display original
        ax = plt.subplot(2, n, i +1 + n)
        plt.imshow(noise[i].reshape(32, 32, 3))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

# ae model



#Simple AutoEncoder

input_img = Input(shape = (32, 32, 3))

x = Conv2D(32, (3, 3), padding = 'same')(input_img)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((2, 2), padding = 'same')(x)

x = Conv2D(32, (3, 3), padding = 'same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
encoded = MaxPooling2D((2, 2), padding = 'same')(x)

x = Conv2D(32, (3, 3), padding = 'same')(encoded)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = UpSampling2D((2, 2))(x)

x = Conv2D(32, (3, 3), padding = 'same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = UpSampling2D((2, 2))(x)

x = Conv2D(3, (3, 3), padding = 'same')(x)
x = BatchNormalization()(x)

decoded = Activation('sigmoid')(x)

'''
#U-net Model
inputs = Input((32, 32, 3))
conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)
up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)
up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)
up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)
conv10 = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(conv9)
'''

model = Model(input_img, decoded)
model.compile(optimizer = 'adam', loss = 'MAE', metrics = ['acc'])

model.summary()

#es_cb = EarlyStopping(monitor = 'val_loss', patience = 2, verbose = 1, mode = 'auto')
chkpt = saveDir + 'AutoEncoder_Cifar10_denoise_weights.{epoch:02d}-{loss:.2f}-{val_loss:.2f}.hdf5'
cp_cb = ModelCheckpoint(filepath = chkpt, monitor = 'val_loss', verbose = 1, save_best_only = True, mode = 'auto')
history = model.fit(x_train_noisy,
                    x_train,
                    batch_size = batch_size,
                    epochs = 70,
                    verbose = 1,
                    validation_data = (x_val_noisy, x_val),
                    callbacks = [cp_cb],
                    shuffle = True)
model.save('D:/aemodel.h5')
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))


plt.plot(epochs, acc, 'bo', label = 'Training acc')
plt.plot(epochs, val_acc, 'b', label = 'Validation acc')
plt.title('training and validation accuracy')
plt.legend()

plt.figure()
plt.plot(epochs, loss, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
plt.title('training and validation loss')
plt.legend()

plt.show()
