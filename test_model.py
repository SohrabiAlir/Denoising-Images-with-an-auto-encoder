from keras.models import load_model
import numpy as np
import cv2
from keras import backend as K
import os
import time
import random
from noisy_generator import noisy
model = load_model('D:/aemodel.h5')
#model.load_weights(saveDir + "AutoEncoder_Cifar10_Deep_weights.05-0.56-0.56.hdf5")
model.get_weights
inp = model.input                                           # input placeholder
outputs = [layer.output for layer in model.layers]          # all layer outputs
functors = [K.function([inp] + [0], [out]) for out in outputs]  # evaluation functions
cap = cv2.VideoCapture(0)
'''dir = 'New folder'
files = os.listdir(dir)'''
#cv2.destroyAllWindows()
dir = 'set/testmodel'

files = os.listdir(dir)
random.shuffle(files)
a = []







for file in files:
    #time.sleep(0.01)
    img = cv2.imread(os.path.join(dir, file))
    img = cv2.resize(img, (32, 32))
    cv2.imshow('Ground Truth', cv2.resize(img, (256, 256)))
    #img = noisy("poisson", img, summit = 1000)6
    img = noisy("gauss", img, var = 500)
    #noisyf(img, 8)
    cv2.imshow("Input(Noisy)", cv2.resize(img, (256, 256)))
    img = img / 255
    #img = cv2.reshape(np.array(img), (1, 250, 150, 3))
    pred = model.predict(np.array([img]))
    for i in range(0):
        pred = model.predict(np.array(pred))
    for i in range(0):
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        pred[0] = cv2.filter2D(pred[0], -1, kernel)
    pred[0] = np.array(pred[0].astype('float32'))
    cv2.imshow("Predict", cv2.resize(pred[0], (256, 256)))
    cv2.imwrite('preddd.jpeg', pred[0].astype('float32') * 255)
    cv2.imshow("asli", file)

    
'''
while True:
    ret, frame = cap.read()
    
    model_img = cv2.resize(frame, (256, 256))
    
    # model_img = model_img / 255.
    #model_img = noisy("poisson", model_img)
    
    pred = model.predict(np.array([model_img]))

    print(pred.shape)
    pred = cv2.resize(pred[0], (256, 256))
    p = ""

    #pred = np.clip(pred, 0, 255)
    cv2.imshow("predd", pred)

    '''
'''
    layer = 1
    for chanel in range(layer_outs[layer][0].shape[3]):
        img = layer_outs[layer][0][0, :, :, chanel]
        img = cv2.resize(img, (250, 150), interpolation=cv2.INTER_NEAREST)
        cv2.imshow('layer ' + str(layer) + ' - chanel ' + str(chanel), img)
    
    pred = model.predict(np.array([model_img]))
    print(pred)
    
    if pred[0][0] >= 0.9:
        cv2.putText(frame, 'Face', (10, 50), 1, 3, (0, 255, 0), 3)
    elif pred[0][0] <= 0.1:
        cv2.putText(frame, 'Motorbike', (10, 50), 1, 3, (0, 255, 0), 3)
    '''
'''
    cv2.imshow(":/", cv2.resize(model_img, (256, 256)))
    if cv2.waitKey(300) == 13:
        break
  '''  


cv2.destroyAllWindows()

