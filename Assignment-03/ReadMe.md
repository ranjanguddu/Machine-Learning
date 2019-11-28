# 1. Final Accuracy of base Model is: 
## 81.77

# 2. My Model:

** Note: "out" stands for: size of output and "R" stands for: Size of Reptive Field

model = Sequential()

model.add(SeparableConv2D(48, 3, 3, activation= 'relu', border_mode='same',depthwise_initializer ='he_uniform', input_shape=(32, 32, 3))) # out: 32*32*48 and  R:3
model.add(BatchNormalization())
model.add(Dropout(0.1))


model.add(SeparableConv2D(48, 3, 3, depthwise_initializer ='he_uniform', activation= 'relu')) # out: 30*30*48 and  R:5
model.add(BatchNormalization())
model.add(Dropout(0.1))


model.add(MaxPooling2D(pool_size=(2, 2))) #out: 15*15*48 and R:6
model.add(Dropout(0.1))

model.add(SeparableConv2D(96, 3, 3, depthwise_initializer ='he_uniform', border_mode='same', activation= 'relu')) #out: 15*15*96 and R:10
model.add(BatchNormalization())
model.add(Dropout(0.1))


model.add(SeparableConv2D(96, 3, 3, depthwise_initializer ='he_uniform', activation= 'relu')) #13*13*96 and R:14
model.add(BatchNormalization())
model.add(Dropout(0.1))


model.add(MaxPooling2D(pool_size=(2, 2))) # out: 6*6*96 and R:16
model.add(Dropout(0.1))

model.add(SeparableConv2D(192, 3, 3, depthwise_initializer ='he_uniform', activation= 'relu', border_mode='same')) #out: 6*6*192 and R:24
model.add(BatchNormalization())
model.add(Dropout(0.1))


model.add(SeparableConv2D(192, 3, 3, depthwise_initializer ='he_uniform', activation= 'relu')) #out: 4*4*192 and R:32
model.add(BatchNormalization())
model.add(Dropout(0.1))


model.add(MaxPooling2D(pool_size=(2, 2))) #out: 2*2*192 and R:36
model.add(Dropout(0.1))


model.add(SeparableConv2D(64, 2, 2, depthwise_initializer ='he_uniform', activation= 'relu', border_mode='same'))# out: 2*2*64 and R:44
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(SeparableConv2D(32, 2, 2, depthwise_initializer ='he_uniform', activation= 'relu')) #out: 1*1*32 and R:52
model.add(BatchNormalization())
model.add(Dropout(0.1))



model.add(Convolution2D(num_classes, 1, 1)) #out: 1*1*10 and R:52
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(Flatten())
model.add(Activation('softmax'))

from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
def scheduler(epoch, lr):
  return round(0.003 * 1/(1 + 0.319 * epoch), 10)

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.003), metrics=['accuracy'])

model_info = model.fit(train_features, train_labels, batch_size=32, epochs=50, verbose=1, validation_data=(test_features, test_labels), callbacks=[LearningRateScheduler(scheduler, verbose=1)])

# 3. My Model's 50 epoch logs:
Train on 50000 samples, validate on 10000 samples
Epoch 1/50

Epoch 00001: LearningRateScheduler setting learning rate to 0.003.
50000/50000 [==============================] - 60s 1ms/step - loss: 1.6235 - acc: 0.4219 - val_loss: 1.2651 - val_acc: 0.5545
Epoch 2/50

Epoch 00002: LearningRateScheduler setting learning rate to 0.0022744503.
50000/50000 [==============================] - 54s 1ms/step - loss: 1.2691 - acc: 0.5552 - val_loss: 1.1085 - val_acc: 0.6140
Epoch 3/50

Epoch 00003: LearningRateScheduler setting learning rate to 0.0018315018.
50000/50000 [==============================] - 53s 1ms/step - loss: 1.1266 - acc: 0.6094 - val_loss: 0.9137 - val_acc: 0.6873
Epoch 4/50

Epoch 00004: LearningRateScheduler setting learning rate to 0.0015329586.
50000/50000 [==============================] - 53s 1ms/step - loss: 1.0376 - acc: 0.6428 - val_loss: 0.9149 - val_acc: 0.6922
Epoch 5/50

Epoch 00005: LearningRateScheduler setting learning rate to 0.0013181019.
50000/50000 [==============================] - 53s 1ms/step - loss: 0.9778 - acc: 0.6628 - val_loss: 0.7773 - val_acc: 0.7405
Epoch 6/50

Epoch 00006: LearningRateScheduler setting learning rate to 0.0011560694.
50000/50000 [==============================] - 53s 1ms/step - loss: 0.9266 - acc: 0.6816 - val_loss: 0.7464 - val_acc: 0.7479
Epoch 7/50

Epoch 00007: LearningRateScheduler setting learning rate to 0.0010295127.
50000/50000 [==============================] - 53s 1ms/step - loss: 0.8833 - acc: 0.6984 - val_loss: 0.7309 - val_acc: 0.7550
Epoch 8/50

Epoch 00008: LearningRateScheduler setting learning rate to 0.0009279307.
50000/50000 [==============================] - 54s 1ms/step - loss: 0.8550 - acc: 0.7079 - val_loss: 0.6829 - val_acc: 0.7739
Epoch 9/50

Epoch 00009: LearningRateScheduler setting learning rate to 0.0008445946.
50000/50000 [==============================] - 53s 1ms/step - loss: 0.8236 - acc: 0.7194 - val_loss: 0.6590 - val_acc: 0.7836
Epoch 10/50

Epoch 00010: LearningRateScheduler setting learning rate to 0.0007749935.
50000/50000 [==============================] - 54s 1ms/step - loss: 0.8076 - acc: 0.7241 - val_loss: 0.6296 - val_acc: 0.7906
Epoch 11/50

Epoch 00011: LearningRateScheduler setting learning rate to 0.0007159905.
50000/50000 [==============================] - 53s 1ms/step - loss: 0.7871 - acc: 0.7316 - val_loss: 0.6262 - val_acc: 0.7941
Epoch 12/50

Epoch 00012: LearningRateScheduler setting learning rate to 0.000665336.
50000/50000 [==============================] - 53s 1ms/step - loss: 0.7684 - acc: 0.7374 - val_loss: 0.6308 - val_acc: 0.7873
Epoch 13/50

Epoch 00013: LearningRateScheduler setting learning rate to 0.0006213753.
50000/50000 [==============================] - 53s 1ms/step - loss: 0.7527 - acc: 0.7436 - val_loss: 0.6005 - val_acc: 0.8035
Epoch 14/50

Epoch 00014: LearningRateScheduler setting learning rate to 0.0005828638.
50000/50000 [==============================] - 53s 1ms/step - loss: 0.7366 - acc: 0.7532 - val_loss: 0.6045 - val_acc: 0.7999
Epoch 15/50

Epoch 00015: LearningRateScheduler setting learning rate to 0.0005488474.
50000/50000 [==============================] - 53s 1ms/step - loss: 0.7314 - acc: 0.7521 - val_loss: 0.5890 - val_acc: 0.8085
Epoch 16/50

Epoch 00016: LearningRateScheduler setting learning rate to 0.0005185825.
50000/50000 [==============================] - 53s 1ms/step - loss: 0.7188 - acc: 0.7546 - val_loss: 0.5769 - val_acc: 0.8108
Epoch 17/50

Epoch 00017: LearningRateScheduler setting learning rate to 0.000491481.
50000/50000 [==============================] - 53s 1ms/step - loss: 0.7140 - acc: 0.7578 - val_loss: 0.5852 - val_acc: 0.8080
Epoch 18/50

Epoch 00018: LearningRateScheduler setting learning rate to 0.0004670715.
50000/50000 [==============================] - 53s 1ms/step - loss: 0.7004 - acc: 0.7626 - val_loss: 0.5657 - val_acc: 0.8148
Epoch 19/50

Epoch 00019: LearningRateScheduler setting learning rate to 0.0004449718.
50000/50000 [==============================] - 53s 1ms/step - loss: 0.7006 - acc: 0.7634 - val_loss: 0.5756 - val_acc: 0.8106
Epoch 20/50

Epoch 00020: LearningRateScheduler setting learning rate to 0.000424869.
50000/50000 [==============================] - 53s 1ms/step - loss: 0.6855 - acc: 0.7672 - val_loss: 0.5742 - val_acc: 0.8125
Epoch 21/50

Epoch 00021: LearningRateScheduler setting learning rate to 0.0004065041.
50000/50000 [==============================] - 53s 1ms/step - loss: 0.6715 - acc: 0.7720 - val_loss: 0.5669 - val_acc: 0.8108
Epoch 22/50

Epoch 00022: LearningRateScheduler setting learning rate to 0.000389661.
50000/50000 [==============================] - 54s 1ms/step - loss: 0.6747 - acc: 0.7723 - val_loss: 0.5827 - val_acc: 0.8060
Epoch 23/50

Epoch 00023: LearningRateScheduler setting learning rate to 0.0003741581.
50000/50000 [==============================] - 53s 1ms/step - loss: 0.6652 - acc: 0.7747 - val_loss: 0.5671 - val_acc: 0.8146
Epoch 24/50

Epoch 00024: LearningRateScheduler setting learning rate to 0.0003598417.
50000/50000 [==============================] - 53s 1ms/step - loss: 0.6556 - acc: 0.7771 - val_loss: 0.5662 - val_acc: 0.8137
Epoch 25/50

Epoch 00025: LearningRateScheduler setting learning rate to 0.0003465804.
50000/50000 [==============================] - 53s 1ms/step - loss: 0.6560 - acc: 0.7766 - val_loss: 0.5674 - val_acc: 0.8112
Epoch 26/50

Epoch 00026: LearningRateScheduler setting learning rate to 0.0003342618.
50000/50000 [==============================] - 53s 1ms/step - loss: 0.6485 - acc: 0.7782 - val_loss: 0.5509 - val_acc: 0.8139
Epoch 27/50

Epoch 00027: LearningRateScheduler setting learning rate to 0.0003227889.
50000/50000 [==============================] - 54s 1ms/step - loss: 0.6507 - acc: 0.7792 - val_loss: 0.5495 - val_acc: 0.8188
Epoch 28/50

Epoch 00028: LearningRateScheduler setting learning rate to 0.0003120774.
50000/50000 [==============================] - 53s 1ms/step - loss: 0.6385 - acc: 0.7846 - val_loss: 0.5607 - val_acc: 0.8164
Epoch 29/50

Epoch 00029: LearningRateScheduler setting learning rate to 0.000302054.
50000/50000 [==============================] - 53s 1ms/step - loss: 0.6339 - acc: 0.7839 - val_loss: 0.5435 - val_acc: 0.8238
Epoch 30/50

Epoch 00030: LearningRateScheduler setting learning rate to 0.0002926544.
50000/50000 [==============================] - 53s 1ms/step - loss: 0.6296 - acc: 0.7863 - val_loss: 0.5540 - val_acc: 0.8155
Epoch 31/50

Epoch 00031: LearningRateScheduler setting learning rate to 0.0002838221.
50000/50000 [==============================] - 54s 1ms/step - loss: 0.6181 - acc: 0.7903 - val_loss: 0.5355 - val_acc: 0.8219
Epoch 32/50

Epoch 00032: LearningRateScheduler setting learning rate to 0.0002755074.
50000/50000 [==============================] - 54s 1ms/step - loss: 0.6219 - acc: 0.7882 - val_loss: 0.5411 - val_acc: 0.8243
Epoch 33/50

Epoch 00033: LearningRateScheduler setting learning rate to 0.000267666.
50000/50000 [==============================] - 54s 1ms/step - loss: 0.6225 - acc: 0.7886 - val_loss: 0.5396 - val_acc: 0.8216
Epoch 34/50

Epoch 00034: LearningRateScheduler setting learning rate to 0.0002602585.
50000/50000 [==============================] - 54s 1ms/step - loss: 0.6130 - acc: 0.7906 - val_loss: 0.5368 - val_acc: 0.8226
Epoch 35/50

Epoch 00035: LearningRateScheduler setting learning rate to 0.00025325.
50000/50000 [==============================] - 54s 1ms/step - loss: 0.6057 - acc: 0.7930 - val_loss: 0.5399 - val_acc: 0.8232
Epoch 36/50

Epoch 00036: LearningRateScheduler setting learning rate to 0.0002466091.
50000/50000 [==============================] - 54s 1ms/step - loss: 0.6047 - acc: 0.7942 - val_loss: 0.5431 - val_acc: 0.8209
Epoch 37/50

Epoch 00037: LearningRateScheduler setting learning rate to 0.0002403076.
50000/50000 [==============================] - 54s 1ms/step - loss: 0.6083 - acc: 0.7928 - val_loss: 0.5342 - val_acc: 0.8262
Epoch 38/50

Epoch 00038: LearningRateScheduler setting learning rate to 0.0002343201.
50000/50000 [==============================] - 54s 1ms/step - loss: 0.6028 - acc: 0.7932 - val_loss: 0.5388 - val_acc: 0.8256
Epoch 39/50

Epoch 00039: LearningRateScheduler setting learning rate to 0.0002286237.
50000/50000 [==============================] - 54s 1ms/step - loss: 0.5913 - acc: 0.7984 - val_loss: 0.5438 - val_acc: 0.8217
Epoch 40/50

Epoch 00040: LearningRateScheduler setting learning rate to 0.0002231977.
50000/50000 [==============================] - 54s 1ms/step - loss: 0.5969 - acc: 0.7961 - val_loss: 0.5385 - val_acc: 0.8248
Epoch 41/50

Epoch 00041: LearningRateScheduler setting learning rate to 0.0002180233.
50000/50000 [==============================] - 54s 1ms/step - loss: 0.5967 - acc: 0.7975 - val_loss: 0.5351 - val_acc: 0.8259
Epoch 42/50

Epoch 00042: LearningRateScheduler setting learning rate to 0.0002130833.
50000/50000 [==============================] - 54s 1ms/step - loss: 0.5924 - acc: 0.7990 - val_loss: 0.5351 - val_acc: 0.8242
Epoch 43/50

Epoch 00043: LearningRateScheduler setting learning rate to 0.0002083623.
50000/50000 [==============================] - 54s 1ms/step - loss: 0.5877 - acc: 0.7997 - val_loss: 0.5284 - val_acc: 0.8275
Epoch 44/50

Epoch 00044: LearningRateScheduler setting learning rate to 0.0002038459.
50000/50000 [==============================] - 54s 1ms/step - loss: 0.5818 - acc: 0.8018 - val_loss: 0.5341 - val_acc: 0.8262
Epoch 45/50

Epoch 00045: LearningRateScheduler setting learning rate to 0.0001995211.
50000/50000 [==============================] - 54s 1ms/step - loss: 0.5854 - acc: 0.8007 - val_loss: 0.5343 - val_acc: 0.8238
Epoch 46/50

Epoch 00046: LearningRateScheduler setting learning rate to 0.0001953761.
50000/50000 [==============================] - 54s 1ms/step - loss: 0.5826 - acc: 0.8027 - val_loss: 0.5431 - val_acc: 0.8219
Epoch 47/50

Epoch 00047: LearningRateScheduler setting learning rate to 0.0001913998.
50000/50000 [==============================] - 54s 1ms/step - loss: 0.5795 - acc: 0.8012 - val_loss: 0.5319 - val_acc: 0.8248
Epoch 48/50

Epoch 00048: LearningRateScheduler setting learning rate to 0.0001875821.
50000/50000 [==============================] - 54s 1ms/step - loss: 0.5751 - acc: 0.8042 - val_loss: 0.5342 - val_acc: 0.8241
Epoch 49/50

Epoch 00049: LearningRateScheduler setting learning rate to 0.0001839137.
50000/50000 [==============================] - 54s 1ms/step - loss: 0.5757 - acc: 0.8050 - val_loss: 0.5275 - val_acc: 0.8242
Epoch 50/50

Epoch 00050: LearningRateScheduler setting learning rate to 0.000180386.
50000/50000 [==============================] - 54s 1ms/step - loss: 0.5758 - acc: 0.8034 - val_loss: 0.5320 - val_acc: 0.8269
