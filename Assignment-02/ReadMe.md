# 1. Logs for 20 epochs

Train on 60000 samples, validate on 10000 samples
Epoch 1/20

Epoch 00001: LearningRateScheduler setting learning rate to 0.003.
60000/60000 [==============================] - 40s 673us/step - loss: 0.4723 - acc: 0.8536 - val_loss: 0.0591 - val_acc: 0.9874
Epoch 2/20

Epoch 00002: LearningRateScheduler setting learning rate to 0.0022744503.
60000/60000 [==============================] - 38s 635us/step - loss: 0.2548 - acc: 0.9149 - val_loss: 0.0436 - val_acc: 0.9900
Epoch 3/20

Epoch 00003: LearningRateScheduler setting learning rate to 0.0018315018.
60000/60000 [==============================] - 38s 639us/step - loss: 0.2109 - acc: 0.9286 - val_loss: 0.0323 - val_acc: 0.9903
Epoch 4/20

Epoch 00004: LearningRateScheduler setting learning rate to 0.0015329586.
60000/60000 [==============================] - 38s 629us/step - loss: 0.1880 - acc: 0.9340 - val_loss: 0.0279 - val_acc: 0.9922
Epoch 5/20

Epoch 00005: LearningRateScheduler setting learning rate to 0.0013181019.
60000/60000 [==============================] - 37s 625us/step - loss: 0.1666 - acc: 0.9406 - val_loss: 0.0241 - val_acc: 0.9930
Epoch 6/20

Epoch 00006: LearningRateScheduler setting learning rate to 0.0011560694.
60000/60000 [==============================] - 37s 622us/step - loss: 0.1624 - acc: 0.9405 - val_loss: 0.0234 - val_acc: 0.9938
Epoch 7/20

Epoch 00007: LearningRateScheduler setting learning rate to 0.0010295127.
60000/60000 [==============================] - 37s 610us/step - loss: 0.1569 - acc: 0.9422 - val_loss: 0.0223 - val_acc: 0.9931
Epoch 8/20

Epoch 00008: LearningRateScheduler setting learning rate to 0.0009279307.
60000/60000 [==============================] - 38s 629us/step - loss: 0.1483 - acc: 0.9440 - val_loss: 0.0231 - val_acc: 0.9935
Epoch 9/20

Epoch 00009: LearningRateScheduler setting learning rate to 0.0008445946.
60000/60000 [==============================] - 37s 625us/step - loss: 0.1447 - acc: 0.9447 - val_loss: 0.0193 - val_acc: 0.9949
Epoch 10/20

Epoch 00010: LearningRateScheduler setting learning rate to 0.0007749935.
60000/60000 [==============================] - 37s 625us/step - loss: 0.1418 - acc: 0.9453 - val_loss: 0.0201 - val_acc: 0.9943
Epoch 11/20

Epoch 00011: LearningRateScheduler setting learning rate to 0.0007159905.
60000/60000 [==============================] - 38s 626us/step - loss: 0.1391 - acc: 0.9468 - val_loss: 0.0199 - val_acc: 0.9939
Epoch 12/20

Epoch 00012: LearningRateScheduler setting learning rate to 0.000665336.
60000/60000 [==============================] - 38s 635us/step - loss: 0.1353 - acc: 0.9474 - val_loss: 0.0180 - val_acc: 0.9941
Epoch 13/20

Epoch 00013: LearningRateScheduler setting learning rate to 0.0006213753.
60000/60000 [==============================] - 38s 632us/step - loss: 0.1285 - acc: 0.9501 - val_loss: 0.0194 - val_acc: 0.9943
Epoch 14/20

Epoch 00014: LearningRateScheduler setting learning rate to 0.0005828638.
60000/60000 [==============================] - 38s 632us/step - loss: 0.1295 - acc: 0.9489 - val_loss: 0.0187 - val_acc: 0.9946
Epoch 15/20

Epoch 00015: LearningRateScheduler setting learning rate to 0.0005488474.
60000/60000 [==============================] - 38s 626us/step - loss: 0.1256 - acc: 0.9495 - val_loss: 0.0164 - val_acc: 0.9952
Epoch 16/20

Epoch 00016: LearningRateScheduler setting learning rate to 0.0005185825.
60000/60000 [==============================] - 38s 632us/step - loss: 0.1248 - acc: 0.9497 - val_loss: 0.0179 - val_acc: 0.9952
Epoch 17/20

Epoch 00017: LearningRateScheduler setting learning rate to 0.000491481.
60000/60000 [==============================] - 38s 629us/step - loss: 0.1222 - acc: 0.9509 - val_loss: 0.0160 - val_acc: 0.9952
Epoch 18/20

Epoch 00018: LearningRateScheduler setting learning rate to 0.0004670715.
60000/60000 [==============================] - 38s 630us/step - loss: 0.1231 - acc: 0.9495 - val_loss: 0.0181 - val_acc: 0.9949
Epoch 19/20

Epoch 00019: LearningRateScheduler setting learning rate to 0.0004449718.
60000/60000 [==============================] - 37s 623us/step - loss: 0.1207 - acc: 0.9505 - val_loss: 0.0166 - val_acc: 0.9947
Epoch 20/20

Epoch 00020: LearningRateScheduler setting learning rate to 0.000424869.
60000/60000 [==============================] - 37s 616us/step - loss: 0.1188 - acc: 0.9517 - val_loss: 0.0158 - val_acc: 0.9945
<keras.callbacks.History at 0x7f34214bcb00>

# 2. Result of model.evaluate:
[0.015848878198117016, 0.9945]

# 3. Strategy:
1. I have used the "he_uniform" initialisation.
2. To get rid of bias, use_bias= "False"
3. I have used 7 convolution layers of 3x3 and 1 convolution layer of 1x1. 
4. Used Batch Normalization.
5. To avoid overfitting used dropout.
