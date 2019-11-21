# Output of print(score):
[0.03373274423182986, 0.9926]



# 1. Convolution:
Convolutin is the process of elementwise multiplication followed by addition of all the terms. This process use to fetch the important information from the given matrices. By this method we determine edge, texture of the given image. Convolution is a linear transformamtion of the input matices.

# 2. Filters/Kernels:
Filter is again a matrix with the help of which we do convolution process. Depth of the kernel depends on the input image depth. In general we use 3x3 kernel. This kerner reduces the size of output by 2 if size of stride is 1.


# 3. Epochs:
Epoch is defined as the count of the no of times the model has seens the whole dataset. For Example we have trained our model once with entire training data then no on epoch is 1, if we have traind 2 times ther epoch will be 2 and so on.

# 4. 1x1 Convolution:
1x1 convolution is the process to reduce the no of channels which finally reduce the computation cost. This process of convolution retain the feature but reuce the dimension of the input image.

# 5. 3x3 Convolution:
The process of convolution by which we do edge detection and all. After convolving by 3x3 filter size gets reduced by 2, if stride size is 1.

# 6. Feature Maps:
Feature Map is the result of convolving a kernel/filter on the input image.

# 7. Activation Function:
Activation functon decides if a given neuron should get activated or not.
There are various types of Activation function:
1. Sigmoid    2. tanh     3. ReLu     etc.

# 8. Receptive Field
This field is spatial extension for a filter. Receptive field can be understood as a part of image visible by the filetr at any given time.
There are 2 types of Receptive Field:
Local Receptive Field and Global Receptive Field.
