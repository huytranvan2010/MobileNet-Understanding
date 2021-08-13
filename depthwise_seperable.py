import tensorflow as tf
import numpy as np

x = np.random.randn(20, 50, 50, 10)     # tensor 4 chiều batch_ize, row, height, channels
x = x.astype('float32')

def depthwise_seperable_conv(x):
    # depthwise conv do số filters bằng channels output nên ở đây ko cần số filters
    depth_wise = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, padding='same', depth_multiplier=1)      
    point_wise = tf.keras.layers.Conv2D(filters=100, kernel_size=1, strides=1)

    y = depth_wise(x)
    y = point_wise(y)

    return y

y = depthwise_seperable_conv(x)
print(y.shape)

""" 
    input - (20, 50, 50, 10)
    sau depthwise - (20, 50, 50, 10)    số channels không đổi
    sau pointwise - (20, 50, 50, 100)   đầu ra mong muốn
"""