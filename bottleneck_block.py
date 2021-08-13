import tensorflow as tf
import numpy as np

x = np.random.randn(20, 50, 50, 10)     # tensor 4 chiều batch_ize, row, height, channels
x = x.astype('float32')

def bottleneck_block(x):
    # depthwise conv do số filters bằng channels output nên ở đây ko cần số filters
    expansion = tf.keras.layers.Conv2D(filters=100, kernel_size=1, strides=1)   # tăng số channels 100
    depth_wise = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, padding='same', depth_multiplier=1)   # số channels vẫn 100  
    point_wise = tf.keras.layers.Conv2D(filters=10, kernel_size=1, strides=1)   # giảm số channels xuống 10

    y = expansion(x)
    y = depth_wise(y)
    y = point_wise(y)

    return y

y = bottleneck_block(x)
print("Output shape: ", y.shape)

""" 
    input - (20, 50, 50, 10)
    sau expansion - (20, 50, 50, 100)   tăng số channels
    sau depthwise - (20, 50, 50, 100)    số channels không đổi
    sau pointwise - (20, 50, 50, 10)   giảm số channels (projection)
"""