import tensorflow as tf 
import numpy as np
import cv2
# w8=None
# model = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
# for layer in model.layers:
#     print(layer.name)
#     if(layer.name == "block1_pool"):
#         w8 = layer.get_weights()
# # model.add(tf.keras.layers.Dense(4))
# print(model.summary())
# # print(model.get_config())
# print(w8)



input_1 = tf.keras.Input(([880, 1920, 3]), name='input_1', dtype='float32', sparse=False, ragged=False)


"""
 We will use the same implementations as VGG19 for the downsampling part
 and we will use the same weights as well
 the weights will be assigned at the end
"""
# downsampling
block1_conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), name='block1_conv1', dtype='float32', strides=(1,1), padding='same', data_format='channels_last', dilation_rate=(1,1), activation='relu', use_bias=True, bias_initializer='zeros', kernel_initializer='glorot_uniform')(input_1)
block1_conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), name='block1_conv2', dtype='float32', strides=(1,1), padding='same', data_format='channels_last', dilation_rate=(1,1), activation='relu', use_bias=True, bias_initializer='zeros', kernel_initializer='glorot_uniform')(block1_conv1)
block1_pool = tf.keras.layers.MaxPool2D(pool_size=(2,2), dtype='float32', name='block1_pool', padding='valid', strides=(2,2), data_format='channels_last')(block1_conv2)

block2_conv1 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), name='block2_conv1', dtype='float32', strides=(1,1), padding='same', data_format='channels_last', dilation_rate=(1,1), activation='relu', use_bias=True, bias_initializer='zeros', kernel_initializer='glorot_uniform')(block1_pool)
block2_conv2 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), name='block2_conv2', dtype='float32', strides=(1,1), padding='same', data_format='channels_last', dilation_rate=(1,1), activation='relu', use_bias=True, bias_initializer='zeros', kernel_initializer='glorot_uniform')(block2_conv1)
block2_pool = tf.keras.layers.MaxPool2D(pool_size=(2,2), dtype='float32', name='block2_pool', padding='valid', strides=(2,2), data_format='channels_last')(block2_conv2)

block3_conv1 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), name='block3_conv1', dtype='float32', strides=(1,1), padding='same', data_format='channels_last', dilation_rate=(1,1), activation='relu', use_bias=True, bias_initializer='zeros', kernel_initializer='glorot_uniform')(block2_pool)
block3_conv2 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), name='block3_conv2', dtype='float32', strides=(1,1), padding='same', data_format='channels_last', dilation_rate=(1,1), activation='relu', use_bias=True, bias_initializer='zeros', kernel_initializer='glorot_uniform')(block3_conv1)
block3_conv3 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), name='block3_conv3', dtype='float32', strides=(1,1), padding='same', data_format='channels_last', dilation_rate=(1,1), activation='relu', use_bias=True, bias_initializer='zeros', kernel_initializer='glorot_uniform')(block3_conv2)
block3_conv4 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), name='block3_conv4', dtype='float32', strides=(1,1), padding='same', data_format='channels_last', dilation_rate=(1,1), activation='relu', use_bias=True, bias_initializer='zeros', kernel_initializer='glorot_uniform')(block3_conv3)
block3_pool = tf.keras.layers.MaxPool2D(pool_size=(2,2), dtype='float32', name='block3_pool', padding='valid', strides=(2,2), data_format='channels_last')(block3_conv4)

block4_conv1 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), name='block4_conv1', dtype='float32', strides=(1,1), padding='same', data_format='channels_last', dilation_rate=(1,1), activation='relu', use_bias=True, bias_initializer='zeros', kernel_initializer='glorot_uniform')(block3_pool)
block4_conv2 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), name='block4_conv2', dtype='float32', strides=(1,1), padding='same', data_format='channels_last', dilation_rate=(1,1), activation='relu', use_bias=True, bias_initializer='zeros', kernel_initializer='glorot_uniform')(block4_conv1)
block4_conv3 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), name='block4_conv3', dtype='float32', strides=(1,1), padding='same', data_format='channels_last', dilation_rate=(1,1), activation='relu', use_bias=True, bias_initializer='zeros', kernel_initializer='glorot_uniform')(block4_conv2)
block4_conv4 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), name='block4_conv4', dtype='float32', strides=(1,1), padding='same', data_format='channels_last', dilation_rate=(1,1), activation='relu', use_bias=True, bias_initializer='zeros', kernel_initializer='glorot_uniform')(block4_conv3)
block4_pool = tf.keras.layers.MaxPool2D(pool_size=(2,2), dtype='float32', name='block4_pool', padding='valid', strides=(2,2), data_format='channels_last')(block4_conv4)

block5_conv1 = tf.keras.layers.Conv2D(filters=1024, kernel_size=(3,3), name='block5_conv1', dtype='float32', strides=(1,1), padding='same', data_format='channels_last', dilation_rate=(1,1), activation='relu', use_bias=True, bias_initializer='zeros', kernel_initializer='glorot_uniform')(block4_pool)
block5_conv2 = tf.keras.layers.Conv2D(filters=1024, kernel_size=(3,3), name='block5_conv2', dtype='float32', strides=(1,1), padding='same', data_format='channels_last', dilation_rate=(1,1), activation='relu', use_bias=True, bias_initializer='zeros', kernel_initializer='glorot_uniform')(block5_conv1)


# ------------ upsampling -------------------
block6_up = tf.keras.layers.UpSampling2D(size=(2,2),data_format='channels_last')(block5_conv2)
block6_merge = tf.keras.layers.concatenate([block6_up, block4_conv4], axis=3)
block6_conv1 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), name='block6_conv1', dtype='float32', strides=(1,1), padding='same', data_format='channels_last', dilation_rate=(1,1), activation='relu', use_bias=True, bias_initializer='zeros', kernel_initializer='glorot_uniform')(block6_merge)
block6_conv2 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), name='block6_conv2', dtype='float32', strides=(1,1), padding='same', data_format='channels_last', dilation_rate=(1,1), activation='relu', use_bias=True, bias_initializer='zeros', kernel_initializer='glorot_uniform')(block6_conv1)
block6_conv3 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), name='block6_conv3', dtype='float32', strides=(1,1), padding='same', data_format='channels_last', dilation_rate=(1,1), activation='relu', use_bias=True, bias_initializer='zeros', kernel_initializer='glorot_uniform')(block6_conv2)
block6_conv4 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), name='block6_conv4', dtype='float32', strides=(1,1), padding='same', data_format='channels_last', dilation_rate=(1,1), activation='relu', use_bias=True, bias_initializer='zeros', kernel_initializer='glorot_uniform')(block6_conv3)

block7_up = tf.keras.layers.UpSampling2D(size=(2,2),data_format='channels_last')(block6_conv4)
block7_merge = tf.keras.layers.concatenate([block7_up, block3_conv4], axis=3)
block7_conv1 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), name='block7_conv1', dtype='float32', strides=(1,1), padding='same', data_format='channels_last', dilation_rate=(1,1), activation='relu', use_bias=True, bias_initializer='zeros', kernel_initializer='glorot_uniform')(block7_merge)
block7_conv2 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), name='block7_conv2', dtype='float32', strides=(1,1), padding='same', data_format='channels_last', dilation_rate=(1,1), activation='relu', use_bias=True, bias_initializer='zeros', kernel_initializer='glorot_uniform')(block7_conv1)
block7_conv3 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), name='block7_conv3', dtype='float32', strides=(1,1), padding='same', data_format='channels_last', dilation_rate=(1,1), activation='relu', use_bias=True, bias_initializer='zeros', kernel_initializer='glorot_uniform')(block7_conv2)
block7_conv4 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), name='block7_conv4', dtype='float32', strides=(1,1), padding='same', data_format='channels_last', dilation_rate=(1,1), activation='relu', use_bias=True, bias_initializer='zeros', kernel_initializer='glorot_uniform')(block7_conv3)

block8_up = tf.keras.layers.UpSampling2D(size=(2,2), data_format='channels_last')(block7_conv4)
block8_merge = tf.keras.layers.concatenate([block8_up, block2_conv2], axis=3)
block8_conv1 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), name='block8_conv1', dtype='float32', strides=(1,1), padding='same', data_format='channels_last', dilation_rate=(1,1), activation='relu', use_bias=True, bias_initializer='zeros', kernel_initializer='glorot_uniform')(block8_merge)
block8_conv2 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), name='block8_conv2', dtype='float32', strides=(1,1), padding='same', data_format='channels_last', dilation_rate=(1,1), activation='relu', use_bias=True, bias_initializer='zeros', kernel_initializer='glorot_uniform')(block8_conv1)

block9_up = tf.keras.layers.UpSampling2D(size=(2,2), data_format='channels_last')(block8_conv2)
block9_merge = tf.keras.layers.concatenate([block9_up, block1_conv2], axis=3)
block9_conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), name='block9_conv1', dtype='float32', strides=(1,1), padding='same', data_format='channels_last', dilation_rate=(1,1), activation='relu', use_bias=True, bias_initializer='zeros', kernel_initializer='glorot_uniform')(block9_merge)
block9_conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), name='block9_conv2', dtype='float32', strides=(1,1), padding='same', data_format='channels_last', dilation_rate=(1,1), activation='relu', use_bias=True, bias_initializer='zeros', kernel_initializer='glorot_uniform')(block9_conv1)

block10_conv1 = tf.keras.layers.Conv2D(filters=2, kernel_size=(3,3), name='block10_conv1', dtype='float32', strides=(1,1), padding='same', data_format='channels_last', dilation_rate=(1,1), activation='relu', use_bias=True, bias_initializer='zeros', kernel_initializer='glorot_uniform')(block9_conv2)

output_1 = tf.keras.layers.Conv2D(filters=1, kernel_size=(3,3), name='output_1', dtype='float32', strides=(1,1), padding='same', data_format='channels_last', dilation_rate=(1,1), activation='relu', use_bias=True, bias_initializer='zeros', kernel_initializer='glorot_uniform')(block10_conv1)



model = tf.keras.Model(inputs=input_1, outputs=output_1)
model.compile(optimizer = tf.keras.optimizers.Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
print(model.summary())

