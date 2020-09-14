import tensorflow as tf 
import numpy as np
import cv2
from tensorflow_examples.models.pix2pix import pix2pix
import matplotlib.pyplot as plt
import sys


"""
    first we will use 2 output channels 
    one for the road lines and the rest
    for the background
"""
OUTPUT_CHANNELS = 2

"""
    it will be a good idea to used a pretrained model
    which is able to do object detection like Vgg19
    or (in our case) MobileNetV2.
    MobileNetV2 is pretty much the same as Vgg19
    but with less trainable paramerets and it was
    ment to be use on mobile phones. It is not
    that accurate when compared to Vgg19, but the
    difference between the models is quite small.
    Taking into consideration the tradeoff between
    accuracy and the nr of parameters i choosed
    to use MobileNetV2.
"""

TRAIN_PATH="C:\\Users\\bulzg\\Desktop\\road_detection\\crop"
TEST_PATH="C:\\Users\\bulzg\\Desktop\\road_detection\\crop"

mobileNetModel = tf.keras.applications.MobileNetV2(input_shape=[350, 1450, 3], include_top=False)

# Use the activations of these layers
layer_names = [
    'block_1_expand_relu',   
    'block_3_expand_relu',   
    'block_6_expand_relu',   
    'block_13_expand_relu',  
    'block_16_project',      
]
layers = [mobileNetModel.get_layer(name).output for name in layer_names]

# Create the feature extraction model
down_stack = tf.keras.Model(inputs=mobileNetModel.input, outputs=layers)

down_stack.trainable = False

up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),   # 32x32 -> 64x64
]



def unet_model(output_channels):
  inputs = tf.keras.layers.Input(shape=[128, 128, 3])
  x = inputs

  # Downsampling through the model
  skips = down_stack(x)
  x = skips[-1]
  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    concat = tf.keras.layers.Concatenate()
    x = concat([x, skip])

  # This is the last layer of the model
  last = tf.keras.layers.Conv2DTranspose(
      output_channels, 3, strides=2,
      padding='same')  #64x64 -> 128x128

  x = last(x)
  return tf.keras.Model(inputs=inputs, outputs=x)

def create_mask(pred_mask):
  pred_mask = tf.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask[0]



model = unet_model(OUTPUT_CHANNELS)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

img = cv2.imread(TRAIN_PATH + "//mask//0.jpeg")
# plt.imshow(img)
# plt.show()
img2 = tf.image.resize(img, (512, 512))
# img3 = tf.make_ndarray(img2)
im4 = cv2.resize(img, (512,512))
with tf.Session() as sess:  print((img2/255.0).eval()) 
plt.imshow(im4[:,:,0])
plt.show()


def normalize(input_image, input_mask):
  input_image = tf.cast(input_image, tf.float32) / 255.0
  mask = input_mask[:,:,1]
  mask = tf.cast(mask, tf.float32) / 255.0
  mask[input_mask > 0.5] = 1
  mask[input_mask > 0.5] = 0

  return input_image, input_mask


