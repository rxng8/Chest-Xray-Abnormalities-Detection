#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
   Author: Alex Nguyen
   Gettysburg College
"""

from tensorflow_examples.models.pix2pix import pix2pix
import tensorflow as tf

def make_dense_layer(out_channels, activation='relu'):
    return tf.keras.layers.Dense(out_channels, activation=activation)

def make_conv_layer(out_channels, strides=1, activation='relu', padding='same'):
    layer = tf.keras.layers.Conv2D(
        filters=out_channels,
        kernel_size=(4, 4),
        activation=activation,
        padding=padding
    )
    return layer

def make_dropout_layer(rate=0.5):
    return tf.keras.layers.Dropout(rate)

def make_max_pooling_layer():
    return tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2),
        padding='same'
    )

def make_batch_norm_layer(**kwargs):
    return tf.keras.layers.BatchNormalization(**kwargs)

def make_down_conv_sequence(out_channels):
    return tf.keras.Sequential([
        make_conv_layer(out_channels),
        make_max_pooling_layer(),
        make_dropout_layer()
    ])

def make_up_conv_layer(out_channels):
    return tf.keras.Sequential([
        pix2pix.upsample(out_channels, 4),
        make_dropout_layer()
    ])

def make_deconv_layer(out_channels, activation='relu'):
    return tf.keras.layers.Conv2DTranspose(
        out_channels, 4, strides=1,
        padding='same',
        activation=activation
    )