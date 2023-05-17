import tensorflow as tf
from DIFRINT_Implementation.PWCDCNet import PWCDCNet
import numpy as np
import tensorflow_addons as tfa


class PWC(tf.keras.Model):
    '''takes in 2 images and computes optical flow, 
    then warps the first image with the optical flow
    '''
    def __init__(self):
        super(PWC,self).__init__()
        self.model = PWCDCNet()
        #self.build(input_shape)

    def build(self,input_shape):
        b, h, w, c = input_shape
        if h % 64 != 0 or w % 64 != 0:
            new_h = (int(h/64) + 1) * 64
            new_w = (int(w/64) + 1) * 64
            input_shape = (b,new_h,new_w,c)
        temp = tf.zeros((input_shape),dtype=tf.float32)
        self.model(temp)
        self.model.load_weights('pwc.h5')

    def call(self,input):
        b,h,w,c = tf.unstack(tf.shape(input))
        if h % 64 != 0 or w % 64 != 0:
            new_h = (int(h/64) + 1) * 64
            new_w = (int(w/64) + 1) * 64
            input = tf.image.pad_to_bounding_box(input, 0, 0, new_h, new_w)
        flows = self.model(input,is_training=False)
        flows = tf.image.crop_to_bounding_box(flows, 0, 0, h//4, w//4)
        flows = tf.image.resize(flows, (h, w), method=tf.image.ResizeMethod.BILINEAR)
        input = tf.image.resize(input,(h,w), method = tf.image.ResizeMethod.BILINEAR)
        flows *= 4
        frame = input[...,0:3]
        warped = tfa.image.dense_image_warp(frame,flows)
        return(warped)

##################################################################

class ResNet(tf.keras.Model):
    def __init__(self):
        super(ResNet, self).__init__()
        class ConvBlock(tf.keras.layers.Layer):
            def __init__(self, in_ch, out_ch):
                super(ConvBlock, self).__init__()
                self.conv = tf.keras.layers.Conv2D(out_ch, kernel_size=1, strides=1, padding='same')
                self.lrelu = tf.keras.layers.LeakyReLU(0.2)
                self.gate_conv = tf.keras.Sequential([
                    tf.keras.layers.Conv2D(in_ch, kernel_size=3, strides=1, padding='same'),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Activation('relu'),
                    tf.keras.layers.Conv2D(out_ch, kernel_size=3, strides=1, padding='same'),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Activation('sigmoid')
                ])

            def call(self, x):
                x_conv = self.conv(x)
                x_lrelu = self.lrelu(x_conv)
                x_gate_conv = self.gate_conv(x)
                return x_lrelu * x_gate_conv
        class ResBlock(tf.keras.layers.Layer):
            def __init__(self, num_ch):
                super(ResBlock, self).__init__()
                self.conv = tf.keras.layers.Conv2D(num_ch, kernel_size=1, strides=1, padding='same')
                self.lrelu = tf.keras.layers.LeakyReLU(0.2)
                self.gate_conv = tf.keras.Sequential([
                    tf.keras.layers.Conv2D(num_ch, kernel_size=3, strides=1, padding='same'),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Activation('relu'),
                    tf.keras.layers.Conv2D(num_ch, kernel_size=3, strides=1, padding='same'),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Activation('sigmoid')
                ])

            def call(self, x):
                x_conv = self.conv(x)
                x_lrelu = self.lrelu(x_conv)
                x_gate_conv = self.gate_conv(x)
                return x_lrelu * x_gate_conv + x    


        self.conv1 = ConvBlock(6, 32)
        self.res1 = ResBlock(32)
        self.res2 = ResBlock(32)
        self.res3 = ResBlock(32)
        self.res4 = ResBlock(32)
        self.res5 = ResBlock(32)
        self.conv2 = ConvBlock(32, 3)
        self.tanh = tf.keras.layers.Activation('tanh')

    def call(self, inputs):
        x = tf.concat(inputs, axis=-1)
        x = self.conv1(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.conv2(x)
        x = self.tanh(x)
        return x
###################################################################
class Encoder(tf.keras.layers.Layer):
            def __init__(self, in_nc, out_nc, stride, k_size=3, pad=(1,1)):
                super(Encoder, self).__init__()

                self.seq = tf.keras.Sequential([
                    tf.keras.layers.Conv2D(out_nc, kernel_size=k_size, strides=stride, padding='same'),
                    tf.keras.layers.LeakyReLU(0.2)
                ])
                self.GateConv = tf.keras.Sequential([
                    tf.keras.layers.Conv2D(out_nc, kernel_size=k_size, strides=stride, padding='same'),
                    tf.keras.layers.Activation('sigmoid')
                ])

            def call(self, x):
                feature = self.seq(x)
                gate = self.GateConv(x)
                return feature * gate
            
class Decoder(tf.keras.layers.Layer):
            def __init__(self, in_nc, out_nc, stride, k_size=3, pad=1, tanh=False):
                super(Decoder, self).__init__()

                self.seq = tf.keras.Sequential([
                    tf.keras.layers.Conv2D(in_nc, kernel_size=k_size, strides=stride, padding='same'),
                    tf.keras.layers.Conv2D(out_nc, kernel_size=k_size, strides=stride, padding='same')
                ])

                if tanh:
                    self.activ = tf.keras.layers.Activation('tanh')
                else:
                    self.activ = tf.keras.layers.LeakyReLU(0.2)

                self.GateConv = tf.keras.Sequential([
                    tf.keras.layers.Conv2D(in_nc, kernel_size=k_size, strides=stride, padding='same'),
                    tf.keras.layers.Conv2D(out_nc, kernel_size=k_size, strides=stride, padding='same'),
                    tf.keras.layers.Activation('sigmoid')
                ])

            def call(self, x):
                s = self.seq(x)
                s = self.activ(s)
                return s * self.GateConv(x)

class UNet(tf.keras.Model):
    def __init__(self):
        super(UNet,self).__init__()
        self.enc0 = Encoder(6, 32, stride=2)
        self.enc1 = Encoder(32, 32, stride=2)
        self.enc2 = Encoder(32, 32, stride=2)
        self.enc3 = Encoder(32, 32, stride=2)
        self.dec0 = Decoder(32, 32, stride=1 )
        # up-scaling + concat
        self.dec1 = Decoder(32+32, 32, stride=1)
        self.dec2 = Decoder(32+32, 32, stride=1)
        self.dec3 = Decoder(32+32, 32, stride=1)

        self.dec4 = Decoder(32, 3, stride=1, tanh=True)

    def call(self,input):
        s0 = self.enc0(input)
        s1 = self.enc1(s0)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)

        s4 = self.dec0(s3)
        #upscaling and concatenating
        s4 = tf.keras.layers.UpSampling2D(size = (2,2),interpolation='nearest')(s4)
        s5 = self.dec1(tf.concat([s4,s2],axis=-1))
        s5 = tf.keras.layers.UpSampling2D((2,2),interpolation='nearest')(s5)
        s6 = self.dec2(tf.concat([s5,s1],axis=-1))
        s6 = tf.keras.layers.UpSampling2D((2,2),interpolation='nearest')(s6)
        s7 = self.dec3(tf.concat([s6,s0],axis=-1))
        s7 = tf.keras.layers.UpSampling2D(size = (2,2),interpolation='nearest')(s7)
        out = self.dec4(s7)
        return out
