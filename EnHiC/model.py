import numpy as np
import tensorflow as tf
import time
from IPython import display
import datetime
import os
import logging


class Reconstruct_R1M(tf.keras.layers.Layer):
    def __init__(self, filters, name='RR'):
        super(Reconstruct_R1M, self).__init__(name=name)
        self.num_outputs = filters

    def build(self, input_shape):
        w_init = tf.ones_initializer()
        self.w = tf.Variable(initial_value=w_init(
            shape=(1, 1, 1, self.num_outputs), dtype='float32'))

    def call(self, input):
        v = tf.math.add(input, tf.constant(1e-6, dtype=tf.float32))
        vt = tf.transpose(v, perm=[0, 2, 1, 3])
        rank1m = tf.multiply(tf.multiply(v, vt), self.w)
        return rank1m


class Weight_R1M(tf.keras.layers.Layer):
    def __init__(self, name='WR1M'):
        super(Weight_R1M, self).__init__(name=name)

    def build(self, input_shape):
        w_init = tf.keras.initializers.RandomUniform(minval=0, maxval=4.0)
        self.w = tf.Variable(initial_value=w_init(
            shape=(1, 1, 1, input_shape[-1]), dtype='float32'))

    def call(self, input):
        self.w.assign(tf.nn.relu(self.w))
        return tf.multiply(input, self.w)


class Downpixel(tf.keras.layers.Layer):
    def __init__(self, r, name=None):
        super(Downpixel, self).__init__(name=name)
        self.r = r

    def _phase_shift(self, I):
        r = self.r
        X = tf.nn.space_to_depth(
            input=I, block_size=r, data_format='NHWC', name=None)
        return X

    def call(self, inputs):
        r = self.r
        kernel = tf.ones(shape=(r, r, 1, 1), dtype=tf.float32)/(r*r)
        conv = tf.nn.conv2d(inputs, kernel, padding='SAME', strides=(1, 1))
        return self._phase_shift(conv)


class Subpixel(tf.keras.layers.Conv2D):
    def __init__(self, filters, kernel_size, r, padding='valid', data_format=None,
                 strides=(1, 1), activation=None, use_bias=True,
                 kernel_initializer='glorot_uniform', bias_initializer='zeros',
                 kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                 kernel_constraint=None, bias_constraint=None, **kwargs):
        super(Subpixel, self).__init__(filters=r*r*filters, kernel_size=kernel_size,
                                       strides=strides, padding=padding, data_format=data_format,
                                       activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer,
                                       bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer,
                                       bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer,
                                       kernel_constraint=kernel_constraint, bias_constraint=bias_constraint, **kwargs)
        self.r = r

    def _phase_shift(self, I):
        r = self.r
        X = tf.nn.depth_to_space(
            input=I, block_size=r, data_format='NHWC', name=None)
        return X

    def call(self, inputs):
        return self._phase_shift(super(Subpixel, self).call(inputs))

    def compute_output_shape(self, input_shape):
        unshifted = super(Subpixel, self).compute_output_shape(input_shape)
        return (unshifted[0], self.r*unshifted[1], self.r*unshifted[2], unshifted[3]/(self.r*self.r))


class Sum_R1M(tf.keras.layers.Layer):
    def __init__(self, name=None):
        super(Sum_R1M, self).__init__(name=name)

    def call(self, input):
        return tf.reduce_sum(input, axis=-1, keepdims=True)


class Symmetry_R1M(tf.keras.layers.Layer):
    def __init__(self, name=None):
        super(Symmetry_R1M, self).__init__(name=name)

    def build(self, input_shape):
        ones = tf.ones(shape=(input_shape[1], input_shape[2]), dtype='float32')
        diag = tf.linalg.band_part(ones, 0, 0)*0.5
        upper = tf.linalg.band_part(ones, 0, -1)

        self.w = upper - diag
        self.w = tf.expand_dims(self.w, 0)
        self.w = tf.expand_dims(self.w, -1)

    def call(self, input):
        up = tf.multiply(input, self.w)
        low = tf.transpose(up, perm=[0, 2, 1, 3])
        return up + low


class Normal(tf.keras.layers.Layer):
    def __init__(self, input_dim, name=None):
        super(Normal, self).__init__(name=name)
        self.dim = input_dim

    def build(self, input_shape):
        w_init = tf.ones_initializer()
        self.w = tf.Variable(initial_value=w_init(
            shape=(1, self.dim, 1, 1), dtype='float32'), trainable=True)

    def call(self, inputs):
        rowsr = tf.math.sqrt(tf.math.reduce_sum(
            tf.multiply(inputs, inputs), axis=1, keepdims=True))
        colsr = tf.math.sqrt(tf.math.reduce_sum(
            tf.multiply(inputs, inputs), axis=2, keepdims=True))
        sumele = tf.math.multiply(rowsr, colsr)
        Div = tf.math.divide_no_nan(inputs, sumele)
        self.w.assign(tf.nn.relu(self.w))
        WT = tf.transpose(self.w, perm=[0, 2, 1, 3])
        M = tf.multiply(self.w, WT)
        return tf.multiply(Div, M)


def block_downsample_decomposition(len_size, channels_decompose, input_len_size, input_channels, downsample_ratio, name=None):
    result = tf.keras.Sequential(name=name)
    result.add(tf.keras.layers.Input(
        shape=(input_len_size, input_len_size, input_channels)))
    result.add(Downpixel(downsample_ratio))
    result.add(tf.keras.layers.Conv2D(channels_decompose, [1, len_size], strides=(1, 1), padding='valid', data_format="channels_last",
                                      activation='relu', use_bias=False,
                                      kernel_constraint=tf.keras.constraints.NonNeg(),
                                      kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.01, stddev=0.1)))
    result.add(Weight_R1M())
    result.add(Reconstruct_R1M(channels_decompose))
    return result


def block_rank1channels_convolution(channels, input_len_size, input_channels, name=None):
    result = tf.keras.Sequential(name=name)
    result.add(tf.keras.layers.Input(
        shape=(input_len_size, input_len_size, input_channels)))
    result.add(tf.keras.layers.Conv2D(channels, [1, 1], strides=1, padding='same', data_format="channels_last",
                                      activation='relu', use_bias=False,
                                      name=name))
    result.add(Weight_R1M())
    result.add(Symmetry_R1M())
    return result


def block_upsample_convolution(channels, input_len_size, input_channels, upsample_ratio, name=None):
    result = tf.keras.Sequential(name=name)
    result.add(tf.keras.layers.Input(
        shape=(input_len_size, input_len_size, input_channels)))
    result.add(Subpixel(filters=int(channels), kernel_size=(3, 3), r=upsample_ratio,
                        activation='relu', use_bias=False, padding='same',
                        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.01, stddev=0.1)))
    result.add(tf.keras.layers.BatchNormalization())
    result.add(Symmetry_R1M())
    return result


def block_rank1_estimation(dims, input_len_size, input_channels, name=None):
    result = tf.keras.Sequential(name=name)
    result.add(tf.keras.layers.Input(
        shape=(input_len_size, input_len_size, input_channels)))
    result.add(Sum_R1M())
    result.add(Normal(dims))
    return result


def block_channel_combination(channels, name=None):
    result = tf.keras.Sequential(name=name)
    result.add(tf.keras.layers.Conv2D(channels, [1, 1], strides=1, padding='same', data_format="channels_last",
                                      kernel_initializer=tf.keras.initializers.RandomUniform(
                                          minval=0., maxval=1./channels),
                                      kernel_constraint=tf.keras.constraints.NonNeg(),
                                      activation='relu', use_bias=False))
    return result


def make_generator_model(len_high_size=128, scale=4):

    len_low_size_x2 = int(len_high_size/(scale/2))
    len_low_size_x4 = int(len_high_size/scale)
    len_low_size_x8 = int(len_high_size/(scale*2))
    inp = tf.keras.layers.Input(
        shape=(len_high_size, len_high_size, 1), name='in', dtype=tf.float32)

    low_x2 = tf.keras.layers.AveragePooling2D(
        pool_size=(2, 2), strides=2, padding='valid', name='p_x2')(inp)
    low_x4 = tf.keras.layers.AveragePooling2D(
        pool_size=(2, 2), strides=2, padding='valid', name='p_x4')(low_x2)

    dsd_x4 = block_downsample_decomposition(len_size=len_low_size_x4, channels_decompose=384, input_len_size=len_high_size,
                                            input_channels=1, downsample_ratio=4, name='dsd_x4')
    rech_x4 = dsd_x4(inp)
    r1c = block_rank1channels_convolution(
        channels=128, input_len_size=len_low_size_x4, input_channels=384, name='r1c_x4')
    sym_x4 = r1c(rech_x4)
    r1e = block_rank1_estimation(
        dims=len_low_size_x4, input_len_size=len_low_size_x4, input_channels=384, name='r1e_x4')
    out_low_x4 = r1e(rech_x4)

    usc_x4 = block_upsample_convolution(
        channels=128, input_len_size=len_low_size_x4, input_channels=128, upsample_ratio=2, name='usc_x4')
    sym_x4 = usc_x4(sym_x4)

    dsd_x2 = block_downsample_decomposition(len_size=len_low_size_x2, channels_decompose=768, input_len_size=len_high_size,
                                            input_channels=1, downsample_ratio=2, name='dsd_x2')
    rech_x2 = dsd_x2(inp)
    r1c_x2 = block_rank1channels_convolution(
        channels=128, input_len_size=len_low_size_x2, input_channels=768, name='r1c_x2')
    sym_x2 = r1c_x2(rech_x2)
    r1e_x2 = block_rank1_estimation(
        dims=len_low_size_x2, input_len_size=len_low_size_x2, input_channels=768, name='r1e_x2')
    out_low_x2 = r1e_x2(rech_x2)

    concat = tf.keras.layers.concatenate([sym_x4, sym_x2], axis=-1)
    concat = block_channel_combination(channels=128, name='cc_x2')(concat)
    usc_x2 = block_upsample_convolution(
        channels=64, input_len_size=len_low_size_x2, input_channels=128, upsample_ratio=2, name='usc_x2')
    sym = usc_x2(concat)

    Sumh = tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1),
                                  strides=(1, 1), padding='same',
                                  data_format="channels_last",
                                  kernel_constraint=tf.keras.constraints.NonNeg(),
                                  activation='relu', use_bias=False, name='sum_high')(sym)
    high_out = Normal(int(len_high_size), name='out_high')(Sumh)

    model = tf.keras.models.Model(
        inputs=[inp], outputs=[out_low_x2, out_low_x4, high_out, low_x2, low_x4])
    return model


def block_rank1_decompose_reconstruct(len_size, channels_decompose, input_len_size, input_channels, name=None):
    result = tf.keras.Sequential(name=name)
    result.add(tf.keras.layers.Input(
        shape=(input_len_size, input_len_size, input_channels)))
    result.add(tf.keras.layers.Conv2D(channels_decompose, [1, len_size], strides=(1, 1), padding='valid', data_format="channels_last",
                                      activation='relu', use_bias=False,
                                      kernel_constraint=tf.keras.constraints.NonNeg(),
                                      kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.01, stddev=0.1)))
    result.add(Reconstruct_R1M(channels_decompose))
    return result


def block_down_convolution(channels, input_len_size, input_channels, name=None):
    result = tf.keras.Sequential(name=name)
    result.add(tf.keras.layers.Input(
        shape=(input_len_size, input_len_size, input_channels)))
    result.add(tf.keras.layers.Conv2D(channels, kernel_size=(3, 3), strides=(1, 1), padding='same', data_format="channels_last",
                                      activation=None, use_bias=False,
                                      kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.01, stddev=0.1)))
    result.add(tf.keras.layers.LeakyReLU(0.2))
    result.add(tf.keras.layers.MaxPool2D(
        pool_size=(2, 2), strides=None, padding='valid'))
    return result


def make_discriminator_model(len_high_size=128, scale=4):
    '''PatchGAN 1 pixel of output represents X pixels of input: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/issues/39
    The "70" is implicit, it's not written anywhere in the code
    but instead emerges as a mathematical consequence of the network architecture.
    The math is here: https://github.com/phillipi/pix2pix/blob/master/scripts/receptive_field_sizes.m
    compute input size from a given output size:
    f = @(output_size, ksize, stride) (output_size - 1) * stride + ksize; fix output_size as 1 '''
    len_x1 = int(len_high_size)
    len_x2 = int(len_high_size/(scale/2))
    len_x4 = int(len_high_size/scale)
    len_x8 = int(len_high_size/(scale*2))
    inp = tf.keras.layers.Input(
        shape=(len_high_size, len_high_size, 1), name='in', dtype=tf.float32)

    b_r1dr = block_rank1_decompose_reconstruct(len_size=len_x1,
                                               channels_decompose=512, input_len_size=len_x1,
                                               input_channels=1, name='r1dr_x1')
    r1dr_x1 = b_r1dr(inp)
    b_dc = block_down_convolution(
        channels=80, input_len_size=len_x1, input_channels=512, name='dc_x1')
    dc_x1 = b_dc(r1dr_x1)

    ratio = 2
    dp_x2 = Downpixel(r=ratio, name='dp_x2')(inp)
    b_r1dr = block_rank1_decompose_reconstruct(len_size=len_x2,
                                               channels_decompose=512, input_len_size=len_x2,
                                               input_channels=ratio**2, name='r1dr_x2')
    r1dr_x2 = b_r1dr(dp_x2)
    b_r1c = block_rank1channels_convolution(
        channels=40, input_len_size=len_x2, input_channels=512, name='r1c_x2')
    r1c_x2 = b_r1c(r1dr_x2)

    concat_x1_x2 = tf.keras.layers.Concatenate()([r1c_x2, dc_x1])
    b_dc = block_down_convolution(
        channels=120, input_len_size=len_x2, input_channels=120, name='dc_x2')
    dc_x2 = b_dc(concat_x1_x2)

    ratio = 4
    dp_x4 = Downpixel(r=ratio, name='dp_x4')(inp)
    b_r1dr = block_rank1_decompose_reconstruct(
        len_size=len_x4, channels_decompose=256, input_len_size=len_x4, input_channels=ratio**2, name='r1dr_x4')
    r1dr_x4 = b_r1dr(dp_x4)
    b_r1c = block_rank1channels_convolution(
        channels=20, input_len_size=len_x4, input_channels=256, name='r1c_x4')
    r1c_x4 = b_r1c(r1dr_x4)

    concat_x2_x4 = tf.keras.layers.Concatenate()([r1c_x4, dc_x2])
    b_dc = block_down_convolution(
        channels=60, input_len_size=len_x4, input_channels=140, name='dc_x4')
    dc_x4 = b_dc(concat_x2_x4)

    ratio = 8
    dp_x8 = Downpixel(r=ratio, name='dp_x8')(inp)
    b_r1dr = block_rank1_decompose_reconstruct(
        len_size=len_x8, channels_decompose=128, input_len_size=len_x8, input_channels=ratio**2, name='r1dr_x8')
    r1dr_x8 = b_r1dr(dp_x8)
    b_r1c = block_rank1channels_convolution(
        channels=10, input_len_size=len_x8, input_channels=128, name='r1c_x8')
    r1c_x8 = b_r1c(r1dr_x8)

    concat_x4_x8 = tf.keras.layers.Concatenate()([r1c_x8, dc_x4])
    b_dc = block_down_convolution(
        channels=80, input_len_size=len_x8, input_channels=70, name='dc_x8')
    dc_x8 = b_dc(concat_x4_x8)

    conv = tf.keras.layers.Conv2D(filters=80, kernel_size=(
        1, 1), strides=1, padding='same')(dc_x8)
    conv = tf.keras.layers.Flatten()(conv)
    last = tf.keras.layers.Dense(1, activation=None)(conv)
    return tf.keras.Model(inputs=inp, outputs=last)


def discriminator_bce_loss(real_output, fake_output):
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    real_loss = loss_object(tf.ones_like(real_output), real_output)
    generated_loss = loss_object(tf.zeros_like(fake_output), fake_output)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss


def generator_bce_loss(d_pred):
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    gan_loss = loss_object(tf.ones_like(d_pred), d_pred)
    return gan_loss


def generator_ssim_loss(y_pred, y_true):  # , m_filter):
    return tf.reduce_mean((1 - tf.image.ssim(y_pred, y_true, max_val=1.0, filter_size=11))/2.0)


def generator_mse_loss(y_pred, y_true):  # , m_filter):
    mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
    diff = mse(y_pred, y_true)
    diff = tf.reduce_sum(diff, [1,2])
    diff = tf.reduce_mean(diff)
    return diff


# @tf.function
# solution: https://github.com/tensorflow/tensorflow/issues/27120#issuecomment-615870307
def _train_step_generator(Gen, Dis, imgl, imgr, loss_filter, loss_weights, opts):
    with tf.GradientTape() as x, tf.GradientTape() as gen_tape_high:
        fake_hic = Gen(imgl, training=True)
        fake_hic_l_x2 = fake_hic[0]
        # imgl_x2 = fake_hic[4]
        imgl_x2 = fake_hic[3]
        mfilter_low = tf.expand_dims(loss_filter[0], axis=0)
        mfilter_low = tf.expand_dims(mfilter_low, axis=-1)
        mfilter_low = tf.cast(mfilter_low, tf.float32)
        fake_hic_l_x2 = tf.multiply(fake_hic_l_x2, mfilter_low)
        imgl_x2_filter = tf.multiply(imgl_x2, mfilter_low)

        fake_hic_l_x4 = fake_hic[1]
        # imgl_x4 = fake_hic[5]
        imgl_x4 = fake_hic[4]
        mfilter_low = tf.expand_dims(loss_filter[1], axis=0)
        mfilter_low = tf.expand_dims(mfilter_low, axis=-1)
        mfilter_low = tf.cast(mfilter_low, tf.float32)
        fake_hic_l_x4 = tf.multiply(fake_hic_l_x4, mfilter_low)
        imgl_x4_filter = tf.multiply(imgl_x4, mfilter_low)

        loss_low_ssim_x2 = generator_ssim_loss(
            fake_hic_l_x2, imgl_x2_filter)
        loss_low_mse_x2 = generator_mse_loss(fake_hic_l_x2, imgl_x2_filter)

        loss_low_ssim_x4 = generator_ssim_loss(
            fake_hic_l_x4, imgl_x4_filter)
        loss_low_mse_x4 = generator_mse_loss(fake_hic_l_x4, imgl_x4_filter)

        loss_low_ssim = (loss_low_ssim_x4*4.0 + loss_low_ssim_x2*16.0)/20.0
        loss_low_mse = (loss_low_mse_x4*4.0 + loss_low_mse_x2*16.0)/20.0
        loss_low = loss_low_ssim + loss_low_mse

        fake_hic_h = fake_hic[2]
        mfilter_high = tf.expand_dims(loss_filter[2], axis=0)
        mfilter_high = tf.expand_dims(mfilter_high, axis=-1)
        mfilter_high = tf.cast(mfilter_high, tf.float32)

        fake_hic_h = tf.multiply(fake_hic_h, mfilter_high)
        imgr_filter = tf.multiply(imgr, mfilter_high)
        disc_generated_output = Dis(fake_hic_h, training=False)

        loss_high_0 = generator_bce_loss(disc_generated_output)
        loss_high_1 = generator_mse_loss(fake_hic_h, imgr_filter)
        loss_high_2 = generator_ssim_loss(fake_hic_h, imgr_filter)

        loss_high = loss_high_0 * loss_weights[0] + \
                    loss_high_1 * loss_weights[1] + \
                    loss_high_2 * loss_weights[2]

    gen_low_v = []
    gen_low_v += Gen.get_layer('dsd_x2').trainable_variables
    gen_low_v += Gen.get_layer('r1e_x2').trainable_variables
    gen_low_v += Gen.get_layer('dsd_x4').trainable_variables
    gen_low_v += Gen.get_layer('r1e_x4').trainable_variables
    gradients_of_generator_low = x.gradient(loss_low, gen_low_v)

    gen_high_v = []
    gen_high_v += Gen.get_layer('r1c_x2').trainable_variables
    gen_high_v += Gen.get_layer('usc_x2').trainable_variables
    gen_high_v += Gen.get_layer('r1c_x4').trainable_variables
    gen_high_v += Gen.get_layer('usc_x4').trainable_variables
    gen_high_v += Gen.get_layer('cc_x2').trainable_variables
    gen_high_v += Gen.get_layer('sum_high').trainable_variables
    gen_high_v += Gen.get_layer('out_high').trainable_variables
    gradients_of_generator_high = gen_tape_high.gradient(loss_high, gen_high_v)

    # apply gradients
    opts[0].apply_gradients(zip(gradients_of_generator_low, gen_low_v))
    opts[1].apply_gradients(zip(gradients_of_generator_high, gen_high_v))

    # log losses
    return loss_low_ssim, loss_low_mse, loss_high_0, loss_high_1, loss_high_2


# @tf.function
def _train_step_discriminator(Gen, Dis, imgl, imgr, loss_filter, opts):
    with tf.GradientTape() as disc_tape:
        fake_hic = Gen(imgl, training=False)
        # fake_hic_h = fake_hic[3]
        fake_hic_h = fake_hic[2]

        mfilter_high = tf.expand_dims(loss_filter[0], axis=0)
        mfilter_high = tf.expand_dims(mfilter_high, axis=-1)
        mfilter_high = tf.cast(mfilter_high, tf.float32)

        fake_hic_h = tf.multiply(fake_hic_h, mfilter_high)
        imgr_filter = tf.multiply(imgr, mfilter_high)

        disc_generated_output = Dis(fake_hic_h, training=True)
        disc_real_output = Dis(imgr_filter, training=True)
        loss = discriminator_bce_loss(disc_real_output, disc_generated_output)
    discriminator_gradients = disc_tape.gradient(loss, Dis.trainable_variables)
    opts[0].apply_gradients(
        zip(discriminator_gradients, Dis.trainable_variables))
    return loss


if __name__ == '__main__':
    len_size = 400
    scale = 4
    Gen = make_generator_model(len_high_size=len_size, scale=scale)
    Dis = make_discriminator_model(len_high_size=len_size, scale=scale)
    print(Gen.summary())
    tf.keras.utils.plot_model(Gen, to_file='G.png', show_shapes=True)
    print(Dis.summary())
    tf.keras.utils.plot_model(Dis, to_file='D.png', show_shapes=True)
