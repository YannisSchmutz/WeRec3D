from keras.layers import Concatenate, Conv3D, ZeroPadding3D, Conv3DTranspose, Cropping3D
from keras.layers import BatchNormalization
from keras.activations import elu, linear, tanh, leaky_relu, relu


def __get_activation_function(variant, layer):
    if layer == 17:
        return linear
    if variant == 1:
        return elu if layer < 15 else tanh
    elif variant == 2:
        return elu if layer < 16 else tanh
    elif variant == 3:
        return elu
    elif variant == 4:
        return leaky_relu
    elif variant == 5:
        return leaky_relu if layer < 16 else tanh
    elif variant == 6:
        return leaky_relu if layer < 15 else tanh
    elif variant == 7:
        return relu
    elif variant == 8:
        return relu if layer < 16 else tanh
    elif variant == 9:
        return relu if layer < 15 else tanh
    else:
        raise ValueError(f"Activation function variant {variant} not supported.")


def create_model_layers(st_batch, verbose=True, af_variant=1):
    """
    af_variant == 1 -> all elu, 17 linear, 16&15 tanh
    af_variant == 2 -> all elu, 17 linear, 16 tanh
    af_variant == 3 -> all elu, 17 linear
    af_variant == 4 -> all leaky_relu, 17 linear
    af_variant == 5 -> all leaky_relu, 17 linear, 16 tanh
    af_variant == 6 -> all leaky_relu, 17 linear, 16&15 tanh
    af_variant == 7 -> all relu, 17 linear
    af_variant == 8 -> all relu, 17 linear, 16 tanh
    af_variant == 9 -> all relu, 17 linear, 16&15 tanh
    """

    # Half of the channel-dimension size is our output since the other half is the mask
    nbr_output_channels = int(st_batch.shape[-1] / 2)

    if verbose:
        print(f"Input image batch shape: {st_batch.shape}")
        print(f"Number output channels: {nbr_output_channels}")

    # === DOWN (Encoder) ===
    layer_1 = Conv3D(filters=64,
                     kernel_size=(3, 5, 5),
                     strides=(1, 1, 1),
                     padding='same',
                     activation=__get_activation_function(af_variant, 1),
                     name='layer_1',
                     input_shape=st_batch.shape[1:])(st_batch)
    layer_1 = BatchNormalization(name='bn_1')(layer_1)

    layer_2 = ZeroPadding3D(padding=(1, 1, 1))(layer_1)
    layer_2 = Conv3D(filters=128,
                     kernel_size=(3, 4, 4),
                     strides=(1, 2, 2),
                     padding='valid',
                     activation=__get_activation_function(af_variant, 2),
                     name='layer_2')(layer_2)
    layer_2 = BatchNormalization(name='bn_2')(layer_2)

    layer_3 = Conv3D(filters=128,
                     kernel_size=(3, 3, 3),
                     strides=(1, 1, 1),
                     padding='same',
                     activation=__get_activation_function(af_variant, 3),
                     name='layer_3')(layer_2)
    layer_3 = BatchNormalization(name='bn_3')(layer_3)

    layer_4 = ZeroPadding3D(padding=(1, 1, 1))(layer_3)
    layer_4 = Conv3D(filters=256,
                     kernel_size=(3, 3, 3),
                     strides=(1, 2, 2),
                     padding='valid',
                     activation=__get_activation_function(af_variant, 4),
                     name='comb_layer_4')(layer_4)
    layer_4 = BatchNormalization(name='comb_bn_4')(layer_4)

    layer_5 = Conv3D(filters=256,
                     kernel_size=(3, 3, 3),
                     strides=(1, 1, 1),
                     padding='same',
                     activation=__get_activation_function(af_variant, 5),
                     name='layer_5')(layer_4)
    layer_5 = BatchNormalization(name='bn_5')(layer_5)

    layer_6 = Conv3D(filters=256,
                     kernel_size=(3, 3, 3),
                     strides=(1, 1, 1),
                     padding='same',
                     activation=__get_activation_function(af_variant, 6),
                     name='layer_6')(layer_5)
    layer_6 = BatchNormalization(name='bn_6')(layer_6)

    layer_7 = Conv3D(filters=256,
                     kernel_size=(3, 3, 3),
                     strides=(1, 1, 1),
                     padding='same',
                     dilation_rate=(1, 2, 2),
                     activation=__get_activation_function(af_variant, 7),
                     name='layer_7')(layer_6)
    layer_7 = BatchNormalization(name='bn_7')(layer_7)

    layer_8 = Conv3D(filters=256,
                     kernel_size=(3, 3, 3),
                     strides=(1, 1, 1),
                     padding='same',
                     dilation_rate=(1, 2, 2),
                     activation=__get_activation_function(af_variant, 8),
                     name='layer_8')(layer_7)
    layer_8 = BatchNormalization(name='bn_8')(layer_8)

    layer_9 = Conv3D(filters=256,
                     kernel_size=(3, 3, 3),
                     strides=(1, 1, 1),
                     padding='same',
                     dilation_rate=(1, 3, 3),
                     activation=__get_activation_function(af_variant, 9),
                     name='layer_9')(layer_8)
    layer_9 = BatchNormalization(name='bn_9')(layer_9)

    layer_10 = Conv3D(filters=256,
                      kernel_size=(3, 3, 3),
                      strides=(1, 1, 1),
                      padding='same',
                      dilation_rate=(1, 4, 4),
                      activation=__get_activation_function(af_variant, 10),
                      name='layer_10')(Concatenate()([layer_7, layer_9]))
    layer_10 = BatchNormalization(name='bn_10')(layer_10)

    layer_11 = Conv3D(filters=256,
                      kernel_size=(3, 3, 3),
                      strides=(1, 1, 1),
                      padding='same',
                      activation=__get_activation_function(af_variant, 11),
                      name='layer_11')(Concatenate()([layer_6, layer_10]))
    layer_11 = BatchNormalization(name='bn_11')(layer_11)

    layer_12 = Conv3D(filters=256,
                      kernel_size=(3, 3, 3),
                      strides=(1, 1, 1),
                      padding='same',
                      activation=__get_activation_function(af_variant, 12),
                      name='layer_12')(Concatenate()([layer_5, layer_11]))
    layer_12 = BatchNormalization(name='bn_12')(layer_12)

    # === UP (Decoder) ===
    layer_13 = Conv3DTranspose(filters=128,
                               kernel_size=(1, 3, 3),
                               strides=(1, 2, 2),
                               padding='valid',
                               activation=__get_activation_function(af_variant, 13),
                               name='layer_13')(Concatenate()([layer_4, layer_12]))
    layer_13 = Cropping3D(cropping=((0, 0), (0, 1), (0, 1)),
                          name="crop_13")(layer_13)
    layer_13 = BatchNormalization(name='bn_13')(layer_13)

    layer_14 = Conv3D(filters=128,
                      kernel_size=(3, 3, 3),
                      strides=(1, 1, 1),
                      padding='same',
                      activation=__get_activation_function(af_variant, 14),
                      name='layer_14')(Concatenate()([layer_3, layer_13]))
    layer_14 = BatchNormalization(name='bn_14')(layer_14)

    layer_15 = Conv3DTranspose(filters=64,
                               kernel_size=(1, 3, 3),
                               strides=(1, 2, 2),
                               padding='valid',
                               activation=__get_activation_function(af_variant, 15),
                               name='layer_15')(Concatenate()([layer_2, layer_14]))
    layer_15 = Cropping3D(cropping=((0, 0), (0, 1), (0, 1)),
                          name="crop_15")(layer_15)
    layer_15 = BatchNormalization(name='bn_15')(layer_15)

    layer_16 = Conv3D(filters=32,
                      kernel_size=(3, 3, 3),
                      strides=(1, 1, 1),
                      padding='same',
                      activation=__get_activation_function(af_variant, 16),
                      name='layer_16')(Concatenate()([layer_1, layer_15]))
    layer_16 = BatchNormalization(name='bn_16')(layer_16)

    layer_17 = Conv3D(filters=nbr_output_channels,
                      kernel_size=(3, 3, 3),
                      strides=(1, 1, 1),
                      padding='same',
                      activation=__get_activation_function(af_variant, 17),
                      name='layer_17')(layer_16)

    if verbose:
        print(f"Layer1: {layer_1.shape}")
        print(f"Layer2: {layer_2.shape}")
        print(f"Layer3: {layer_3.shape}")
        print(f"Layer4: {layer_4.shape}")
        print(f"Layer5: {layer_5.shape}")
        print(f"Layer6: {layer_6.shape}")
        print(f"Layer7: {layer_7.shape}")
        print(f"Layer8: {layer_8.shape}")
        print(f"Layer9: {layer_9.shape}")
        print(f"Layer10: {layer_10.shape}")
        print(f"Layer11: {layer_11.shape}")
        print(f"Layer12: {layer_12.shape}")
        print(f"Layer13: {layer_13.shape}")
        print(f"Layer14: {layer_14.shape}")
        print(f"Layer15: {layer_15.shape}")
        print(f"Layer16: {layer_16.shape}")
        print(f"Layer17: {layer_17.shape}")

    return layer_17
