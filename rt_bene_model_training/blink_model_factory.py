import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization, Average, ReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


def create_model_base(backbone, input_shape):
    if backbone == 'mobilenetv2':
        base = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet', input_tensor=None,
                                                 input_shape=input_shape, pooling='avg')

    elif backbone == 'densenet121':
        base = tf.keras.applications.DenseNet121(include_top=False, weights='imagenet', input_tensor=None,
                                                 input_shape=input_shape, pooling='avg')

    elif backbone == 'resnet50':
        base = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_tensor=None,
                                              input_shape=input_shape, pooling='avg')
    else:
        raise Exception('Wrong backbone')

    for layer in base.layers:
        layer.trainable = True

    main_input = Input(shape=input_shape)
    main_output = base(main_input)

    final_fc_layer = Dense(512, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(main_output)
    final_fc_layer = BatchNormalization(epsilon=1e-3, momentum=0.999)(final_fc_layer)
    final_fc_layer = ReLU(6.)(final_fc_layer)
    final_fc_layer = Dropout(0.6)(final_fc_layer)
    output_tensor = Dense(1, activation='sigmoid')(final_fc_layer)  # probability

    model = Model(inputs=main_input, outputs=output_tensor)

    return model


def create_model(backbone, input_shape, lr, metrics):
    base = create_model_base(backbone, input_shape)

    # define the 2 inputs (left and right eyes)
    left_input = Input(shape=input_shape)
    right_input = Input(shape=input_shape)

    # get the 2 outputs using shared layers
    out_left = base(left_input)
    out_right = base(right_input)

    # average the predictions
    merged = Average()([out_left, out_right])
    model = Model(inputs=[right_input, left_input], outputs=merged)

    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=lr), metrics=metrics)
    model.summary()

    return model
