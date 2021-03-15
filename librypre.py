import numpy as np
import librys
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D, Flatten, BatchNormalization, Dropout
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.losses import sparse_categorical_crossentropy, categorical_crossentropy, CategoricalCrossentropy
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications import VGG16
from tensorflow.compat.v1 import ConfigProto, InteractiveSession

def inceptionV3Modeling(input_shape=(150,150,3), output_label=1, verbose=False):
    # https://github.com/tensorflow/tensorflow/issues/24828
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

    model_path = librys.pathDir().pmDir()+"inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
    pretrained_model = InceptionV3(input_shape=input_shape,
                                    include_top=False,
                                    weights=None)

    pretrained_model.load_weights(filepath=model_path)

    for layer in pretrained_model.layers:
        layer.trainable = False

    '''
    Une pratique courante consiste à utiliser la sortie de la toute dernière couche avant l'opération d'aplatissement, appelée "couche de goulot d'étranglement".
     Le raisonnement ici est que les couches entièrement connectées suivantes seront trop spécialisées pour la tâche sur laquelle le réseau a été formé,
     et donc les fonctionnalités apprises par ces couches ne seront pas très utiles pour une nouvelle tâche. Les caractéristiques des goulots d'étranglement conservent cependant une grande généralité.
    '''
    last_layer_output = pretrained_model.get_layer('mixed7').output

    flat = Flatten()(last_layer_output)
    dense1 = Dense(units=512, activation=tf.nn.relu)(flat)
    dense2 = Dense(units=512, activation=tf.nn.relu)(dense1)
    drop1 = Dropout(0.2)(dense2)
    output = Dense(units=output_label, activation=tf.nn.softmax)(drop1)

    model = Model(inputs=pretrained_model.input, outputs=output)

    '''En substance, le lissage des étiquettes aidera notre modèle à s'entraîner autour de données mal étiquetées et par conséquent à améliorer sa robustesse et ses performances. '''
    #label_smoothing(y_true, y_pred):
    #return categorical_crossentropy(y_true, y_pred, label_smoothing=0.1)

    model.compile(optimizer=RMSprop(learning_rate=0.0001, momentum=0.9), loss=CategoricalCrossentropy(label_smoothing=0.1), metrics=["accuracy"])

    if verbose:
        print(model.summary())

    return model

def vgg16Modeling(input_shape=(150,150,3), output_label=1, verbose=False):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    model_path = librys.pathDir().pmDir()+"vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
    pretrained_model = VGG16(input_shape=input_shape,
                                    include_top=False,
                                    weights=None)

    pretrained_model.load_weights(filepath=model_path)

    for layer in pretrained_model.layers:
        layer.trainable = False

    last_layer_ouput = pretrained_model.get_layer('block5_pool').output
    flat = Flatten()(last_layer_ouput)
    dense1 = Dense(units=4096, activation=tf.nn.relu)(flat)
    drop1 = Dropout(0.5)(dense1)
    dense2 = Dense(units=4096, activation=tf.nn.relu)(drop1)
    drop2 = Dropout(0.5)(dense2)
    output = Dense(units=output_label, activation=tf.nn.softmax)(drop2)

    model = Model(inputs=pretrained_model.input, outputs=output)

    model.compile(optimizer=Adam(learning_rate=0.0001), loss=CategoricalCrossentropy(label_smoothing=0.1), metrics=["accuracy"])

    if verbose:
        print(model.summary())

    return model

def resNet50Modeling(input_shape=(150,150,3), output_label=1, verbose=False):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    model_path = librys.pathDir().pmDir()+"resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"
    pretrained_model = ResNet50(input_shape=input_shape,
                                    include_top=False,
                                    weights=None)

    # pretrained_model.load_weights(filepath=model_path)
    # for layer in pretrained_model.layers:
    #     layer.trainable = False

    last_layer_ouput = pretrained_model.output
    gavg_pool = GlobalAveragePooling2D()(last_layer_ouput)
    drop1 = Dropout(0.7)(gavg_pool)
    output = Dense(units=output_label, activation=tf.nn.softmax)(drop1)

    model = Model(inputs=pretrained_model.input, outputs=output)

    model.compile(optimizer=Adam(learning_rate=0.0001), loss=categorical_crossentropy, metrics=["accuracy"])

    if verbose:
        print(model.summary())

    return model