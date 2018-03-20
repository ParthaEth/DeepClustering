import numpy as np
from keras.layers import Dense, Input, Reshape, Add, Flatten, Lambda, LeakyReLU, \
    Conv2D, MaxPooling2D, BatchNormalization, UpSampling2D, Concatenate
import tensorflow as tf
from keras.models import Model
from keras.optimizers import Adamax
import keras.backend as K

# K.set_floatx('float64')

################### Build GMM pdf ######################################
num_classes = 10
tfd = tf.contrib.distributions
mix = 1.0/num_classes

components=[]
for i in range(num_classes):
    loc = np.zeros((num_classes,))
    loc[i] = 1
    components.append(tfd.MultivariateNormalDiag(loc=loc, scale_diag=np.ones((num_classes,))/10.0))

mix_gauss = tfd.Mixture(cat=tfd.Categorical(probs=mix*np.ones((num_classes,))), components=components)


def gmm_likelihood(y_true, y_pred):
    """y_true does not matter kept only since keras expectes loss to have this form"""
    return -K.log(tf.cast(mix_gauss.prob(tf.cast(y_pred, tf.float64)), tf.float32) + 1e-10) # We need to maximize this!!


def mse_plus_likelihood(y_true, y_pred):
    # zeroing out those for which y_true is all zeros
    mse = K.sum(K.sqare(y_true-y_pred), axis=-1) * K.sum(y_true, axis=-1)
    return mse + gmm_likelihood(y_true, y_pred)


def get_model(input_shape, encoding_dim):
    # Build encoder
    I_en = Input(shape=input_shape, name="input_image")

    # For MNIST
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(I_en)
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    # x = Dropout(0.25)(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    # x = Dropout(0.25)(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)

    x = Flatten()(x)

    x = Dense(128, activation='relu')(x)
    # x = Dropout(0.5)(x)

    x = Dense(encoding_dim, activation='tanh')(x)
    x = Lambda(lambda x: x*1.5)(x)

    encoder = Model(input=I_en, output=x, name='encoder')
    encoder.compile(optimizer='SGD', loss='mse')
    encoder.summary()

    # Build decoder
    I_enc = Input(shape=(encoding_dim,), name="input_latent_image")
    I_noi = Input(shape=(encoding_dim,), name="noise")
    I_de = Add()([I_enc, I_noi])

    # For MNIST
    x = Dense(196, activation='linear')(I_de)
    x = LeakyReLU(alpha=.3)(x)
    batch_1_start = Reshape(target_shape=(7, 7, 4))(x)
    x = Conv2D(64, (3, 3), padding='same', activation='linear')(batch_1_start)
    x = LeakyReLU(alpha=.3)(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(64, (3, 3), padding='same', activation='linear')(x)
    x = LeakyReLU(alpha=.3)(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(64, (3, 3), padding='same', activation='linear')(x)
    x = LeakyReLU(alpha=.3)(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(64, (3, 3), padding='same', activation='linear')(x)
    x = LeakyReLU(alpha=.3)(x)
    x = BatchNormalization(axis=-1)(x)
    x = UpSampling2D((2, 2))(x)

    batch_1_start = UpSampling2D((2, 2))(batch_1_start)
    x = Concatenate(axis=-1)([batch_1_start, x])
    batch_2_start = Conv2D(32, (3, 3), padding='same', activation='linear')(x)
    x = LeakyReLU(alpha=.3)(batch_2_start)
    x = Conv2D(32, (3, 3), padding='same', activation='linear')(x)
    x = LeakyReLU(alpha=.3)(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(32, (3, 3), padding='same', activation='linear')(x)
    x = LeakyReLU(alpha=.3)(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(32, (3, 3), padding='same', activation='linear')(x)
    x = LeakyReLU(alpha=.3)(x)
    x = BatchNormalization(axis=-1)(x)
    x = UpSampling2D((2, 2))(x)

    batch_1_start = UpSampling2D((2, 2))(batch_1_start)
    batch_2_start = UpSampling2D((2, 2))(batch_2_start)
    x = Concatenate(axis=-1)([batch_1_start, batch_2_start, x])
    batch_3_start = Conv2D(16, (3, 3), padding='same', activation='linear')(x)
    x = LeakyReLU(alpha=.3)(batch_3_start)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(16, (3, 3), padding='same', activation='linear')(x)
    x = LeakyReLU(alpha=.3)(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(16, (3, 3), padding='same', activation='linear')(x)
    x = LeakyReLU(alpha=.3)(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(16, (3, 3), padding='same', activation='linear')(x)
    x = LeakyReLU(alpha=.3)(x)
    x = BatchNormalization(axis=-1)(x)

    x = Conv2D(1, (3, 3), padding='same', activation='sigmoid')(x)

    decoder = Model(input=[I_enc, I_noi], output=x, name='decoder')
    decoder.compile(optimizer='SGD', loss='mse')
    decoder.summary()

    # Build the whole autoencoder
    I = Input(shape=input_shape, name="input_image")
    encoded = encoder(I)
    reconstructed = decoder([encoded, I_noi])
    auto_encoder = Model(input=[I, I_noi], output=[encoded, reconstructed], name='encoder_decoder')
    auto_encoder.compile(optimizer=Adamax(lr=1e-4), loss=[mse_plus_likelihood, "mse"],
                         loss_weights=[0.9, 0.1])
    # auto_encoder.compile(optimizer=Adam(lr=1e-4), loss=[max_likelihood, 'mse'], loss_weights=[0.01, 0.99])
    # auto_encoder.compile(optimizer=rmsprop(lr=1e-2), loss=[max_likelihood, 'mse'], loss_weights=[0.01, 0.99])
    auto_encoder.summary()

    return encoder, decoder, auto_encoder
