from keras.datasets import mnist
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import build_models

####################### General param choices ###########################
encoding_dim = 10
batch_size = 32
num_classes = 10
number_of_epochs = 200

###################### Prepare traina and test Data ######################
model_name = 'mnist_auto_encoder.h5'
img_rows, img_cols, channels = 28, 28, 1
input_shape = (img_rows, img_cols, channels)
(x_train, y_train), (x_test, y_test) = mnist.load_data()
embeded_cov = 1 / 300.0
x_train = x_train.astype('float64')
x_test = x_test.astype('float64')
x_train /= 255.0
x_test /= 255.0
x_train = x_train.reshape(x_train.shape[0], input_shape[0], input_shape[1], input_shape[2])
x_test = x_test.reshape(x_test.shape[0], input_shape[0], input_shape[1], input_shape[2])
########################################################################

######################### Build VAE model ##############################
encoder, decoder, auto_encoder = build_models.get_model(input_shape, encoding_dim)
############################################################################

######################### Make a data generator ############################
def _train_data_generator(x, y, batch_size, encoding_dim, embeded_cov):
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images
    datagen.fit(x)

    gen = datagen.flow(x, y, batch_size=batch_size)

    while True:
        (x, y) = gen.next()
        I_noi = np.random.multivariate_normal(mean=np.zeros(encoding_dim),
                                              cov=np.eye(encoding_dim) * embeded_cov,
                                              size=(x.shape[0]))
        yield ([x + np.random.uniform(0, 0.05, size=x.shape), I_noi], [y, x])
################################################################################
# auto_encoder.load_weights(model_name)
######################### Fit the model ########################################
# Note that we don't want to use y_train so just pass zeros ####################
try:
        auto_encoder.fit_generator(
                _train_data_generator(x_train, 0*y_train, batch_size, encoding_dim, embeded_cov), verbose=1,
                epochs=number_of_epochs, steps_per_epoch=x_train.shape[0]/batch_size + 1,
                validation_data=([x_test, np.zeros((x_test.shape[0], encoding_dim))], [y_test, x_test]))
finally:
        auto_encoder.save_weights(model_name)
