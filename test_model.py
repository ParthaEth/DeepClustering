from keras.datasets import mnist
import matplotlib.pyplot as plt
import build_models
import numpy as np

####################### General param choices ###########################
encoding_dim = 10
batch_size = 8
num_classes = 10
number_of_epochs = 200

###################### Prepare traina and test Data ######################
model_name = 'mnist_auto_encoder.h5'
img_rows, img_cols, channels = 28, 28, 1
input_shape = (img_rows, img_cols, channels)
(x_train, y_train), (x_test, y_test) = mnist.load_data()
embeded_cov = 1 / 20.0
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

encoder, decoder, auto_encoder = build_models.get_model(input_shape, encoding_dim)
auto_encoder.load_weights(model_name)

for i in range(num_classes):
    mean = np.zeros(encoding_dim)
    mean[i] = 1
    # I_enc = np.random.multivariate_normal(mean=mean,
    #                                       cov=np.eye(encoding_dim) * embeded_cov,
    #                                       size=(1,))
    I_enc = encoder.predict(x_test[i:i+1, :, :, :])
    print I_enc
    recon_img = decoder.predict([I_enc, np.zeros((1, encoding_dim))])
    plt.imshow(recon_img[0, :, :, 0])
    plt.show()