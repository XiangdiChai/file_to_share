try:
    import h5py
except ImportError:
    h5py = None
from keras.datasets import fashion_mnist
import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time
import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from sklearn.metrics import confusion_matrix
import itertools

def plot_confusion_matrix(cm, classes,
                          title=None,
                          cmap=plt.cm.Blues):
    f5 = plt.figure(5)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.xticks([0,1,2,3,4,5,6,7,8,9], classes, rotation=45)
    plt.yticks([0,1,2,3,4,5,6,7,8,9], classes)

    print(cm)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j]),ha="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel='True label',
    plt.xlabel='Predicted label'
    plt.pause(30)

def show_results(nn_model_train): # plot performance over the training epochs
    accuracy     = nn_model_train.history['acc']
    val_accuracy = nn_model_train.history['val_acc']
    loss         = nn_model_train.history['loss']
    val_loss     = nn_model_train.history['val_loss']
    #this can produc the plots, speed of convergence over epochs (slope).
    #Whether the model may have already converged (plateau of the line).
    #Whether the mode may be over-learning the training data (inflection for validation line).
    epochs       = range(len(accuracy))
    #epoch means how many times for training all the sample , it epochs = 5, it means 5 times
    nb_epochs    = len(epochs)

    f2 = plt.figure(2)
    plt.subplot(1,2,1)
    plt.axis((0,nb_epochs,0,1.2))
    plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
    plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1,2,2)
    plt.axis((0,nb_epochs,0,1.2))
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.draw()
    plt.pause(30)

def predict_and_visualize_results(model, input_X, output_Y):
    predicted_classes = model.predict(input_X) # Computes for every image in the test dataset
                                              # a probability distribution over the 10 categories
    predicted_classes = np.argmax(np.round(predicted_classes),axis=1) # Choose the prediction with the highest probability
    correctIndex = np.where(predicted_classes==output_Y)[0]
    incorrectIndex = np.where(predicted_classes!=output_Y)[0]
    print("Found %d correct labels" % len(correctIndex))
    print("Found %d incorrect labels" % len(incorrectIndex))

    # show the some correctly predicted categories
    f3 = plt.figure(3)
    for i, correct in enumerate(correctIndex[:9]):
        plt.subplot(3,3,i+1)
        plt.imshow(test_X[correct].reshape(28,28), cmap='gray')
        plt.title("Predicted {}, Class {}".format(predicted_classes[correct], test_Y[correct]))
        plt.draw()
        plt.pause(0.001)
    plt.tight_layout()
    plt.pause(1)

    # show the some incorrectly predicted categories
    f4 = plt.figure(4)
    for i, incorrect in enumerate(incorrectIndex[:9]):
        plt.subplot(3,3,i+1)
        plt.imshow(test_X[incorrect].reshape(28,28), cmap='gray', interpolation='none')
        plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], test_Y[incorrect]))
        plt.draw()
        plt.pause(0.001)
    plt.tight_layout()
    plt.pause(1)
    cm = confusion_matrix(output_Y, predicted_classes)
    label = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']
    plot_confusion_matrix(cm,label, title='confusion matrix')

## -- Load the Data -- ##
# Label	Description example
#   0	T-shirt/top
#   1	Trouser
#   2	Pullover
#   3	Dress
#   .	.
#   .	.
#   .	.

# -- Get Training+Validation Set & Test Set from dataset -- ##
# fashion mnist: Dataset of 60,000 28x28 grayscale images of 10 fashion categories
# along with a test set of 10,000 images
(trainAndVal_X,trainAndVal_Y), (test_X,test_Y) = fashion_mnist.load_data()
classes  = np.unique(trainAndVal_Y)
nClasses = len(classes)

#---------------------------------------------------------------------------------
#------------------------------ PROBLEM 0: testing -------------------------------
#---------------------------------------------------------------------------------
## -- Display images from training data and test data -- ##
if False: # Set it to "False" if you don't want to see the image display
    f1 = plt.figure(1)
    for i in range(10):
        plt.subplot(1,2,1)
        plt.ion()
        plt.imshow(trainAndVal_X[i,:,:], cmap='gray')
        plt.title("Class ID: {}".format(trainAndVal_Y[i]))

        plt.subplot(1,2,2)
        plt.ion()
        plt.imshow(test_X[i,:,:], cmap='gray')
        plt.title("Class ID: {}".format(test_Y[i]))
        plt.show()
        plt.pause(1)
        plt.cla()
#---------------------------------------------------------------------------------
#-------------------------------- PROBLEM 0: end ---------------------------------
#---------------------------------------------------------------------------------

imgPixelDim = 28 # image pixel dimension

## -- Reshape the images to match the NN input format  -- ##
trainAndVal_X = trainAndVal_X.reshape(-1, imgPixelDim, imgPixelDim, 1) # nb of images: -1 for automatically assigned; pixels: imgPixelDim x imgPixelDim ; nb of channels: 1 for grey scale, 3 for RGB
test_X        = test_X.reshape( -1, imgPixelDim, imgPixelDim, 1)

## -- Convert data type from int8 to float32 -- ##
trainAndVal_X = trainAndVal_X.astype('float32')
test_X = test_X.astype('float32')

## -- Normalize the dtata: rescale the pixel values in range 0 - 1 inclusive for training purposes -- ##
trainAndVal_X = trainAndVal_X / 255.
test_X = test_X / 255.

## -- Change the labels from categorical to one-hot encoding -- ##
# Example: image label 7 becomes [0 0 0 0 0 0 0 1 0] ; The output neurons of the NN will be trained to match the one_hot encoded array
trainAndVal_Y_one_hot = to_categorical(trainAndVal_Y)
test_Y_one_hot  = to_categorical(test_Y)

# Display the change for category label using one-hot encoding
print('Original label             :', trainAndVal_Y[0])
print('After conversion to one-hot encoded:', trainAndVal_Y_one_hot[0])

## -- Split the trainAndVal data into training dataset and validation dataset -- ##
# The moodel is trained over the training dataset
# The validation dataset is used to monitor when the model starts overfitting on the training dataset
train_X,valid_X,train_label,valid_label = train_test_split(trainAndVal_X, trainAndVal_Y_one_hot, test_size=0.2, random_state=13)

#---------------------------------------------------------------------------------
#------------------------------------ PROBLEM ------------------------------------
#---------------------------------------------------------------------------------
if True:
    #------------------------------Hyper Parameters-----------------------------------
    batch_size  = 256   # how many images with their corresponding cotegories to use per
    # per NN weights update step
    epochs      = 15   # how many times to loop over the entire training dataset
    # example: for a batch_size=64 and training dataset size of 48000
    # then each epoch will consist of 48000/64=750 updates of the network weights
    learning_rate = 0.001


    model = Sequential()

    #------------------------------------Architecture---------------------------------
    # slowly extracting values from image
    # 6 means the number of filters in the convolution
    # the kernel size specify the size of two-dimensional convolutional window
    # the kernel is the weight matrix
    # input shape specify the heilght, weight, channel dimensions of the images
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(28,28,1) ,padding='same'))
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',padding='same'))
    model.add(Conv2D(64, kernel_size=(3, 3),activation='relu',padding='same'))
    # maxpooling find the biggest number, in this case 2 by 2 matrix
    model.add(MaxPooling2D((2, 2),padding='same'))

    model.add(MaxPooling2D((2, 2),padding='same'))

    model.add(Dropout(0.25))

    # convolutional = 2D, need to flaten to 1D for dense
    model.add(Flatten())
    #---------------------------------------------------------------------------------
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    #output layer. softmax for probability distribution
    model.add(Dense(nClasses, activation='softmax'))

    model.summary()

    #------------------------------Optimizer-----------------------------------
    opt = keras.optimizers.RMSprop(lr=learning_rate)
    # loss is the degree error(what you got wrong). NN dont maxi accuracy but mini loss
    # Classification accuracy is the number of correct predictions made divided by the
    # total number of predictions made, multiplied by 100 to turn it into a percentage
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=opt, metrics=['accuracy'])


#---------------------------------------------------------------------------------
#---------------------------------- PROBLEM ---- ---------------------------------
#---------------------------------------------------------------------------------

############################################################
## -- Test the performmance of the untrained model over the test dataset -- ##
predicted_classes = model.predict(test_X) # Computes for every image in the test dataset
                                                  # a probability distribution over the 10 categories
predicted_classes = np.argmax(np.round(predicted_classes),axis=1) # Choose the prediction with the highest probability

correctIndex = np.where(predicted_classes==test_Y)[0]
incorrectIndex = np.where(predicted_classes!=test_Y)[0]
print("Found %d correct labels using the untrained model" % len(correctIndex))
print("Found %d incorrect labels using the untrained model" % len(incorrectIndex))

## -- Train the Neural Network -- ##
start_time = time.time()
#the validation data in the data that would be used for validation
#used to train the model
fashion_train_dropout = model.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_label))
# batch size: how many samples we want to group together
trainTime = (time.time() - start_time)
print('Training time = {}'.format(trainTime))

## -- Test the performmance of the trained model over the test dataset -- ##
test_eval = model.evaluate(test_X, test_Y_one_hot, verbose=0)
print('Test loss:', test_eval[0])      # this is the categorical_crossentropy
print('Test accuracy:', test_eval[1])  # the accuracy evaluated by the model during training and testing

show_results(fashion_train_dropout)

predict_and_visualize_results(model, test_X, test_Y)
