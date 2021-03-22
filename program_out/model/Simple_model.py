#Importation des modules necessaire dans Keras
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K

#La definition de la model de entrainement. 
class Model_cnn:
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()
        inputShape = (height, width, depth)
        #Dans Keras documents: 'https://github.com/keras-team/keras/issues/1921'
        #For Convolution2D layers with dim_ordering=“th” (the default), use axis=1,
        #For Convolution2D layers with dim_ordering=“tf”, use axis=-1 (i.e. the default).
        chanDim = -1

        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1
        
        #Convolution couche 1 Input = 96,96,3 Output = 96,96,32
        model.add(Conv2D(32,(3,3),padding="same", activation="relu", input_shape=inputShape))
        #BatchNoramlisation
        model.add(BatchNormalization(axis=chanDim))
        #Maxpolling Input = 96 96 32 Output = 32 32 32
        model.add(MaxPooling2D(pool_size=(3,3)))
        #Set dropout rate at 1/4
        model.add(Dropout(0.25))
        
        #Convolution couche 2 Input = 32 32 32 Output = 32 32 64
        model.add(Conv2D(64, (3,3), padding="same", activation="relu"))
        model.add(BatchNormalization(axis=chanDim))
        #Convolution couche 3 Input = 32 32 64 Output = 32 32 64
        model.add(Conv2D(64, (3,3), padding="same", activation="relu"))
        model.add(BatchNormalization(axis=chanDim))
        #Maxpooling Input = 32 32 64 Output = 16 16 64
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.25))
        
        #Convolution couche 4 Input = 16 16 64 Output = 16 16 128
        model.add(Conv2D(128, (3,3), padding="same", activation="relu"))
        model.add(BatchNormalization(axis=chanDim))
        #Convolution couche 5 Input = 16 16 128 Output = 16 16 128
        model.add(Conv2D(128, (3,3), padding="same", activation="relu"))
        model.add(BatchNormalization(axis=chanDim))
        #Maxpooling Input = 16 16 128 Output = 8 8 128
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.25))
        
        #Flatten couche Input = 8 8 128 Output = 8192
        model.add(Flatten())
        #Dense couche Input = 8192 Output = 1024
        model.add(Dense(1024,activation="relu"))
        model.add(BatchNormalization())
        #Dropout pour diminuer overfitting
        model.add(Dropout(0.5))
        #sigmoid pour regulariser output en 0/1 Input = 1024, Output = 2
        model.add(Dense(classes, activation="sigmoid"))

        return model
