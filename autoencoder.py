from os.path import join
from os import listdir
import glob

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

import keras
from keras.preprocessing.image import load_img, img_to_array 
from keras.models import Model, Sequential, load_model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D
from keras.layers import MaxPooling2D, Dropout, UpSampling2D
from keras import backend as K

import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
import seaborn as sns

if __name__ == "__main__":
    
    print("Sklearn version: " + str(sklearn.__version__))
    print("keras version: " + str(keras.__version__))
    print("numpy version: " + str(np.__version__))
    print("pandas version: " + str(pd.__version__))

    image_dir = './dataset/'
    img_paths = [join(image_dir, filename) for filename in listdir(image_dir)]

    ## Loades Image paths are not sorted, which we can sort them using the below functions
    import re

    def atoi(text):
        return int(text) if text.isdigit() else text

    def natural_keys(text):
        '''
        alist.sort(key=natural_keys) sorts in human order
        http://nedbatchelder.com/blog/200712/human_sorting.html
        (See Toothy's implementation in the comments)
        '''
        return [ atoi(c) for c in re.split(r'(\d+)', text) ]


    # Sort the images according to their numbers

    img_paths.sort(key= natural_keys) 


    # Function to Read and Prep Images for Modeling

    image_size = 224
    channels = 3

    def read_and_prep_images(img_paths, img_height=image_size, img_width=image_size, channels=channels):
        
        image_list = np.zeros((len(img_paths), img_height, img_width, channels))
        
        for i, fig in enumerate(img_paths):
            
            img = load_img(fig,  target_size=(img_height, img_width))
            img_array = img_to_array(img).astype('float32')
            x = img_array / 255.0 # Normalize between 0 and 1
            image_list[i] = x
        
        return image_list


    # Call the function to load and preprocess images
    img_data = read_and_prep_images(img_paths)

    print("Load images successfull.\n")

    # Split the data into train and validation sets

    print("Spliting into train and validation sets.\n")

    def train_val_split(x):
        rnd = np.random.RandomState(seed=42)
        perm = np.random.permutation(len(x))
        train_idx = perm[:int(0.7 * len(x))] # 70% data for trainining and 30% images for validation
        val_idx = perm[int(0.7 * len(x)):]
        return x[train_idx], x[val_idx]

    x_train, x_val = train_val_split(img_data)
    print("Train shape: {} \nTest shape: {} " .format(x_train.shape, x_val.shape))

    # Create a class for our Autoencoder network

    class Autoencoder():
        def __init__(self):
            self.img_rows = 224
            self.img_cols = 224
            self.channels = 3
            self.img_shape = (self.img_rows, self.img_cols, self.channels)
            
            optimizer = Adam(lr=0.001)
            
            self.autoencoder_model = self.build_model()
            self.autoencoder_model.compile(loss='mse', optimizer=optimizer)
            self.autoencoder_model.summary()
        
        def build_model(self):
            input_layer = Input(shape=self.img_shape)
                    
            # encoder
            h = Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)
            h = MaxPooling2D((2, 2), padding='same')(h)
            
            h = Conv2D(8, (3, 3), activation='relu', padding='same')(h)
            h = MaxPooling2D((2, 2), padding='same')(h)
            
            # decoder
            h = Conv2D(8, (3, 3), activation='relu', padding='same')(h)
            h = UpSampling2D((2, 2))(h)
            
            h = Conv2D(64, (3, 3), activation='relu', padding='same')(h)
            h = UpSampling2D((2, 2))(h)
            
            output_layer = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(h)
            
            return Model(input_layer, output_layer)
        
        def train_model(self, x_train, x_val, epochs, batch_size=20):
            early_stopping = EarlyStopping(monitor='val_loss',
                                        min_delta=0,
                                        patience=5,
                                        verbose=1, 
                                        mode='auto')
            history = self.autoencoder_model.fit(x_train, x_train,
                                                batch_size=batch_size,
                                                epochs=epochs,
                                                validation_data=(x_val, x_val),
                                                callbacks=[early_stopping])
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('Model loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Test'], loc='upper left')
            plt.show()
        
        def eval_model(self, x_test):
            preds = self.autoencoder_model.predict(x_test)
            return preds
    
    # Create a object from our previous Autoencoder class and fit it on trian data

    ae = Autoencoder()
    ae.train_model(x_train, x_val, epochs=50, batch_size=20)

    #Save the model
    ae.autoencoder_model.save('ae_model.hdf5')

    # Let’s visualize some restorations.

    restored_imgs = ae.autoencoder_model.predict(x_val)

    # Visualize the reconstructed inputs and the encoded representations. use Matplotlib

    n = 6 # How many origina images we will display
    plt.figure(figsize= (15,8))

    for i in range(n):
        # Display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow((x_val[i] * 255).astype(np.uint8))
        ax.set_title("Original images")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        # Display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow((restored_imgs[i] * 255).astype(np.uint8))
        ax.set_title("Reconstructed images")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.savefig("./reconstructed_images.png")
    plt.show()
    

    # Compressed Representations

    compressed_layer = 4
    
    # Extract the encoder
    get_4th_layer_output = K.function([model.layers[0].input], [model.layers[compressed_layer].output])

    # Encode the testing set
    compressed_images = get_4th_layer_output([x_val])[0]
    
    #flatten compressed representation to 1 dimensional array
    compressed_images = compressed_images.reshape(-1, 56*56*8)

    distortions = []
    cl = range(1,15)
    for k in cl:
        kmeanModel = KMeans(n_clusters=k)
        kmeanModel.fit(compressed_images)
        distortions.append(kmeanModel.inertia_)

    # Plotting the distortions of K-Means
    plt.figure(figsize=(16,8))
    plt.plot(cl, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.savefig("./scree_plot.png")
    plt.show()

    # # Cluster the testing set
    n_clusters = 6
    kmeans = KMeans(n_clusters=n_clusters, n_init=25, max_iter = 600, random_state= 123)
    clusters = kmeans.fit_predict(compressed_images)

    # Visualize images with their respective clusters
    # Plot the actual pictures grouped by clustering

    fig = plt.figure(figsize=(12,15))
    n_clusters=6

    for cluster in np.arange(n_clusters):
        cluster_member_indices = np.where(clusters == cluster)[0]
        print("There are %s members in cluster %s" % (len(cluster_member_indices), cluster))
        
    #random_member = np.random.choice(cluster_member_indices, size = num)
    # x_val[random_member[i],:,:,:]
    # take 3 images from each cluster
    for c, val in enumerate(x_val[cluster_member_indices,:,:,:][0:3]):
            i = 3*cluster+c+1 # max 18 plots
            fig.add_subplot(6, 3, i)
            plt.imshow(val)
            plt.xticks([])
            plt.yticks([])
            plt.xlabel('cluster: '+str(cluster))
            
    plt.tight_layout()
    plt.savefig("clustering_img.png", bbox_inches='tight')

    plt.show()




