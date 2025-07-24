# Import libraries
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import TruncatedSVD

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout

def run_dimension_reduction(X_train, X_test, y_train, technique, number_components, epochs, batchs):
    """Apply dimension reduction"""

    if technique == 'pca':
        print(f"Applying PCA with {number_components} components...")
        pca = PCA(n_components=number_components)
        X_train_reduced = pca.fit_transform(X_train)
        X_test_reduced = pca.transform(X_test)

        cols = [(x * 1000) - 1 for x in range(2, number_components + 2)]
        X_train_reduced = pd.DataFrame(X_train_reduced, columns=cols)


        cols = [(x * 1000) - 1 for x in range(2, number_components + 2)]
        X_test_reduced = pd.DataFrame(X_test_reduced, columns=cols)

        pca = PCA().fit(X_train)
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('number of components')
        plt.ylabel('cumulative explained variance')

    elif technique == 'kernel_pca':
        print(f"Applying Kernel PCA with {number_components} components...")
        kernel_pca = KernelPCA(n_components=number_components, kernel='rbf')
        X_train_reduced = kernel_pca.fit_transform(X_train, y_train)
        X_test_reduced = kernel_pca.transform(X_test)

        cols = [(x * 1000) - 1 for x in range(2, number_components + 2)]
        X_train_reduced = pd.DataFrame(X_train_reduced, columns=cols)

        cols = [(x * 1000) - 1 for x in range(2, number_components + 2)]
        X_test_reduced = pd.DataFrame(X_test_reduced, columns=cols)


    elif technique == 'svd':
        print(f"Applying SVD with {number_components} components...")
        SVD = TruncatedSVD(n_components=number_components)
        X_train_reduced = SVD.fit_transform(X_train, y_train)
        X_test_reduced = SVD.transform(X_test)

        cols = [(x * 1000) - 1 for x in range(2, number_components + 2)]
        X_train_reduced = pd.DataFrame(X_train_reduced, columns=cols)


        cols = [(x * 1000) - 1 for x in range(2, number_components + 2)]
        X_test_reduced = pd.DataFrame(X_test_reduced, columns=cols)


    elif technique == 'auto_encoder':
        print(f"Applying Auto encoder...")
        # This code comes from https://www.analyticsvidhya.com/blog/2021/06/dimensionality-reduction-using-autoencoders-in-python/


        class AutoEncoders(Model):

            def __init__(self, output_units):
                super().__init__()
                self.encoder = Sequential(
                    [
                        Dense(1024, activation="relu", input_shape=(1447,)),
                        Dense(750, activation="relu"),
                        Dense(600, activation="relu"),
                        Dense(500, activation="relu")  # Bottleneck with 500 components
                    ]
                )

                self.decoder = Sequential(
                    [
                        Dense(600, activation="relu"),
                        Dense(750, activation="relu"),
                        Dense(1024, activation="relu"),
                        Dense(1447, activation="sigmoid")  # Or "linear" depending on input scale
                    ]
                )

            def call(self, inputs):
                encoded = self.encoder(inputs)
                decoded = self.decoder(encoded)
                return decoded

        auto_encoder = AutoEncoders(len(X_train.columns))

        auto_encoder.compile(loss='mae', metrics=['mae'], optimizer='adam')

        history = auto_encoder.fit(
            X_train,
            X_train,
            epochs=epochs,
            batch_size=batchs,
            validation_data=(X_test, X_test)
        )

        encoder_layer = auto_encoder.get_layer('sequential')
        X_train_reduced = pd.DataFrame(encoder_layer.predict(X_train))
        X_train_reduced = X_train_reduced.add_prefix('feature_')


        X_test_reduced = pd.DataFrame(encoder_layer.predict(X_test))
        X_test_reduced = X_test_reduced.add_prefix('feature_')

    else:
        raise ValueError(f"Unknown dimension reduction technique: {technique}")

    return X_train_reduced, X_test_reduced
