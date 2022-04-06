import os
import cv2
import requests
import numpy as np
from PIL import Image
from io import BytesIO

from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping

from matplotlib import pyplot as plt

from .model.model_sample import vgg16_model_v0

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class ModelManager:
    def __init__(self, config_model=None):
        """
        :param config_model: dictionary with configuration model epochs and batch_size
        """
        if config_model is None:
            config_model = {"epochs": 40, "batch_size": 64}

        self.history = None
        self.models = vgg16_model_v0()
        self.epochs = config_model["epochs"]
        self.batch = config_model["batch_size"]
        self.callbacks = ModelCheckpoint('model/weights/model_age_weights.h5', save_weights_only=True,
                                         monitor='val_loss', save_best_only=True, mode='min')
        self.early_stop = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1)

    def load_model(self, weights_path: str):
        """
        :param weights_path: neural network weights location
        """
        self.models.load_weights(weights_path)

    def fit(self, train_sample: list, validate_sample: list):
        """
        :param train_sample: train samples list
        :param validate_sample: validate samples list
        """
        self.history = self.models.fit(train_sample,
                                       verbose=1,
                                       epochs=self.epochs,
                                       batch_size=self.batch,
                                       validation_data=validate_sample,
                                       callbacks=[self.callbacks, self.early_stop])

    @staticmethod
    def fit_transform(path_file: str) -> object:
        """
        :type path_file: path file with the image to inference
        """
        image = Image.open(path_file).convert('L')
        image = np.asarray(image.resize((48, 48)), dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = image.reshape(1, 48, 48, 3)
        return image

    @staticmethod
    def fit_transform_url(url: str) -> object:
        response = requests.get(url)
        image = Image.open(BytesIO(response.content)).convert('L')
        image = np.asarray(image.resize((48, 48)), dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = image.reshape(1, 48, 48, 3)
        return image

    @staticmethod
    def show_image(image: object, inference: int) -> object:
        """
        :param image: np array with the image used in prediction
        :param inference: prediction age
        """
        plt.title(f'Predição: {inference} anos')
        plt.imshow(image[0])
        plt.show()

    def predict(self, path_file: str, verbose: bool = False) -> int:
        """
        :param verbose: show the prediction and image in plot
        :type path_file: location image to prediction
        """
        image = self.fit_transform(path_file)
        inference = self.models.predict(image).astype(np.int)[0]
        predict_age = inference.astype(np.int)[0]
        if verbose:
            self.show_image(image, predict_age)
        return predict_age

    def predict_url(self, url: str, verbose: bool = False) -> int:
        """
        :param verbose: show the prediction and image in plot
        :type url: location image (url)
        """
        image = self.fit_transform_url(url)
        inference = self.models.predict(image).astype(np.int)[0]
        predict_age = inference.astype(np.int)[0]
        if verbose:
            self.show_image(image, predict_age)
        return predict_age
