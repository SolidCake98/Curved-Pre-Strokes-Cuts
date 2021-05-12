import glob
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.signal import argrelextrema
from tensorflow import keras
from itertools import *
from read_smp import Samples
from slider_window import WindowSlider, Detector

PROJECT_ROOT_DIR = "."
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images/le_net_model")
os.makedirs(IMAGES_PATH, exist_ok=True)


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


def look_at_sample(im, label, r):
    plt.imshow(im, cmap='gray')
    plt.xlabel(r)
    save_fig(label)
    plt.close()


# разбиение с помощью вертикальной гистограммы
class HSeparate:

    def __init__(self, samples, model):
        self.smp = Samples()
        self.smp.loadFromSmp(samples)
        self.model = keras.models.load_model(model)
        self.marks = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
                      'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
                      'W', 'X', 'Y', 'Z']

    def horizontal_histogram(self, image):
        h_histogram = []
        w, h = image.shape
        for i in range(0, w):
            x_value = 0
            for j in range(0, h):
                if image[i, j] != 255:
                    x_value += 1
            h_histogram.append(x_value)
        return h_histogram

    def vertical_histogram(self, image):
        v_histogram = []
        w, h = image.shape
        for i in range(0, h):
            y_value = 0
            for j in range(0, w):
                if image[j, i] != 255:
                    y_value += 1
            try:
                if v_histogram[i - 1] == y_value:
                    v_histogram.append(y_value - 1)
                else:
                    v_histogram.append(y_value)
            except:
                v_histogram.append(y_value)
        return v_histogram

    def find_possible_letters(self, i):
        im = self.smp.imgs[i]
        histogram = self.vertical_histogram(im)
        min_h = argrelextrema(np.array(self.vertical_histogram(im)), np.less)
        n_min = [0]
        for i in min_h[0]:
            if histogram[i] <= 5:
                n_min.append(i)
        i = 0
        while i < len(n_min) - 1:
            if n_min[i + 1] - n_min[i] <= im.shape[1] / 9:
                del n_min[i + 1]
            else:
                i += 1
        if im.shape[1] - n_min[i] <= im.shape[1] / 9:
            n_min[i] = im.shape[1]
        else:
            n_min.append(im.shape[1])
        return n_min, im.shape[1] / 9

    def make_prediction(self):
        for index in range(len(self.smp.imgs)):
            l, gap = self.find_possible_letters(index)
            img = self.smp.imgs[index]
            possibles = []
            otr = []
            for comb in combinations(l, 2):
                img1 = img[:, comb[0]:comb[1]]
                img1 = cv2.resize(img1, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
                otr.append(comb)
                possibles.append(img1)

            x_test = np.array(possibles).astype('float32')
            x_test /= 255
            x_test = np.asarray(x_test)
            x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')
            prediction = self.model.predict(x_test)
            prediction_proba = self.model.predict_proba(x_test)
            r = self.predict_n(otr, prediction_proba, prediction)
            yield r
            look_at_sample(img, self.smp.labels[index] + str(index), r)

    def predict_n(self, otr, prediction_proba, prediction):
        n = 0
        k = 0
        recog = []
        while n != max(otr, key=lambda x: x[1])[1]:
            max_predict_otr = otr[k][1]
            max_predict_v = k
            while n == otr[k][0] and k != len(otr) - 1:
                if max(prediction_proba[k]) >= max(prediction_proba[max_predict_v]):
                    max_predict_otr = otr[k][1]
                    max_predict_v = k
                k += 1
            n = max_predict_otr
            while otr[k][0] != n and k != len(otr) - 1:
                k += 1
            recog.append(self.marks[np.argmax(prediction[max_predict_v])])

        return recog

    def predict_2(self, otr, prediction_proba, prediction, gap):
        recog = ["a", "a"]
        ms = max(otr, key=lambda x: x[1])[1]
        otr_fr_zero = list((n for n in otr if n[0] == 0 and ms - n[1] >= gap))
        max_pred = 0
        for i in range(len(otr_fr_zero)):
            second_otr = list((k, n) for n, k in enumerate(otr) if k[0] == otr_fr_zero[i][1] and k[1] == ms)
            pred = (max(prediction_proba[i]) + max(prediction_proba[second_otr[0][1]]))
            if pred >= max_pred:
                max_pred = pred
                recog[0] = self.marks[np.argmax(prediction[i])]
                recog[1] = self.marks[np.argmax(prediction[second_otr[0][1]])]
        return recog
