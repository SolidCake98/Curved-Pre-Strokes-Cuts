import math

import cv2
import os
import numpy as np
from tensorflow import keras
from read_smp import Samples


# разбиение с помощью скользящего окна
class WindowSlider:

    def __init__(self, window_size, window_step, scale_step):
        self.scale_step = scale_step
        self.window_size = window_size
        self.window_step = window_step

    def scaler(self, w_size, img):
        yield w_size
        while ((
                img.shape[1] > w_size[0] * self.scale_step)):
            w_size = (math.ceil(w_size[0] * self.scale_step), w_size[1])
            yield w_size

    def slider(self, img, ws):
        xs, ys = self.window_step
        ww, wh = ws
        h, w = img.shape

        if ys != 0:
            for y in range(0, h - wh, ys):
                for x in range(0, w - ww, xs):
                    yield (x, y, img[y:y + wh, x:x + ww])
        else:
            y = 0
            for x in range(0, w - ww, xs):
                yield (x, y, img[y:y + wh, x:x + ww])

    def run(self, img):
        i = 1
        for w_size in self.scaler(w_size=self.window_size, img=img):
            for (x, y, window) in self.slider(img=img, ws= w_size):
                self.proc_slide(window, (x, y), i)
                i += 1

    def proc_slide(self, window, pos=(0, 0), index=0):
        cv2.imwrite(
            "result/%04i-%.3f-%ix%i_%i-%i.png" % (index, 1, window.shape[1], window.shape[0], pos[0], pos[1]),
            window)
        print("result/%04i-%.3f-%ix%i_%i-%i.png" % (index, 1, window.shape[1], window.shape[0], pos[0], pos[1]))


class Detector(WindowSlider):
    def __init__(self, model, window_size, window_step=(1, 0), scale_step=1.2, model_threshold=0):
        self.model = keras.models.load_model(model)
        self.marks = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
                      'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
                      'W', 'X', 'Y', 'Z']
        self.model_threshold = model_threshold
        self.boxes = []
        super().__init__(window_size, window_step, scale_step)

    def search(self, img):
        # img = cv2.imread(f,cv2.IMREAD_GRAYSCALE) # 100x40
        self.boxes = []
        self.run(img)
        self.clean_boxes()
        return self.boxes

    def proc_slide(self, window, pos=(0, 0), index=0):
        h, w = window.shape
        img1 = cv2.resize(window, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
        img1 = img1 / 255
        img1 = np.asarray(img1)
        img1 = img1.reshape(1, 28, 28, 1).astype('float32')
        o = max(self.model.predict_proba(img1)[0])
        mark = self.marks[np.argmax(self.model.predict(img1)[0])]
        if o > self.model_threshold:
            x, y = pos
            rx, ry, rx2, ry2 = x, y, x + w, y + h
            self.boxes.append([rx, ry, rx2, ry2, abs(o), mark])

    def clean_boxes(self):
        # расстояние между точками
        def dist(x, y):
            return np.sqrt(np.sum(np.square(np.array(center(x)) - np.array(center(y)))))

        # вычисляем точку-центр рамки
        def center(frame):
            (x, y, x2, y2, _, _) = frame
            return [x + ((x2 - x) // 2), y2 + ((y - y2) // 2)]

        # очистка списка найденных рамок
        def clean(X, i=0, max_dist=35):
            # выбираем основной элемент X[i]
            j = i + 1  # и, следующий за ним, элемент X[j]
            while (j < len(X)):  # для всех элементов j (!= i)
                d = dist(X[i], X[j])  # считаем расстояние между точками i,j
                if (d < max_dist):  # i,j в одном кластере
                    if (X[i][4] > X[j][4]):  # сравниваем рейтинг
                        del X[j]  # удаляем элемент j
                    else:
                        del X[i]  # удаляем основной элемент i
                        X = clean(X, i=i, max_dist=max_dist)  # рекурсивно повторяем уже без элемента i
                        break
                else:
                    j += 1  # текущие точки i,j в разных кластерах, берём для проверки следующую точку j+1

            # найденный основной элемент i
            # имеет наибольший рейтинг в своем кластере
            if (i < (len(X) - 1)):  # если есть ещё точки в других, относительно X[i], кластерах
                X = clean(X, i=i + 1, max_dist=max_dist)  # то выполняем проверку

            return X
        self.boxes = clean(self.boxes, max_dist=min(self.window_size))
