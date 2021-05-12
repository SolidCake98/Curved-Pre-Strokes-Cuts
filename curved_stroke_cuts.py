import itertools
import math
from queue import Queue

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from tensorflow import keras

from auto_separate import look_at_sample, save_fig
from read_smp import Samples
import cv2


def gray_img_to_to_file(img, name):
    try:
        f = open(name, "w")
        f.write(str(img.shape[1]) + " " + str(img.shape[0]) + "\n")
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                f.write(str(img[y, x]) + " ")
        f.close()
    except:
        print("Can't write to file")


def value_to_coords(value, width):
    x = value % width
    y = int(value / width)
    return x, y


def read_trace(name):
    way_trough = []
    try:
        f = open(name, "r")
        w_h_str = f.readline().split(" ")
        w, h = int(w_h_str[0]), int(w_h_str[1])
        values = f.readline().split(" ")
        for i in values:
            way_trough.append(value_to_coords(int(i), w))
        f.close()
    except:
        print("Can't read from file")
    return way_trough


class CurvedPreStrokeCuts:
    def __init__(self, model):
        self.model = keras.models.load_model(model)
        self.marks = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
                      'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
                      'W', 'X', 'Y', 'Z']

    # стоимость точки
    # предостерегающая от излишнего искривления пути
    def func_cs(self, delta):
        if abs(delta) == 0:
            return 0
        elif abs(delta) == 1:
            return 1
        else:
            return math.inf

    # стоимость точки
    # предостерегающая от разрезания буквы
    # и поощрающая за нахождения пути близкому к левому краю
    def func_ci(self, x, y, img):
        if x >= img.shape[1] - 1:
            if img[y, x - 1] == 0:
                return 0
            else:
                return 1
        elif (int(img[y, x + 1]) - img[y, x]) < 0 and img[y, x] >= 100:
            return -3
        elif (int(img[y, x + 1]) - img[y, x]) == 0 and img[y, x] >= 100:
            return 0
        else:
            return 1

    # нахождение точек для прокладывания пути на верх и вниз от y_c +- delta
    def find_points_to_up_and_down(self, point, source, cost):
        point_to_up = None
        point_to_down = None
        while True:
            if point_to_down is None:
                if source[point[1], point[0]][1] > point[1]:
                    point_to_down = point
                    point = (point[0], point[1] - 1)
                else:
                    point_to_up = point
                    point = (point[0], point[1] + 1)
            if point_to_up is None:
                if source[point[1], point[0]][1] < point[1]:
                    point_to_up = point
                    point = (point[0], point[1] + 1)
                else:
                    point_to_down = point
                    point = (point[0], point[1] - 1)
            if point_to_up != None and point_to_down != None:
                break

        weight = 0
        for point in [point_to_up, point_to_down]:
            p = point
            weight += cost[p[1], p[0]]
        return point_to_up, point_to_down, weight

    # алгоритм нахождения оптимального пути между (y_c, h) и (0, y_c)
    def find_path(self, img, x1, x2):
        h, w = img.shape
        y_c = int(h / 2)
        # стоимость пути в точки x, y
        cost = np.full((h, w), math.inf)
        # в i, j элементе хранится координаты следующей точки пути
        source = np.full((h, w), None)

        # поиск оптимального пути от 0 до y_c для x = 0..w
        queue = Queue()
        for i in range(0, w):
            queue.put((i, 0))
            cost[0, i] = 0

        while not queue.empty():
            i, j = queue.get()
            if j < y_c:
                for delta in [-1, 0, 1]:
                    if (i == 0 and delta == -1) or (i == w - 1 and delta == 1):
                        continue
                    new_cost = self.func_cs(delta) + self.func_ci(i + delta, j + 1, img) + cost[j, i]
                    if new_cost < cost[j + 1, i + delta]:
                        source[j + 1, i + delta] = (i, j)
                        cost[j + 1, i + delta] = new_cost
                        queue.put((i + delta, j + 1))

        # поиск оптимального пути от h до y_c для x = 0..w
        queue = Queue()
        for i in range(0, w):
            queue.put((i, h - 1))
            cost[h - 1, i] = 0

        while not queue.empty():
            i, j = queue.get()
            if j > y_c:
                for delta in [-1, 0, 1]:
                    if (i == 0 and delta == -1) or (i == w - 1 and delta == 1):
                        continue
                    new_cost = self.func_cs(delta) + self.func_ci(i + delta, j - 1, img) + cost[j, i]
                    if new_cost < cost[j - 1, i + delta]:
                        source[j - 1, i + delta] = (i, j)
                        cost[j - 1, i + delta] = new_cost
                        queue.put((i + delta, j - 1))

        way_through = self.find_best_path_between_two_points(x1, x2, y_c, source, cost)

        return way_through

    # поиск пути с минимальной стоимостью в диапозоне [x1;x2]
    def find_best_path_between_two_points(self, x1, x2, y_c, source, cost):
        point_to_up = None
        point_to_down = None
        min_weight = 100
        for i in range(x1, x2):
            point = (i, y_c)
            pu, pd, weight = self.find_points_to_up_and_down(point, source, cost)
            if weight < min_weight:
                min_weight = weight
                point_to_up, point_to_down = pu, pd

        way_trough = []
        for point in [point_to_up, point_to_down]:
            p = point
            while p is not None:
                way_trough.append(p)
                p = source[p[1], p[0]]
        return way_trough

    # выделение букв из сегмента
    def extract_sub_images(self, img, k):
        h, w = img.shape

        # скольок сегментов хотим проверить
        # пока константно разбиваем на 5
        delta = int(w/k)

        # выделяем точки в которых мы хотим разрезать изображения
        point = 0
        points = []
        while point < w:
            points.append(point)
            point = point + delta

        # поиск путей в заданных сегментах
        ways = []
        max_ws = [0]
        min_ws = [0]
        points = points[1:]
        way = [(0, x) for x in range(h)]
        ways.append(way)

        for point in points:
            if point + delta/2 >= w or point - delta/2 < 0:
                break
            way = self.find_path(img, point - int(delta/2), point + int(delta/2))
            max_ws.append(max(way, key=lambda x: x[0])[0])
            min_ws.append(min(way, key=lambda x: x[0])[0])
            ways.append(way)

        way = [(w - 1, x) for x in range(h)]
        ways.append(way)
        min_ws.append(w - 1)
        max_ws.append(w - 1)
        print(min_ws)
        print(max_ws)

        # предсказывание букв
        # ищем сегмент от 0 до i c наибольшей вероятностью
        # если конец сегмента - это конец изображения то прекращаем
        # иначе проверям сегменты с i до j и т.д

        possible_imgs = []

        i = 0
        counter = 0
        cuts = []
        while i + 1 < len(ways):
            max_predict_value = 0
            possible_imgs.append(0)
            cuts.append(ways[i])
            for j in range(i + 1, len(ways)):
                possible_img = np.zeros((h, max_ws[j] - min_ws[i])) + 255

                if max_ws[i] >= min_ws[j]:
                    continue
                for k, coord in enumerate(ways[j]):
                    possible_img[coord[1], ways[i][k][0] - min_ws[i]: coord[0] - min_ws[i]] = img[coord[1], ways[i][k][0]: coord[0]]

                possible_img = cv2.resize(possible_img, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)

                x_test = np.array(possible_img).astype('float32')
                x_test /= 255
                x_test = np.asarray(x_test)
                x_test = x_test.reshape(1, 28, 28, 1).astype('float32')
                prediction = self.model.predict(x_test)

                if max(prediction[0]) > max_predict_value:
                    cuts[counter] = ways[j]
                    max_predict_value = max(prediction[0])
                    possible_imgs[counter] = (self.marks[np.argmax(prediction[0])], (min_ws[i], max_ws[j]), j, max_predict_value)

                print(self.marks[np.argmax(prediction[0])], max(prediction[0]), min_ws[i], max_ws[j])
            # self.show_path(index, cuts[counter])
            i = possible_imgs[counter][2]
            counter += 1
        return possible_imgs, cuts

    def predict_img(self, im, k):
        res, cuts = self.extract_sub_images(im, k)
        cuts = cuts[:-1]

        copy_img = cv2.cvtColor(im.copy(), cv2.COLOR_GRAY2RGB)
        for cut in cuts:
            for x, y in cut:
                copy_img[y, x] = [255, 0, 0]
        return res, copy_img

    def show_path(self, img, path):
        copy_img = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2RGB)
        for x, y in path:
            copy_img[y, x] = [255, 0, 0]
        plt.imshow(copy_img)
        plt.show()

