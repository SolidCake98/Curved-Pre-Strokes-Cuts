import os
import numpy as np
from sklearn.model_selection import train_test_split


class Samples:

    def __init__(self):
        self.imgs = []
        self.labels = []

    def loadFromSmpWxH(self, smpname):  # загрузка из smpname=xxx.smpWxH, lblname=xxx.lblWxH
        try:
            pos = smpname.rfind(".smp")  # убрать расширение из имени файла
            if pos >= 0:
                dx = smpname[pos + 4:]
                if dx.find("x") != -1:  # если изображения не квадратные
                    dy = dx[dx.find("x") + 1:]
                    dx = dx[:dx.find("x")]
                else:
                    dy = dx
                dx = int(dx)
                dy = int(dy)

            if pos >= 0:
                lblname = smpname[:pos] + ".lbl" + smpname[pos + 4:]
            else:
                return False  # нельзя сформировать имя lbl
            if not os.path.exists(lblname): return False

            f = open(smpname, "rb")
            line = f.read(dx * dy)
            while (line):
                a = np.frombuffer(line, dtype='u1')
                a = 255 - a
                img = a.reshape([dy, dx])
                self.imgs.append(img)
                line = f.read(dx * dy)
            f.close()

            f = open(lblname, "rb")
            line = f.readline()
            while line:
                line = line[:-1]
                label = line.decode('utf-8')
                self.labels.append(label)
                line = f.readline()
            f.close()
            return True
        except:
            print("exception")
            return False

    def loadFromSmp(self, name, clearall=False):
        if clearall:  # удалить имеющиеся
            self.imgs = []  # изображения
            self.labels = []  # метки
        try:
            f = open(name, "rb")
            n = int.from_bytes(f.read(4), byteorder='big')  # число примеров
            for i in range(n):
                nlabel = int.from_bytes(f.read(1), byteorder='big')  # длина метки
                label = f.read(nlabel).decode('utf-8')  # метка

                dx, dy = int.from_bytes(f.read(4), byteorder='big'), int.from_bytes(f.read(4), byteorder='big')
                a = np.frombuffer(f.read(dx * dy), dtype='u1')
                img = a.reshape([dy, dx])
                self.imgs.append(img)
                self.labels.append(label)
            f.close()
            return True
        except:
            print("exception")
            return False

    def split(self):
        X_train, X_test, y_train, y_test = train_test_split(self.imgs, self.labels, test_size=0.33)
        return X_train, X_test, y_train, y_test

# train = Samples()
# train.loadFromSmpWxH('D:\Work\YTypes\YTrain.smp28')
#
# test = Samples()
# test.loadFromSmpWxH('D:\Work\YTypes\YTest.smp28')
#
# import matplotlib.pyplot as plt
# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train.imgs[i], cmap='gray')
#     plt.xlabel(train.labels[i])
# plt.show()