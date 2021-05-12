from PyQt5 import QtGui
from PyQt5.QtCore import Qt, QPoint, QRect
from PyQt5.QtGui import QImage, QPainter, QPen
from PyQt5.QtWidgets import QMainWindow, QAction, QFileDialog, QLabel

import numpy as np


class DrawArea(QLabel):
    def __init__(self, parent):
        super().__init__(parent)

        self.width = 64
        self.height = 42

        self.img = np.ones((self.height, self.width))

        self.size = self.rect()
        self.scale_x = self.size.width()/self.width
        self.scale_y = self.size.height()/self.height

        self.q_image = QImage(self.width, self.height, QImage.Format_Grayscale8)
        self.q_image.fill(Qt.white)

        self.drawing = False
        self.brush_size = 1

        self.brush_color = QtGui.QColor(0, 0, 0)

        self.last_point = QPoint()
        self.mouse_but = 0

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton or event.button() == Qt.RightButton:
            self.drawing = True
            self.mouse_but = event.button()
            x, y = int(event.pos().x() / self.scale_x), int(event.pos().y()/self.scale_y)
            self.last_point = QPoint(x, y)

    def mouseMoveEvent(self, event):
        if (self.mouse_but == Qt.LeftButton or self.mouse_but == Qt.RightButton) and self.drawing:
            painter = QPainter(self.q_image)

            for x in range(self.width):
                for y in range(self.height):
                    dist = (x - self.last_point.x())**2 + (y - self.last_point.y())**2
                    if dist < 1:
                        dist = 1
                    dist *= dist

                    if self.mouse_but == Qt.LeftButton:
                        self.img[y, x] -= 0.4 / dist
                    else:
                        self.img[y, x] += 0.4 / dist

                    if self.img[y, x] < 0:
                        self.img[y, x] = 0
                    if self.img[y, x] > 0.9:
                        self.img[y, x] = 1

                    color = self.img[y, x] * 255
                    color = QtGui.QColor(color, color, color)
                    painter.setPen(QPen(color, self.brush_size, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
                    painter.drawRect(x, y, 1, 1)

            x, y = int(event.pos().x() / self.scale_x), int(event.pos().y() / self.scale_y)
            self.last_point = QPoint(x, y)
            self.update()

    def clear_area(self):
        painter = QPainter(self.q_image)
        for x in range(self.width):
            for y in range(self.height):
                self.img[y, x] = 255
                color = QtGui.QColor(255, 255, 255)
                painter.setPen(QPen(color, self.brush_size, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
                painter.drawRect(x, y, 1, 1)
        self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False

    def paintEvent(self, event):
        canvas_painter = QPainter(self)
        self.size = self.rect()
        self.scale_x = self.size.width() / self.width
        self.scale_y = self.size.height() / self.height
        canvas_painter.drawImage(self.rect(), self.q_image, self.q_image.rect())
