import cv2
from PyQt5 import QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QListWidgetItem

from curved_stroke_cuts import CurvedPreStrokeCuts
from ui.DrawArea import DrawArea
from ui.SplitSegmentArea import SplitSegmentArea
from ui.Ui_MainWindow import Ui_MainWindow

import numpy as np


class MainWindow(QMainWindow):

    csps = CurvedPreStrokeCuts("models/A-9_model_d.h5")

    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.draw_area = DrawArea(self)
        self.ui.source_image_area.setWidget(self.draw_area)

        self.split_segment_area = SplitSegmentArea(self)
        self.ui.divided_image_area.setWidget(self.split_segment_area)

        self.ui.recognize_button.clicked.connect(self.recognize)

        self.ui.action.triggered.connect(self.load_model)

    def load_model(self):
        name = QFileDialog.getOpenFileName(self, caption='Open file', directory='', filter="H5 (*.h5)")[0]
        if not name:
            return
        self.csps = CurvedPreStrokeCuts(name)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            self.draw_area.clear_area()

    def recognize(self):
        count_of_segments = int(self.ui.count_of_segments.value())

        if self.csps is None:
            return

        t_img = self.draw_area.img.copy() * 255
        t_img = np.array(t_img, dtype=np.uint8)

        res, im = self.csps.predict_img(t_img, count_of_segments)
        self.ui.resultWidget.addItem(QListWidgetItem(str(res)))

        height, width, chanel = im.shape

        q_image = QImage(im, width, height, chanel*width, QImage.Format_RGB888)

        self.split_segment_area.load_image(QPixmap.fromImage(q_image))

