from PyQt5 import QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel


class SplitSegmentArea(QLabel):

    def __init__(self, parent):
        super().__init__(parent)
        self.pix = None

    def load_image(self, pix):
        width = self.rect().width()
        height = self.rect().height()

        pix = pix.scaled(width, height, Qt.KeepAspectRatio, Qt.FastTransformation)
        self.pix = pix

        self.setPixmap(self.pix)
