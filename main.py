from __future__ import annotations

import sys
from abc import ABC, abstractmethod

from PyQt5.QtWidgets import QApplication
import pandas as pd
from auto_separate import HSeparate, look_at_sample
from curved_stroke_cuts import CurvedPreStrokeCuts
from read_smp import Samples
from slider_window import Detector
from ui.MainWindow import MainWindow


class AbstractPredictor(ABC):

    @abstractmethod
    def predict(self, img_s, model):
        pass


class HistogramPredict(AbstractPredictor):

    def predict(self, smp, model):
        h_sep = HSeparate(smp, model)
        res = []
        for i in h_sep.make_prediction():
            res.append(i)
        return res


class SlideWindowPredict(AbstractPredictor):

    def predict(self, smp, model):
        samples = Samples()
        samples.loadFromSmp(smp)
        res = []
        for i in range(len(samples.imgs)):
            im = samples.imgs[i].shape[1]
            detector = Detector(model, (int(im.shape[1] / 4), samples.imgs[i].shape[0]))
            detector.search(samples.imgs[i])
            bx = sorted(detector.boxes, key=lambda x: x[0])
            bx = list((name[0], name[2], name[4], name[5],) for name in bx)
            look_at_sample(im, samples.labels[i] + str(i), bx)
            res.append(bx)


class CPSCPredict(AbstractPredictor):

    data = []

    def predict(self, smp, model):
        samples = Samples()
        samples.loadFromSmp(smp)

        cpsc = CurvedPreStrokeCuts(model)
        res = []
        for i in range(len(samples.imgs)):
            im = samples.imgs[i]
            res_im, im = cpsc.predict_img(im, 5)
            res.append((res_im, samples.labels[i]))
            look_at_sample(im, samples.labels[i] + str(i), res_im)
        self.res_to_data(res)

    def res_to_data(self, res):
        data = {'name':[], 'predict_name':[], 'prob':[]}
        for el in res:
            stroke = ''
            prob = 0
            for i in el[0]:
                stroke += i[0]
                prob += i[3]
            prob = prob/len(el[0])
            data['name'].append(el[1])
            data['predict_name'].append(stroke)
            data['prob'].append(prob)
        df = pd.DataFrame(data=data)
        df.to_excel(r'result.xlsx', index=False)


class Context:
    def __init__(self, predictor: AbstractPredictor) -> None:
        self._predictor = predictor

    @property
    def predictor(self) -> AbstractPredictor:
        return self._predictor

    @predictor.setter
    def strategy(self, predictor: AbstractPredictor) -> None:
        self._predictor = predictor

    def start_predict(self, smp, model) -> None:
        result = self._predictor.predict(smp, model)
        print(result)


if __name__ == "__main__":
    # path_to_smp = 'letters\\connected_letters.smp'
    # path_to_model = 'models\\A-9_model_le_net.h5'
    # context = Context(CPSCPredict())
    # context.start_predict(path_to_smp, path_to_model)
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()