from tensorflow import keras
import numpy as np
from read_smp import Samples
import matplotlib.pyplot as plt

marks = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
             'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
              'W', 'X', 'Y', 'Z']

new_model = keras.models.load_model('models/A-9_model.h5')

test = Samples()
test.loadFromSmpWxH('D:\Work\emnist_corrected\A-9.smp28')

x_test = np.array(test.imgs[:1000]).astype('float32')
x_test /= 255
x_test = np.asarray(x_test)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')


prediction = new_model.predict(x_test)
prediction_proba = new_model.predict_proba(x_test)

for i in range(100,150):
    print(i ,": ", max(prediction_proba[i]), end=" ")
print("\n")
labels = []

for i in range(len(x_test)):
        labels.append(marks[np.argmax(prediction[i])] if marks[np.argmax(prediction[i])].isdigit() else (marks[np.argmax(prediction[i]) + 1]))
print(labels[100:150])
print(test.labels[100:150])


