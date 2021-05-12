import struct
import numpy as np

alphabet = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
             'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
              'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g',
               'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
                's', 't', 'u', 'v', 'w', 'x', 'y', 'z']


# Типы данных и число байт
# формата IDX
DATA_TYPES_IDX = {
    0x08: ('ubyte', 1),
    0x09: ('byte', 1),
    0x0B: ('>i2', 2),
    0x0C: ('>i4', 4),
    0x0D: ('>f4', 4),
    0x0E: ('>f8', 8)
}


def loademnist(imgfile, lblfile, letters=None):
    images = readidx(imgfile)
    labels = readidx(lblfile)
    print("Load EMNIST %s images" % len(images))
    if letters is None:
        return images, labels
    li = []
    d = {alphabet[i]:i for i in range(len(alphabet))}
    for l in letters:
        # Находим метку (порядковый номер)
        label = d[l] if l in d else []
        indices = np.nonzero((labels==label))[0]
        if len(indices) == 0:
            raise Exception("Can't find letter: {}".format(l))
        li.append(indices)
    imgs = np.vstack(([images[indices] for indices in li]))
    lbls = np.hstack([[i] * len(li[i]) for i in range(len(li))])
    return imgs, lbls

def readidx(fname):
    f = open(fname, 'rb')
    magic = struct.unpack('>BBBB', f.read(4))
    # Тип данных
    dt = magic[2]
    # Размерность
    dd = magic[3]
    dims = struct.unpack('>' + 'I' * dd, f.read(4 * dd))
    sz = 1
    for i in range(len(dims)):
        sz = sz * dims[i]
    # Данные
    dtype, dbytes = DATA_TYPES_IDX[dt]
    data = np.frombuffer(f.read(sz * dbytes), dtype=np.dtype(dtype)).reshape(dims)
    f.close()
    return data