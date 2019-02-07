import os
import torch
from torch.autograd import Variable
import numpy as np
from sklearn import preprocessing

def make_folder(path, version):
        if not os.path.exists(os.path.join(path, version)):
            os.makedirs(os.path.join(path, version))


def tensor2var(x, grad=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=grad)

def var2tensor(x):
    return x.data.cpu()

def var2numpy(x):
    return x.data.cpu().numpy()

def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)


def create_lable_func(min_val, max_val, n_bins):
    ''' genrate function to encode continus values to n classes
        return mappingf funciton to lable or to one hot lable
        usage:
        labl, hot_l = creat_lable_func(0, 10, 5)
        x_fit = np.linspace(0, 10, 11)
        print("y leables", labl(x_fit))
        print("yhot leables",  hot_l(x_fit))
    '''
    x_fit = np.linspace(min_val, max_val, 5000)
    bins = np.linspace(min_val, max_val, n_bins)
    x_fit = np.digitize(x_fit, bins)
    le = preprocessing.LabelEncoder()
    le.fit(x_fit)
    le_one_hot = preprocessing.LabelBinarizer()
    assert len(le.classes_) == n_bins
    le_one_hot.fit(le.classes_)
    y = le.transform(x_fit)
    yhot = le_one_hot.transform(x_fit)

    def _enc_lables(data):
        fit = data.cpu().numpy() if isinstance(data, torch.Tensor) else data
        return le.transform(np.digitize(fit, bins))

    def _enc_lables_hot(data):
        fit = data.cpu().numpy() if isinstance(data, torch.Tensor) else data
        return le_one_hot.transform(np.digitize(fit, bins))
    return _enc_lables, _enc_lables_hot
