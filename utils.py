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

def tensor_tp_np(data):
    return data.cpu().numpy() if isinstance(data, torch.Tensor) else data

def create_lable_func(min_val, max_val, bins, clip=False):
    ''' genrate function to encode continus values to n classes
        return mappingf funciton to lable or to one hot lable
        bins(float): number of bins between min and max
        bins(array_like): boearder for new bins
        usage:
        labl, hot_l = create_lable_func(0, 10, 5)
        x_fit = np.linspace(0, 10, 11)
        print("y leables", labl(x_fit))
        print("yhot leables",  hot_l(x_fit))
    '''
    assert min_val < max_val
    x_fit = np.linspace(min_val, max_val, 5000,endpoint=True)# TODO
    if isinstance(bins,int):
        bin_array = np.linspace(min_val, max_val, bins,endpoint=True)
    else:
        assert len(bins) >2
        bin_array =bins
    def _digitize(x):
        return  np.digitize(x, bin_array,right=True)

    x_fit = _digitize(x_fit)
    le = preprocessing.LabelEncoder()
    le.fit(x_fit)
    le_one_hot = preprocessing.LabelBinarizer()
    if isinstance(bins,int):
        assert len(le.classes_) == bins
    le_one_hot.fit(le.classes_)
    y = le.transform(x_fit)
    yhot = le_one_hot.transform(x_fit)

    def _enc_lables(data):
        fit = tensor_tp_np(data)
        if clip:
            fit= np.clip(fit, min_val, max_val)
        return le.transform(_digitize(fit))

    def _enc_lables_hot(data):
        fit = tensor_tp_np(data)
        if clip:
            fit= np.clip(fit, min_val, max_val)
        return le_one_hot.transform(_digitize(fit))
    return _enc_lables, _enc_lables_hot

if __name__ =="__main__":
        def test_fix_bins(x_min,x_max,n_bins):
            to_lable, to_hot = create_lable_func(x_min, x_max, n_bins)
            ''' test for dic '''
            n_data= n_bins#*(x_max - x_min)
            input_data =np.linspace(x_min, x_max, n_data)
            x_fit_lable = to_lable(input_data)
            x_fit_hot = to_hot(input_data)
            assert len(set(x_fit_lable)) ==n_bins, "not all classes in lables, min {}, max {},bins {}".format(x_min,x_max,n_bins)
            expected_hot_size= n_bins if n_bins > 2 else 1
            assert x_fit_hot.shape[1] == expected_hot_size, "missing dim in one hot, min {}, max {},bins {}".format(x_min,x_max,n_bins)
            if expected_hot_size >1:
                # check if lable fit is same with the one hot encoding
                for l, one_hot in zip(x_fit_lable,x_fit_hot):
                    assert np.sum(one_hot)==1, " one hot not set or more are set "
                    idx_hot = np.nonzero(one_hot == 1)
                    assert len(idx_hot)>0, "min {}, max {},bins {}".format(x_min,x_max,n_bins)
                    assert len(idx_hot[0])>0, "min {}, max {},bins {}".format(x_min,x_max,n_bins)
                    idx_hot=idx_hot[0][0]
                    assert idx_hot== l, "min {}, max {},bins {}".format(x_min,x_max,n_bins)
        # test_fix_bins(0,10,5)
        for _ in range(100):
            diff=100
            min_x= np.random.uniform(-diff,diff)
            test_fix_bins(min_x,np.random.uniform(min_x,min_x+diff),np.random.randint(2,111))

        # test for with different bins
        bins=[0,1,2,4,20,200]
        input_data=[0,1,2,3,4,5,19,20,21,100,300]

        to_lable, to_hot = create_lable_func(0, 200, bins,clip=True)
        fit_l=to_lable(input_data)
        print('fit_l: {}'.format(fit_l))
        for i,l,e in zip(input_data,fit_l,[0,1,2,3,3,4,4,4,5,5,5]):
            assert l==e, "for {} got lable{} expected {}".format(i,l,e)


