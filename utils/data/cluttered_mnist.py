import os
import warnings

import codecs
import gzip
import numpy as np
from torch.utils.data import (Dataset,
                              DataLoader)


class setup_mnist_data(object):
    """
    Generates cluttered MNIST data with clutter. Each image has one complete
    digit. The source data is downloaded, as needed.
    
    Prior to generating data, the source data is split into training,
    validation, and testing subsets. For each subset, clutter is then sourced
    from the same subset. Validation and testing data is pregenerated while
    training data is generated on the fly.
    
    data_dir : directory containing the data (or where to download the data)
    n_valid : number of data points in the validation subset
    n_clutter : number of distractors, forming clutter
    size_clutter : the size of each distractor (a square cropped from an image)
    size_output : the size of each output image (square)
    segment_fraction : the fraction of the training set to provide 
        segmentation masks for.
    verbose : whether to print messages to stdout when getting source data
        for the first time
    rng : optional numpy random number generator
    
    Interface:
    Use `gen_train`, `gen_valid`, `gen_test` to get minibatch generators for
    the training, validation, and testing data subsets, respectively.
    """
    
    urls = [
        'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
    ]
    
    def __init__(self, data_dir, n_valid, n_clutter=8, size_clutter=8,
                 size_output=100, segment_fraction=1,
                 unlabeled_digits=None, yield_only_labeled=False,
                 gen_train_online=True, background_noise=0.01,
                 verbose=False, rng=None):
        self.data_dir = data_dir
        self.n_valid = n_valid
        self.n_clutter = n_clutter
        self.size_clutter = size_clutter
        self.size_output = size_output
        self.segment_fraction = segment_fraction
        self.unlabeled_digits = unlabeled_digits
        self.yield_only_labeled = yield_only_labeled
        self.gen_train_online = gen_train_online
        self.background_noise = background_noise
        self.verbose = verbose
        self.rng = rng if rng is not None else np.random.RandomState()
        
        # Download regular MNIST data if necessary.
        self._download()
        
        # Load regular MNIST data.
        self._load_data()
        
        # Pre-generate validation and test sets. Using same rng.
        self._validation_set = self._generate_cluttered(n_valid, fold='valid')
        self._testing_set = self._generate_cluttered(len(self._x['test']),
                                                     fold='test')
        if not self.gen_train_online:
            # Pre-generate training data.
            indices_seg = self._indices_seg
            self._training_set = \
                self._generate_cluttered(len(self._x['train']),
                                         fold='train',
                                         indices_seg=indices_seg)
        
    def _load_data(self):
        # Load.
        data = np.load(os.path.join(self.data_dir, 'unpacked', 'data.npz'))
        x_train = data['x_train']
        y_train = data['y_train']
        x_test  = data['x_test']
        y_test  = data['y_test']
        
        # Type conversion, normalization.
        x_train = x_train.astype(np.float32)/255.
        y_train = y_train.astype(np.int64)
        x_test  = x_test.astype(np.float32)/255.
        y_test  = y_test.astype(np.int64)
        
        # Split out validation.
        rng_split = np.random.RandomState(0)    # Hardcode for split.
        if self.n_valid < 0 or self.n_valid > len(x_train):
            raise ValueError("`n_valid` must be in [0, {}] but is {}"
                             "".format(len(x_train), self.n_valid))
        R = rng_split.permutation(len(x_train))
        x_train = x_train[R]
        y_train = y_train[R]
        x_valid = x_train[len(x_train)-self.n_valid:]
        y_valid = y_train[len(y_train)-self.n_valid:]
        x_train = x_train[:len(x_train)-self.n_valid]
        y_train = y_train[:len(y_train)-self.n_valid]
        
        # Provide segmentation masks for these training cases.
        n_segment = max(1, int(self.segment_fraction*len(x_train)+0.5))
        indices_seg = [i
                       for i in range(len(x_train))
                       if y_train[i] not in (self.unlabeled_digits or [])]
        random_order = self.rng.permutation(len(indices_seg))
        indices_seg  = [indices_seg[i] for i in random_order]
        self._indices_seg = set(indices_seg[:n_segment])
        if len(self._indices_seg) < n_segment:
            actual_fraction = len(self._indices_seg)/float(len(x_train))
            warnings.warn("After limiting the digits that could be labeled, "
                          "the labeled data contains only {:.2f}% of the "
                          "source data rather than the requested fraction "
                          "({:.2f}%).".format(actual_fraction*100,
                                              self.segment_fraction*100))
        if self.yield_only_labeled:
            x_train = x_train[list(self._indices_seg)]
            y_train = y_train[list(self._indices_seg)]
            self._indices_seg = None  # Dataset truncated; use all indices.
        
        self._x = {'train': x_train,
                   'valid': x_valid,
                   'test' : x_test}
        self._y = {'train': y_train,
                   'valid': y_valid,
                   'test' : y_test}
        
    def _download(self):
        '''
        This function adapted from torchvision code in:
        github.com/pytorch/vision/blob/master/torchvision/datasets/mnist.py
        
        commit : 1fb0ccf71620d113cb72696b2eb8317b3e252cbb
        license: BSD
        '''
        from six.moves import urllib
        import gzip
        
        path_raw      = os.path.join(self.data_dir, 'raw')
        path_unpacked = os.path.join(self.data_dir, 'unpacked')
        
        # Download the MNIST data if necessary.
        if not os.path.exists(path_raw):
            os.makedirs(path_raw)
            for url in self.urls:
                self._print('Downloading {}'.format(url))
                data = urllib.request.urlopen(url)
                fn = url.rpartition('/')[2]
                path_file = os.path.join(path_raw, fn)
                with open(path_file, 'wb') as f:
                    f.write(data.read())
                with open(path_file.replace('.gz', ''), 'wb') as out_f, \
                        gzip.GzipFile(path_file) as zip_f:
                    out_f.write(zip_f.read())
                os.unlink(path_file)
        
        # Prepare files from downloaded data if necessary.
        if not os.path.exists(path_unpacked):
            os.makedirs(path_unpacked)
            self._print('Preparing data.')
            d_train = (
                self._read_image_file(os.path.join(self.data_dir, 'raw',
                                                   'train-images-idx3-ubyte')),
                self._read_label_file(os.path.join(self.data_dir, 'raw',
                                                   'train-labels-idx1-ubyte'))
                )
            d_test = (
                self._read_image_file(os.path.join(self.data_dir, 'raw',
                                                   't10k-images-idx3-ubyte')),
                self._read_label_file(os.path.join(self.data_dir, 'raw',
                                                   't10k-labels-idx1-ubyte'))
                )
            with open(os.path.join(path_unpacked, 'data.npz'), 'wb') as f:
                np.savez_compressed(f,
                                    x_train=d_train[0], y_train=d_train[1],
                                    x_test=d_test[0], y_test=d_test[1])
            self._print('Done!')
        
    def _get_int(self, b):
        return int(codecs.encode(b, 'hex'), 16)
        
    def _read_label_file(self, path):
        with open(path, 'rb') as f:
            data = f.read()
            assert self._get_int(data[:4]) == 2049
            length = self._get_int(data[4:8])
            parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
            arr = np.reshape(parsed, newshape=(length,)).astype(np.int64)
            return arr
        
    def _read_image_file(self, path):
        with open(path, 'rb') as f:
            data = f.read()
            assert self._get_int(data[:4]) == 2051
            length = self._get_int(data[4:8])
            num_rows = self._get_int(data[8:12])
            num_cols = self._get_int(data[12:16])
            images = []
            parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
            arr = np.reshape(parsed, newshape=(length, num_rows, num_cols))
            return arr
        
    def _print(self, msg):
        if self.verbose:
            print(msg) 
        
    def _generate_cluttered_sample(self, fold, indices_seg=None):
        # The background is noisy.
        def background():
            noise = self.rng.randn(self.size_output, self.size_output)
            return self.background_noise*noise.astype(np.float32)
        
        # Randomly sample a data point.
        idx_max = len(self._x[fold])
        idx_sample = self.rng.randint(idx_max)
        x = self._x[fold][idx_sample]
        y = self._y[fold][idx_sample]
        x_out = background()
        
        # Helper function to get slices for both dimensions of square crop.
        def random_crop(idx_max, width):
            idx0 = self.rng.randint(idx_max-width)
            idx1 = self.rng.randint(idx_max-width)
            return slice(idx0, idx0+width), slice(idx1, idx1+width)
        
        # Helper function to add clutter to an image (in place).
        # Clutter is sampled from the same data fold.
        def add_clutter(image, num):
            for idx in self.rng.randint(idx_max, size=num):
                crop_target = random_crop(self.size_output, self.size_clutter)
                crop_source = random_crop(28,               self.size_clutter)
                cropped     = self._x[fold][idx][crop_source]
                s = (cropped+image[crop_target])>1  # Oversaturated here.
                m = cropped>0                       # Clutter is here.
                m[s] = 0            # Treat oversaturated region separately.
                image[crop_target][m] = cropped[m]
                image[crop_target][s] = np.maximum(image[crop_target][s],
                                                   cropped[s])
        
        # Generate output with clutter.
        x_crop_indices = random_crop(self.size_output, 28)
        x_out[x_crop_indices][x>0] = x[x>0]
        add_clutter(x_out, num=self.n_clutter)
        
        # Generate clutter images without x.
        clutter = background()
        add_clutter(clutter, num=self.n_clutter)
        
        # Create segmentation mask for x.
        mask = None
        if indices_seg is None or idx_sample in indices_seg:
            mask = np.zeros_like(x_out, dtype=np.int64)
            mask[x_crop_indices][x>0.5] = 1
        
        return (clutter, x_out, mask, y)
        
    def _generate_cluttered(self, num, fold, indices_seg=None):
        output_list = []
        for _ in range(num):
            sample = self._generate_cluttered_sample(fold, indices_seg)
            output_list.append(sample)
        return output_list


class mnist_data_train(Dataset):
    def __init__(self, data, length=None):
        self.data = data
        self.length = length
    def __getitem__(self, idx):
        x = None
        if self.data.gen_train_online:
            return self.data._generate_cluttered_sample('train',
                                                        self.data._indices_seg)
        else:
            return self.data._training_set[idx]
    def __len__(self):
        if self.data.gen_train_online:
            return self.length
        else:
            return len(self.data._training_set)


class mnist_data_valid(Dataset):
    def __init__(self, data):
        self.data = data
    def __getitem__(self, idx):
        return self.data._validation_set[idx]
    def __len__(self):
        return len(self.data._validation_set)


class mnist_data_test(Dataset):
    def __init__(self, data):
        self.data = data
    def __getitem__(self, idx):
        return self.data._testing_set[idx]
    def __len__(self):
        return len(self.data._testing_set)


if __name__=='__main__':
    """
    Interactive debug code.
    
    """
    data = setup_mnist_data(data_dir='data',
                            n_valid=500,
                            n_clutter=50,
                            size_clutter=10,
                            size_output=100,
                            verbose=True,
                            rng=np.random.RandomState(1234))
    loader_kwargs = {'batch_size': 10,
                     'shuffle': True,
                     'num_workers': 1}
    
    loader = {'train': iter(DataLoader(mnist_data_train(data, length=10),
                                       **loader_kwargs)),
              'valid': iter(DataLoader(mnist_data_valid(data),
                                       **loader_kwargs)),
              'test':  iter(DataLoader(mnist_data_test(data),
                                       **loader_kwargs))}
    
    import matplotlib.pyplot as plt
    def make_panel(batch, axis, ax_handle):
        precat = [np.concatenate(list(zip(*batch['train'][:3]))[axis], axis=0),
                  np.concatenate(list(zip(*batch['valid'][:3]))[axis], axis=0),
                  np.concatenate(list(zip(*batch['test'][:3]))[axis], axis=0)]
        cat = np.concatenate(precat, axis=1)
        cat[range(0, cat.shape[0], 100), :] = 1
        cat[:, range(0, cat.shape[1], 100)] = 1
        ax_handle.imshow(cat)
    fig, ax = plt.subplots(1, 3)
    plt.gray()
    for i in range(10):
        batch = dict([(key, [x.cpu().numpy() for x in next(loader[key])])
                      for key in loader])
        print("batch {}: shape_train={}, shape_valid={}, shape_test={}"
              "".format(i, len(batch['train']), len(batch['valid']),
                        len(batch['test'])))
        make_panel(batch, 0, ax[0])
        make_panel(batch, 1, ax[1])
        make_panel(batch, 2, ax[2])
        plt.show(block=True)
