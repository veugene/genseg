import torch
import os
import numpy as np
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
from PIL import Image

def convert_to_rgb(img, is_grayscale=False):
    """
    Given an image, make sure it has 3 channels and that it is between 0 and 1.

    Notes
    -----
    Original code:
    https://github.com/costapt/vess2ret
    """
    if len(img.shape) != 3:
        raise Exception("""Image must have 3 dimensions (channels x height x width). """
                        """Given {0}""".format(len(img.shape)))
    img_ch, _, _ = img.shape
    if img_ch != 3 and img_ch != 1:
        raise Exception("""Unsupported number of channels. """
                        """Must be 1 or 3, given {0}.""".format(img_ch))
    imgp = img
    if img_ch == 1:
        imgp = np.repeat(img, 3, axis=0)
    if not is_grayscale:
        imgp = imgp * 127.5 + 127.5
        imgp /= 255.
    return np.clip(imgp.transpose((1, 2, 0)), 0, 1)

def count_params(module, trainable_only=True):
    """
    Count the number of parameters in a module.
    """
    parameters = module.parameters()
    if trainable_only:
        parameters = filter(lambda p: p.requires_grad, parameters)
    num = sum([np.prod(p.size()) for p in parameters])
    return num

class MultipleDataset(torch.utils.data.Dataset):
    """
    (Not be confused with ConcatDataset) This lets us concatenate >1
      dataset so we can return multiple values.

    Notes
    -----
    This is a bit of a hack to a problem I'm having. Please see my reply here:
      https://discuss.pytorch.org/t/train-simultaneously-on-two-datasets/649/7
    My solution to this is a bit weird, and basically it means that one
      epoch is actually multiple passes through the dataset. Since __len__
      is the max size of datasetA and datasetB, for __getitem__(index) we
      will grab the item in the smaller dataset with e.g. 
      datasetA[index % len(datasetA)] and the one in the bigger dataset with
      simply datasetB[index].
    """
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i % len(d)] for d in self.datasets)

    def __len__(self):
        return max(len(d) for d in self.datasets)

class DatasetFromFolder(Dataset):
    """
    Specify specific folders to load images from.

    Notes
    -----
    Original code:
    https://github.com/togheppi/CycleGAN/blob/master/dataset.py
    With some extra modifications done by me.
    """
    def __init__(self, image_dir, subfolder='', images=None, transform=None, resize_scale=None, crop_size=None, fliplr=False):
        """
        images: a list of images you want instead. If set to `None` then it gets all
          images in the directory specified by `image_dir` and `subfolder`.
        """
        super(DatasetFromFolder, self).__init__()
        self.input_path = os.path.join(image_dir, subfolder)
        if images == None:
            self.image_filenames = [x for x in sorted(os.listdir(self.input_path))]
        else:
            if type(images) != set:
                images = set(images)
            self.image_filenames = [ os.path.join(os.path.join(image_dir,subfolder),fname) for fname in images ]
        self.transform = transform
        if type(resize_scale) == int:
            resize_scale = (resize_scale, resize_scale)
        self.resize_scale = resize_scale
        self.crop_size = crop_size
        self.fliplr = fliplr
    def __getitem__(self, index):
        # Load Image
        img_fn = os.path.join(self.input_path, self.image_filenames[index])
        img = Image.open(img_fn).convert('RGB')
        # preprocessing
        if self.resize_scale != None:
            img = img.resize((self.resize_scale[0], self.resize_scale[1]), Image.BILINEAR)
        if self.crop_size:
            x = np.random.randint(0, img.width - self.crop_size + 1)
            y = np.random.randint(0, img.height - self.crop_size + 1)
            img = img.crop((x, y, x + self.crop_size, y + self.crop_size))
        if self.fliplr:
            if np.random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if self.transform is not None:
            img = self.transform(img)
        return img
    def __len__(self):
        return len(self.image_filenames)


class ImagePool():
    """
    Used to implement a replay buffer for CycleGAN.

    Notes
    -----
    Original code:
    https://github.com/togheppi/CycleGAN/blob/master/utils.py
    """
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        from torch.autograd import Variable
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images.data:
            image = torch.unsqueeze(image, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = np.random.uniform(0, 1)
                if p > 0.5:
                    random_id = np.random.randint(0, self.pool_size-1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = Variable(torch.cat(return_images, 0))
        return return_images
