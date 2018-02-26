from PIL import Image
from data_tools.wrap import (delayed_view,
                             multi_source_array)
from data_tools.io import data_flow
import os

class folder_indexer(object):
    """
    Single-value index interface for loading images from a folder.
    """
    def __init__(self, path, convert_rgb=True):
        self.path = path
        self.convert_rgb = convert_rgb
        self.fn_list = sorted(os.listdir(path))
        self.shape = (len(self.fn_list),)
        self.dtype = 'strings'
      
    def __getitem__(self, index):
        fn = self.fn_list[index]
        img = Image.open("%s/%s" % (self.path,fn))
        if self.convert_rgb:
            img = img.convert('RGB')
        return img
    
    def __len__(self):
        return len(self.fn_list)


class data_flow_sampler(data_flow):
    """
    A data_flow implementation that does not restrict every source in `data`
    to be of equal length. 
    
    Whenever an iterator is created by data_flow (eg. each epoch), a random
    (equal length) non-contiguous subset of data is viewed from each source
    that is longer than the shortest length.
    
    NOTE: this means that these sources are shuffled, regardless of which
          sampling arguments are passed.
    """
    def __init__(self, data, *args, **kwargs):
        min_length = min(len(d) for d in data)
        _data = []
        self.reshuffle_indicator = []
        for d in data:
            if len(d) > min_length:
                d = delayed_view(d, shuffle=True)
                d = delayed_view(d, idx_min=0, idx_max=min_length)
                self.reshuffle_indicator.append(True)
            else:
                self.reshuffle_indicator.append(False)
            _data.append(d)
            
        super(data_flow_sampler, self).__init__(data=_data, *args, **kwargs)
        
    def flow(self):
        for i, reshuffle in enumerate(self.reshuffle_indicator):
            if reshuffle:
                self.data[i].arr.re_shuffle()
        return super(data_flow_sampler, self).flow()
