import unittest
from unittest import TestCase
import numpy as np
import os
import util
import iterators

import torch
from torch.autograd import Variable
from torch import nn
import util
from architectures import image2image

class TestDilatedFCN(TestCase):
    def get_fcn(self, **kwargs):
        fcn = image2image.DilatedFCN(C=3, norm_layer=nn.InstanceNorm2d, **kwargs)
        #print(fcn)
        print("# params: %i" % util.count_params(fcn))
        return fcn
        
    def test_with_reflection_padding(self):
        fcn = self.get_fcn(padding_type='reflect')
        xfake = Variable( torch.randn((1,3,256,256)) )
        out = fcn(xfake)
        # should maintain the output shape
        assert list(out.size()) == [1,3,256,256]
        
    def test_with_zero_padding(self):
        fcn = self.get_fcn(padding_type='zero')
        xfake = Variable( torch.randn((1,3,256,256)) )
        out = fcn(xfake)
        # should maintain the output shape
        assert list(out.size()) == [1,3,256,256]
