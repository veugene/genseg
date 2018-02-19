import unittest
from unittest import TestCase
import numpy as np
import os
import util
import iterators

class TestIdridSegmentationIterator(TestCase):
    def test(self):
        # TODO: this method currently only returns one iterator
        batch_size = 2
        it_train = iterators.get_idrid_seg_folder_iterators(batch_size)
        aa, bb = iter(it_train).next()
        assert list(aa.size()) == [batch_size,3,256,256]
        assert list(bb.size()) == [batch_size,3,256,256]
        
class TestKaggleIterator(TestCase):
    def test(self):
        batch_size = 2
        it_train, it_val = iterators.get_kaggle_folder_iterators(batch_size)
        aa, bb = iter(it_train).next()
        assert list(aa.size()) == [batch_size,3,256,256]
        assert list(bb.size()) == [batch_size,3,256,256]

if __name__ == '__main__':
    unittest.main()
