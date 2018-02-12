import math

import numpy

from chainer.backends import cuda
from chainer import function_node
from chainer import utils
from chainer.utils import type_check


class STFT(function_node.FunctionNode):


    def __init__(self, frame_length, hop, fft_size=None, return_onesided=True, window=None, pad_end=0):
        pass

    @property
    def label(self):
        return 'stft'

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward(self, inputs):
        xp = cuda.get_array_module(x)
