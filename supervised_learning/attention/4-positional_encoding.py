#!/usr/bin/env python3
'''cfvdfdfbfgbgbg'''

import numpy as np


def positional_encoding(max_seq_len, dm):
    '''kvhjyfhfukmkm'''
    angle_rads = np.arange(max_seq_len)[:, np.newaxis] / np.power(
        10000, (2 * (np.arange(dm)[np.newaxis, :] // 2)) / np.float32(dm))
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[...]

    return angle_rads
