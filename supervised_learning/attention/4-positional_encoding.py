#!/usr/bin/env python3
'''cfvdfdfbfgbgbg'''

import numpy as np


def positional_encoding(max_seq_len, dm):
    '''kvhjyfhfukmkm'''
    angle_rads = np.arange(position)[:, np.newaxis] / np.power(
        10000, (2 * (np.arange(d_model)[np.newaxis, :] // 2)) / 
            np.float32(d_model)
    )
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[...]

    return angle_rads
