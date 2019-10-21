# -*- coding: utf-8 -*-
"""
Created on 18-11-30 下午3:49
IDE PyCharm 

@author: Meng Dong
"""

import numpy as np
from itertools import groupby
from operator import mul
from functools import reduce


def binary_mask_to_rle(binary_mask):
    """
    :param binary_mask: 2D or 3D
    :return: a dictionary {'counts': rle, 'size': mask.shape}
    """
    fortran_binary_mask = np.asfortranarray(binary_mask)
    fortran_binary_mask_ravel = np.ravel(fortran_binary_mask, order='F')#or x.T.flatten()
    rle = {'counts': [], 'size': list(fortran_binary_mask.shape)}
    counts = rle.get('counts')
    '''
    for i, (value, elements) in enumerate(groupby(fortran_binary_mask_ravel)):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    '''
    # a faster algorithm
    dots = np.where(fortran_binary_mask_ravel)[0]
    if len(dots)==0:
        counts.append(fortran_binary_mask.size)
        return rle
    prev = -1
    for i in dots:
        if i > prev+1:
            counts.extend((i - prev - 1, 0))
        elif i==0:
            counts.extend((0,))
        counts[-1] += 1
        prev = i
    if dots[0] == 0:
        counts.insert(0, 0)
    if prev+1 < len(fortran_binary_mask_ravel):
        counts.append(len(fortran_binary_mask_ravel)-prev-1)
    return rle

def rle_to_binary_mask(rle):
    """
    :param rle: the return from  binary_mask_to_rle
    :return: a numpy.array (0,1) as binary mask, 2D or 3D
    """
    rle_counts = rle['counts']
    rle_size = rle['size']
    assert sum(rle_counts) == reduce(mul, rle_size, 1)
    mask = np.zeros(reduce(mul, rle_size, 1), dtype=np.uint8)
    N = len(rle_counts)
    n = 0
    val = 1
    if N == 1: # no mask, blank
        binary_mask = mask.reshape(rle_size, order='F')
        return binary_mask
    for pos in range(N):
        val = not val
        for c in range(rle_counts[pos]):
            mask[n] = val
            n += 1

    binary_mask = mask.reshape(rle_size, order='F')
    return binary_mask



if __name__ == '__main__':
    a = np.array([[[1,1,1,0,0,0], [1,1,1, 0,0,0]],[[1,1,1,1,1,0], [1,1,1,0,0,0]]])
    r = binary_mask_to_rle(a)
    b = rle_to_binary_mask(r)
    print(r)
    print(b)