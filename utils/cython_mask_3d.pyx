# -*- coding: utf-8 -*-
"""
Created on 18-11-30 下午3:49
IDE PyCharm

@author: Meng Dong
"""

cimport cython
import numpy as np
cimport numpy as np
from itertools import groupby
from operator import mul
from functools import reduce

DTYPE = np.uint8
ctypedef np.uint8_t DTYPE_t

def binary_mask_to_rle(np.ndarray[DTYPE_t, ndim=3] binary_mask):
    """
    :param binary_mask: 3D
    :return: a dictionary {'counts': rle, 'size': mask.shape}
    """
    cdef int i, value, prev
    cdef dict rle
    cdef np.ndarray[DTYPE_t, ndim=3] fortran_binary_mask
    cdef np.ndarray[DTYPE_t, ndim=1] fortran_binary_mask_ravel
    cdef np.ndarray[long, ndim=1] dots
    cdef list counts
    cdef object elements
    fortran_binary_mask = np.asfortranarray(binary_mask)
    rle = {'counts': [], 'size': list((<object>fortran_binary_mask).shape)}
    counts = rle.get('counts')
    fortran_binary_mask_ravel = np.ravel(fortran_binary_mask, order='F')
    dots = np.where(fortran_binary_mask_ravel)[0]
    if len(dots)==0:
        counts.append(fortran_binary_mask.size)
        return rle
    prev = -1
    for i in dots:
        if i > prev+1:
            counts.extend((i - prev - 1, 0))
        elif i == 0:
            counts.append(0)
        counts[-1] += 1
        prev = i
    if dots[0] == 0:
        counts.insert(0, 0)
    if prev+1 < len(fortran_binary_mask_ravel):
        counts.append(len(fortran_binary_mask_ravel)-prev-1)
    return rle

def rle_to_binary_mask(dict rle):
    """
    :param rle: the return from  binary_mask_to_rle
    :return: a numpy.array (0,1) as binary mask, 2D or 3D
    """
    cdef list rle_counts = rle['counts']
    cdef list rle_size = rle['size']
    assert sum(rle_counts) == reduce(mul, rle_size, 1)
    cdef DTYPE_t[:] mask = np.zeros(reduce(mul, rle_size, 1), dtype=np.uint8)
    cdef int N = len(rle_counts)
    cdef int n = 0
    cdef DTYPE_t val = 1
    cdef pos
    cdef np.ndarray[DTYPE_t, ndim=3] binary_mask = np.zeros(rle_size, dtype=DTYPE)
    if N == 1: # no mask, blank
        binary_mask = np.reshape(mask, rle_size, order='F')
        return binary_mask
    for pos in range(N):
        val = not val
        for c in range(rle_counts[pos]):
            mask[n] = val
            n += 1

    binary_mask = np.reshape(mask, rle_size, order='F')
    return binary_mask
