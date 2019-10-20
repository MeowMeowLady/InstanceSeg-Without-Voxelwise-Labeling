# -*- coding: utf-8 -*-
"""
Created on 19-2-28 下午4:55
IDE PyCharm 

@author: Meng Dong
"""
import numpy as np
import numba as nb

def mask_iou(mask_a, mask_b):
    """Calculate the Intersection of Unions (IoUs) between masks.
    IoU is calculated as a ratio of area of the intersection
    and area of the union.
    This function accepts both :obj:`numpy.ndarray` and :obj:`cupy.ndarray` as
    inputs. Please note that both :obj:`mask_a` and :obj:`mask_b` need to be
    same type.
    The output is same type as the type of the inputs.
    Args:
        mask_a (array): An array whose shape is :math:`(N, H, W)`.
            :math:`N` is the number of masks.
            The dtype should be :obj:`numpy.bool`.
        mask_b (array): An array similar to :obj:`mask_a`,
            whose shape is :math:`(K, H, W)`.
            The dtype should be :obj:`numpy.bool`.
    Returns:
        array:
        An array whose shape is :math:`(N, K)`. \
        An element at index :math:`(n, k)` contains IoUs between \
        :math:`n` th mask in :obj:`mask_a` and :math:`k` th mask \
        in :obj:`mask_b`.
    """
    if mask_a.shape[1:] != mask_b.shape[1:]:
        raise IndexError
    #xp = cuda.get_array_module(mask_a)
    n_mask_a = len(mask_a)
    n_mask_b = len(mask_b)
    iou = np.empty((n_mask_a, n_mask_b), dtype=np.float32)
    for n, m_a in enumerate(mask_a):
        for k, m_b in enumerate(mask_b):
            intersect = np.sum(m_a & m_b)#np.sum(bool_and(m_a, m_b))#xp.bitwise_and(m_a, m_b).sum()
            union = np.sum((m_a | m_b)) #xp.bitwise_or(m_a, m_b).sum()
            iou[n, k] = intersect / union
    return iou

"""
'_fast' means the improved version for higher running speed
"""
@nb.jit(nopython=True)
def mask_iou_fast(mask_a, mask_b):
    iou = np.empty((mask_a.shape[0], mask_b.shape[0]), dtype=np.float32)
    s, h, w = mask_a.shape[1:]
    for n in range(mask_a.shape[0]):
        for k in range(mask_b.shape[0]):
            m_a = mask_a[n, :]
            m_b = mask_b[k, :]
            intersect = 0.0
            union = 0.0
            for ss in range(s):
                for hh in range(h):
                    for ww in range(w):
                        if m_a[ss, hh, ww] and m_b[ss, hh, ww]:
                            intersect += 1
                        if m_a[ss, hh, ww] or m_b[ss, hh, ww]:
                            union += 1
            #print(intersect, union)
            iou[n, k] = intersect / union
    return iou

@nb.jit(nopython=True)
def mask_ios_fast(mask_a, mask_b):
    ios = np.empty((mask_a.shape[0], mask_b.shape[0]), dtype=np.float32)
    s, h, w = mask_a.shape[1:]
    for n in range(mask_a.shape[0]):
        for k in range(mask_b.shape[0]):
            m_a = mask_a[n, :]
            m_b = mask_b[k, :]
            intersect = 0.0
            seg = 0.0
            for ss in range(s):
                for hh in range(h):
                    for ww in range(w):
                        if m_a[ss, hh, ww] and m_b[ss, hh, ww]:
                            intersect += 1
                        if m_a[ss, hh, ww]:
                            seg += 1
            #print(intersect, union)
            ios[n, k] = intersect / seg
    return ios

@nb.jit(nopython=True)
def mask_iog_fast(mask_a, mask_b):
    iog = np.empty((mask_a.shape[0], mask_b.shape[0]), dtype=np.float32)
    s, h, w = mask_a.shape[1:]
    for n in range(mask_a.shape[0]):
        for k in range(mask_b.shape[0]):
            m_a = mask_a[n, :]
            m_b = mask_b[k, :]
            intersect = 0.0
            gt = 0.0
            for ss in range(s):
                for hh in range(h):
                    for ww in range(w):
                        if m_a[ss, hh, ww] and m_b[ss, hh, ww]:
                            intersect += 1
                        if m_b[ss, hh, ww]:
                            gt += 1
            #print(intersect, union)
            iog[n, k] = intersect / gt
    return iog