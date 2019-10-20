# -*- coding: utf-8 -*-
"""
Created on 19-3-5 下午4:20
IDE PyCharm 

@author: Meng Dong

this function is otsu binarization algorithm
reference: https://pdfs.semanticscholar.org/fa29/610048ae3f0ec13810979d0f27ad6971bdbf.pdf
also refers skimage threshold_otsu and matlab's otsuthresh
support 3d image
"""
import numpy as np
import math
from matplotlib import pyplot as plt
from skimage import io, exposure

def otsu_py(image, nbins=256):
    """Return threshold value based on Otsu's method.
    Parameters
    ----------
    image : (S, H, W) ndarray
        Grayscale input image.
    nbins : int, optional
        Number of bins used to calculate histogram. This value is ignored for
        integer arrays.
    Returns
    -------
    threshold : float
        Upper threshold value. All pixels with an intensity higher than
        this value are assumed to be foreground.
    Raises
    ------
    ValueError
         If `image` only contains a single grayscale value.
    ------
    Exmples:
    thresh = threshold_otsu(image)
    binary = image <= thresh
    ------
    Notes
    -----
    The input image must be grayscale.
    """
    np.seterr(divide='ignore', invalid='ignore')
    hist, bin_edges = np.histogram(image.ravel(), nbins, range=None)
    hist = hist.astype(float)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # class probabilities for all possible thresholds
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    # class means for all possible thresholds
    mean1 = np.cumsum(hist * bin_centers) / weight1
    mean2 = (np.cumsum((hist * bin_centers)[::-1]) / weight2[::-1])[::-1]

    # Clip ends to align class 1 and class 2 variables:
    # The last value of `weight1`/`mean1` should pair with zero values in
    # `weight2`/`mean2`, which do not exist.
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    idx = np.argmax(variance12)
    threshold = bin_centers[:-1][idx]
    return threshold

def otsu_mat(image, nbins=256):
    """Return threshold value based on Otsu's method.
    Parameters
    ----------
    image : (S, H, W) ndarray
        Grayscale input image.
    nbins : int, optional
        Number of bins used to calculate histogram. This value is ignored for
        integer arrays.
    Returns
    -------
    threshold : float
        Upper threshold value. All pixels with an intensity higher than
        this value are assumed to be foreground.
    Raises
    ------
    ValueError
         If `image` only contains a single grayscale value.
    ------
    Exmples:
    thresh = threshold_otsu(image)
    binary = image <= thresh
    ------
    Notes
    -----
    The input image must be grayscale. I can't get the same histogram result as matlab's imhist,
    so the threshold does not match otsu_py.
    """
    np.seterr(divide='ignore', invalid='ignore')
    img_range = (np.iinfo(image.dtype).min, np.iinfo(image.dtype).max)
    hist, bin_edges = np.histogram(image.ravel(), nbins, range=img_range)
    hist = hist.astype(float)

    #Variables names are chosen to be similar to the formulas in the Otsupaper.
    p = hist / np.sum(hist)
    omega = np.cumsum(p)
    mu = np.cumsum(p* range(nbins))
    mu_t = mu[-1]
    sigma_b_squared = (mu_t * omega - mu)**2. / (omega * (1 - omega))
    # Find the location of the maximum value of sigma_b_squared.
    # The maximum may extend over several bins, so average together the locations.If maxval is NaN,
    #  meaning that sigma_b_squared is all  NaN, then return 0.
    maxval = np.nanmax(sigma_b_squared)
    isinf_maxval = math.isinf(maxval)
    if not isinf_maxval:
        idx = np.mean(np.where(sigma_b_squared == maxval)[0])
        # Normalize the threshold to the range[0, 1].
        threshold = (idx - 1) / (nbins - 1) * img_range[1]
    else:
        threshold = 0.0
    return threshold

def otsu_py_2d(image, prm, nbins=256):
    """Return threshold value based on Otsu's method.
    Parameters
    ----------
    image : (S, H, W) ndarray
        Grayscale input image.
    prm : (S, H, W) ndarray
        hint information for image.
    nbins : int, optional
        Number of bins used to calculate histogram. This value is ignored for
        integer arrays.
    Returns
    -------
    img_bi : (S, H, W) bool ndarray
    k, b: oblique line's parameters
    Raises
    ------
    ValueError
         If `image` only contains a single grayscale value.
    ------
    Notes
    -----
    The input image must be grayscale.
    refer: http://andrei.clubcisco.ro/cursuri/f/f-sym/5master/analiza-extragerea-continutului/
                    ... prezentari/ImageSegmentationBasedon2DOtsuMethod.pdf
    """
    np.seterr(divide='ignore', invalid='ignore')
    g_min, g_max = int(np.min(image)), int(np.max(image))
    g_range = g_max - g_min + 1
    hist, bin_edges1, bin_edges2 = np.histogram2d(image.ravel(), prm.ravel(), bins = g_range, range=None)
    bin_centers1 = (bin_edges1[:-1] + bin_edges1[1:]) / 2
    bin_centers2 = (bin_edges2[:-1] + bin_edges2[1:]) / 2
    hist = hist.T # each row has the same prm value
    img_hist,  prm_hist= np.meshgrid(bin_centers1, bin_centers2)

    prob = hist.copy().astype(np.float)/np.sum(hist)
    u_img_hist = img_hist*prob
    u_prm_hist = prm_hist*prob

    ut = np.array([0., 0.])[np.newaxis, :].T  # for whole image
    ut[0] = np.sum(u_img_hist)
    ut[1] = np.sum(u_prm_hist)

    k = -1 # for the gray scale and prm values are normalized into the similar range
    var_b_max = 0
    b_max = 0
    for b in range(2*g_min+1, 2*g_max-1):
        x_g_min = int((g_min - b) / k)
        x_g_max = int((g_max - b) / k)
        p0 = 0.
        p1 = 0.
        u0 = np.array([0., 0.])[np.newaxis, :].T #average value for background's two attributes [img, prm]
        u1 = np.array([0., 0.])[np.newaxis, :].T #for forground
        x0 = range(g_min, np.minimum(x_g_min, g_max))
        for ix in x0:
            y0 = list(range(g_min, np.minimum(int(k*ix+b), g_max+1)))
            y0 = [x-g_min for x in y0]
            line = np.zeros(g_range, dtype=bool)
            line[y0] = 1
            p0 += np.sum(prob[line, ix-g_min])
            u0[0] += np.sum(u_img_hist[line, ix-g_min])
            u0[1] += np.sum(u_prm_hist[line, ix-g_min])
        p1 = 1. - p0

        u1[0] = (ut[0] - p0*u0[0])/p1
        u1[1] = (ut[1] - p0 * u0[1]) / p1
        var_b = np.trace(p0*(u0-ut)*(u0-ut).T + p1*(u1-ut)*(u1-ut).T)
        if var_b > var_b_max:
            var_b_max = var_b
            b_max = b

    img_bi = np.ones(image.shape, dtype=bool)
    x_g_min = int((g_min - b_max) / k)
    x0 = range(g_min, np.minimum(x_g_min, g_max))
    for ix in x0:
        y0_up = np.minimum(int(k*ix+b_max), g_max+1)
        img_bi[(image == ix)&(prm < y0_up)] = 0
    img_bi = img_bi.astype(np.uint8)*255

    return img_bi, k, b_max

def otsu_py_2d_fast(image, prm, b_range=None):
    np.seterr(divide='ignore', invalid='ignore')
    g_min, g_max = int(np.min(image)), int(np.max(image))
    g_range = g_max - g_min + 1
    hist, bin_edges1, bin_edges2 = np.histogram2d(image.ravel(), prm.ravel(), bins = g_range, range=None)
    bin_centers1 = (bin_edges1[:-1] + bin_edges1[1:]) / 2
    bin_centers2 = (bin_edges2[:-1] + bin_edges2[1:]) / 2
    hist = hist.T # each row has the same prm value
    img_hist,  prm_hist= np.meshgrid(bin_centers1, bin_centers2)

    prob = hist.copy().astype(np.float)/np.sum(hist)
    u_img_hist = img_hist*prob
    u_prm_hist = prm_hist*prob


    ut = np.array([0., 0.])[np.newaxis, :].T  # for background

    ut[0] = np.sum(u_img_hist)
    ut[1] = np.sum(u_prm_hist)
    k_list = [-1,]
    var_b_max = 0
    b_max = 0
    for k in k_list:
        p0 = 0
        u0 = np.array([0., 0.])[np.newaxis, :].T  # average value for background's two attributes [img, prm]
        u1 = np.array([0., 0.])[np.newaxis, :].T  # for forground
        if b_range is None:
            b_dw = int((1.-k)*g_min+1)
            b_up = int((1.-k)*g_max-1)
        else:
            b_dw = b_range[0]
            b_up = b_range[1]
        b = b_dw

        x_range = np.linspace(g_min, g_max-1, g_max - g_min)
        line = (k*x_range+b).astype(int)
        y_range = np.maximum(np.minimum(line, g_max) - g_min, 0)
        x_range = x_range.astype(int)
        y_range = y_range.astype(int)
        bg_range = y_range > 0
        for ix, iy in zip(x_range[bg_range], y_range[bg_range]):
            p0 += np.sum(prob[: iy, ix-g_min])
            u0[0] += np.sum(u_img_hist[: iy, ix-g_min])
            u0[1] += np.sum(u_prm_hist[: iy, ix-g_min])
        p1 = 1. - p0
        u1[0] = (ut[0] - p0 * u0[0]) / p1
        u1[1] = (ut[1] - p0 * u0[1]) / p1
        var_b = np.trace(p0 * (u0 - ut) * (u0 - ut).T + p1 * (u1 - ut) * (u1 - ut).T)
        if var_b > var_b_max:
            var_b_max = var_b
            b_max = b
            k_max = k
        for b in range(b_dw+1, b_up):
            line = (k * x_range + b).astype(int)
            y_range_new = np.maximum(np.minimum(line, g_max) - g_min, 0)
            margin = y_range_new - y_range
            bg_range_new = margin>0
            ovlp = bg_range&bg_range_new
            p0 += np.sum(prob[y_range[ovlp], x_range[ovlp]-g_min])
            u0[0] += np.sum(u_img_hist[y_range[ovlp]+1-1, x_range[ovlp]-g_min])
            u0[1] += np.sum(u_prm_hist[y_range[ovlp]+1-1, x_range[ovlp]-g_min])
            bg_add = bg_range_new^ovlp
            for ix, iy in zip(x_range[bg_add], y_range_new[bg_add]):
                p0 += np.sum(prob[: iy, ix - g_min])
                u0[0] += np.sum(u_img_hist[: iy, ix - g_min])
                u0[1] += np.sum(u_prm_hist[: iy, ix - g_min])
            y_range = y_range_new.copy()
            bg_range = y_range > 0
            p1 = 1. - p0
            u1[0] = (ut[0] - p0 * u0[0]) / p1
            u1[1] = (ut[1] - p0 * u0[1]) / p1
            var_b = np.trace(p0 * (u0 - ut) * (u0 - ut).T + p1 * (u1 - ut) * (u1 - ut).T)
            if var_b > var_b_max:
                var_b_max = var_b
                b_max = b
                k_max = k

    img_bi = np.ones(image.shape, dtype=bool)
    x_g_min = int((g_min - b_max) / k_max)
    x0 = range(g_min, np.minimum(x_g_min, g_max))
    for ix in x0:
        y0_up = np.minimum(int(k_max * ix + b_max), g_max + 1)
        img_bi[(image == ix) & (prm < y0_up)] = 0
    img_bi = img_bi.astype(np.uint8) * 255

    return img_bi, k_max, b_max

