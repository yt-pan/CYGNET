"""Core CYGNET statistical routines and matrix helpers."""

import numpy as np
from chiscore import liu_sf, davies_pvalue
import pandas as pd
import os
import shutil
import subprocess
import tempfile
from glob import glob
import warnings
from scipy.stats import chi2
from scipy.linalg import block_diag
from scipy.spatial.distance import pdist


_NONLINEAR_ENV_TYPES = {"mgcv", "non-linear", "nonlinear", "tp"}
_STATSMODELS_ENV_TYPES = {"statsmodels", "bspline", "bsplines", "b-spline", "b-splines"}


def _matrix_rank(matrix, tol=None):
    """Return matrix rank without calling platform BLAS/LAPACK routines.

    Some Windows NumPy builds can terminate the interpreter inside the small
    SVD used by ``numpy.linalg.matrix_rank``. CYGNET only needs this check for
    narrow fixed-effect matrices, where pivoted Gaussian elimination uses the
    same tolerance and avoids that platform-specific failure.
    """
    work = np.asarray(matrix, dtype=float).copy()
    if work.size == 0:
        return 0
    m, n = work.shape
    if tol is None:
        scale = np.max(np.abs(work))
        tol = max(m, n) * np.finfo(float).eps * scale
    if tol == 0:
        return 0

    rank = 0
    row = 0
    for col in range(n):
        pivot_offset = int(np.argmax(np.abs(work[row:, col]))) if row < m else 0
        pivot = row + pivot_offset
        if row >= m or abs(work[pivot, col]) <= tol:
            continue
        if pivot != row:
            work[[row, pivot], :] = work[[pivot, row], :]
        work[row, :] = work[row, :] / work[row, col]
        if row + 1 < m:
            # Eliminate all remaining rows in one NumPy operation to avoid a
            # Python loop over tall spatial-data design matrices.
            factors = work[row + 1 :, col].copy()
            active = np.abs(factors) > tol
            trailing = work[row + 1 :, :]
            trailing[active, :] -= factors[active, None] * work[row, :]
        rank += 1
        row += 1
        if row == m:
            break
    return rank


def _raise_rank_error(matrix_name, matrix):
    """Raise a useful fixed-design rank error for CYGNET callers."""
    rank = _matrix_rank(matrix)
    raise ValueError(
        f"{matrix_name} is not column full rank after adding the tested cell-type vector: "
        f"rank={rank}, columns={matrix.shape[1]}. Remove collinear covariates or one redundant "
        "cell-type/composition column before running CYGNET."
    )


def cygnet_null_multiK(y, X, kernel_lst, maxiter=100):
    """Estimate null-model variance components for multiple low-rank kernels."""
    # print("Sample variances of y: {}".format(np.var(y)))
    converged = False
    n = y.shape[0]
    posi_lst = [X.shape[1]]
    for kernel in kernel_lst:
        posi_lst.append(kernel.shape[1] + posi_lst[-1])
    W = np.concatenate([X] + kernel_lst, axis=1)
    CMnull = np.dot(W.T, W)
    Wty = np.dot(W.T, y)
    num_kernel = len(kernel_lst)
    varcom = np.ones(num_kernel + 1)
    cc_par = 1000
    for i in range(maxiter):
        # print("Iterations: {}".format(i+1))
        # MME
        CM = CMnull / varcom[-1]
        for k in range(num_kernel):
            start = posi_lst[k]
            end = posi_lst[k+1]
            q = end - start
            CM[start:end, start:end] += np.identity(q) / varcom[k]
        CMi = np.linalg.inv(CM)
        WtRiy = Wty / varcom[-1]

        # effect
        beta = np.dot(CMi, WtRiy)
        ehat = y - np.dot(W, beta)
        # Gradient
        gradArr = np.ones(num_kernel+1)
        for k in range(num_kernel):
            start = posi_lst[k]
            end = posi_lst[k + 1]
            q = end - start
            betak = beta[start:end, :]
            gradArr[k] = q / varcom[k] - np.trace(CMi[start:end, start:end]) / (varcom[k] * varcom[k]) - \
                         np.sum(np.dot(betak.T, betak)) / (varcom[k] * varcom[k])
        gradArr[-1] = n / varcom[-1] - np.sum(CMnull * CMi) / (varcom[-1] * varcom[-1]) - \
                     np.sum(np.dot(ehat.T, ehat)) / (varcom[-1] * varcom[-1])
        gradArr *= -0.5

        # AI
        wv = np.ones((n, num_kernel+1))
        for k in range(num_kernel):
            start = posi_lst[k]
            end = posi_lst[k + 1]
            betak = beta[start:end, 0]
            wv[:, k] = np.dot(kernel_lst[k], betak) / varcom[k]
        wv[:, -1] = ehat[:, 0] / varcom[-1]
        Wtwv = np.dot(W.T, wv)
        ai = np.dot(wv.T, wv) / varcom[-1] - np.dot(np.dot(Wtwv.T, CMi), Wtwv) / (varcom[-1] * varcom[-1])
        ai *= 0.5
        # EM
        em = np.zeros((num_kernel+1, num_kernel+1))
        for k in range(num_kernel):
            q = posi_lst[k + 1] - posi_lst[k]
            em[k, k] = q / (2 * varcom[k] * varcom[k])
        em[-1, -1] = n / (2 * varcom[-1] * varcom[-1])
        varcom_update = np.zeros(num_kernel+1)
        delta = np.zeros(num_kernel+1)
        for k in range(101):
            gamma = k * 0.01
            wemai = (1 - gamma) * ai + gamma * em
            delta = np.dot(np.linalg.inv(wemai), gradArr)
            varcom_update = varcom + delta
            if np.min(varcom_update) > 0:
                # print("Weighted value: {}".format(gamma))
                break
        # print("Updated variance: {}".format(varcom_update))
        cc_par = np.sqrt(np.sum(delta * delta / varcom_update))
        # print("cc: {}".format(cc_par))
        if cc_par < 1e-8:
            converged = True
            break
        varcom = varcom_update
    # if cc_par < 1.0e-8:
        #print("Variances converged")
    return varcom, cc_par


def cygnet_null_multiK_checkiter(y, X, kernel_lst, maxiter=100):
    """Estimate null-model variance components and return the final iteration index."""
    # print("Sample variances of y: {}".format(np.var(y)))
    converged = False
    n = y.shape[0]
    posi_lst = [X.shape[1]]
    for kernel in kernel_lst:
        posi_lst.append(kernel.shape[1] + posi_lst[-1])
    W = np.concatenate([X] + kernel_lst, axis=1)
    CMnull = np.dot(W.T, W)
    Wty = np.dot(W.T, y)
    num_kernel = len(kernel_lst)
    varcom = np.ones(num_kernel + 1)
    cc_par = 1000
    for i in range(maxiter):
        # print("Iterations: {}".format(i+1))
        # MME
        CM = CMnull / varcom[-1]
        for k in range(num_kernel):
            start = posi_lst[k]
            end = posi_lst[k+1]
            q = end - start
            CM[start:end, start:end] += np.identity(q) / varcom[k]
        CMi = np.linalg.inv(CM)
        WtRiy = Wty / varcom[-1]

        # effect
        beta = np.dot(CMi, WtRiy)
        ehat = y - np.dot(W, beta)
        # Gradient
        gradArr = np.ones(num_kernel+1)
        for k in range(num_kernel):
            start = posi_lst[k]
            end = posi_lst[k + 1]
            q = end - start
            betak = beta[start:end, :]
            gradArr[k] = q / varcom[k] - np.trace(CMi[start:end, start:end]) / (varcom[k] * varcom[k]) - \
                         np.sum(np.dot(betak.T, betak)) / (varcom[k] * varcom[k])
        gradArr[-1] = n / varcom[-1] - np.sum(CMnull * CMi) / (varcom[-1] * varcom[-1]) - \
                     np.sum(np.dot(ehat.T, ehat)) / (varcom[-1] * varcom[-1])
        gradArr *= -0.5

        # AI
        wv = np.ones((n, num_kernel+1))
        for k in range(num_kernel):
            start = posi_lst[k]
            end = posi_lst[k + 1]
            betak = beta[start:end, 0]
            wv[:, k] = np.dot(kernel_lst[k], betak) / varcom[k]
        wv[:, -1] = ehat[:, 0] / varcom[-1]
        Wtwv = np.dot(W.T, wv)
        ai = np.dot(wv.T, wv) / varcom[-1] - np.dot(np.dot(Wtwv.T, CMi), Wtwv) / (varcom[-1] * varcom[-1])
        ai *= 0.5
        # EM
        em = np.zeros((num_kernel+1, num_kernel+1))
        for k in range(num_kernel):
            q = posi_lst[k + 1] - posi_lst[k]
            em[k, k] = q / (2 * varcom[k] * varcom[k])
        em[-1, -1] = n / (2 * varcom[-1] * varcom[-1])
        varcom_update = np.zeros(num_kernel+1)
        delta = np.zeros(num_kernel+1)
        for k in range(101):
            gamma = k * 0.01
            wemai = (1 - gamma) * ai + gamma * em
            delta = np.dot(np.linalg.inv(wemai), gradArr)
            varcom_update = varcom + delta
            if np.min(varcom_update) > 0:
                # print("Weighted value: {}".format(gamma))
                break
        # print("Updated variance: {}".format(varcom_update))
        cc_par = np.sqrt(np.sum(delta * delta / varcom_update))
        # print("cc: {}".format(cc_par))
        if cc_par < 1e-8:
            converged = True
            break
        varcom = varcom_update
    # if cc_par < 1.0e-8:
        #print("Variances converged")
    return varcom, cc_par, i


def cygnet(y, X, G, E, kernel_lst, maxiter=100): # test
    """Run the CYGNET score test using Liu's moment-matching p-value."""
    y = np.asarray(y, float).reshape(-1, 1) # -1: non constraint on row number
    n = y.shape[0]

    X = np.asarray(X, float).reshape(n, -1) # -1: non constraint on col number
    # X = np.concatenate([np.ones((n, 1)), X], axis=1)
    G = np.asarray(G, float).reshape(n, 1) 
    X = np.concatenate([X, G], axis=1) 
    if _matrix_rank(X) != X.shape[1]:
        _raise_rank_error("X", X)
    q = E.shape[1]

    varcom, cc_par = cygnet_null_multiK(y, X, kernel_lst, maxiter=maxiter) # estimate variance component
    # print("Estimated variances: {}".format(varcom))

    # MME
    num_kernel = len(kernel_lst)
    posi_lst = [X.shape[1]]
    for kernel in kernel_lst:
        posi_lst.append(kernel.shape[1] + posi_lst[-1])
    W = np.concatenate([X] + kernel_lst, axis=1)
    CMnull = np.dot(W.T, W)
    CM = CMnull / varcom[-1]
    for k in range(num_kernel):
        start = posi_lst[k]
        end = posi_lst[k + 1]
        CM[start:end, start:end] += np.identity(end - start) / varcom[k]
    CMi = np.linalg.inv(CM)

    EG = E * G  # diag(G)E
    P_EG = EG / varcom[-1] - np.dot(W, np.dot(CMi, np.dot(W.T, EG))) / (varcom[-1] * varcom[-1])
    P_EGt_y = np.dot(P_EG.T, y)  # diag(g)EPy
    score = np.sum(np.dot(P_EGt_y.T, P_EGt_y))
    EGtPEG = np.dot(EG.T, P_EG)
    a = np.linalg.eigvalsh(EGtPEG)
    (p_val, dof, delta, _) = liu_sf(score, a, np.ones(q), np.zeros(q))  

    return a, score, p_val, cc_par


def cygnet_davies(y, X, G, E, kernel_lst, maxiter=100): # test
    """Run the CYGNET score test using Davies' p-value with Liu fallback."""
    y = np.asarray(y, float).reshape(-1, 1) # -1: non constraint on row number
    n = y.shape[0]

    X = np.asarray(X, float).reshape(n, -1) # -1: non constraint on col number
    # X = np.concatenate([np.ones((n, 1)), X], axis=1)
    G = np.asarray(G, float).reshape(n, 1) 
    X = np.concatenate([X, G], axis=1) 
    if _matrix_rank(X) != X.shape[1]:
        _raise_rank_error("X", X)
    q = E.shape[1]

    varcom, cc_par = cygnet_null_multiK(y, X, kernel_lst, maxiter=maxiter) # estimate variance component
    # print("Estimated variances: {}".format(varcom))

    # MME
    num_kernel = len(kernel_lst)
    posi_lst = [X.shape[1]]
    for kernel in kernel_lst:
        posi_lst.append(kernel.shape[1] + posi_lst[-1])
    W = np.concatenate([X] + kernel_lst, axis=1)
    CMnull = np.dot(W.T, W)
    CM = CMnull / varcom[-1]
    for k in range(num_kernel):
        start = posi_lst[k]
        end = posi_lst[k + 1]
        CM[start:end, start:end] += np.identity(end - start) / varcom[k]
    CMi = np.linalg.inv(CM)

    EG = E * G  # diag(G)E
    P_EG = EG / varcom[-1] - np.dot(W, np.dot(CMi, np.dot(W.T, EG))) / (varcom[-1] * varcom[-1])
    P_EGt_y = np.dot(P_EG.T, y)  # diag(g)EPy
    score = np.sum(np.dot(P_EGt_y.T, P_EGt_y))
    EGtPEG = np.dot(EG.T, P_EG)
    a = np.linalg.eigvalsh(EGtPEG)

    try:
        p_val = davies_pvalue(score, EGtPEG)
    
    except:
        # print("Davies method failed, switch to Liu's method")
        (p_val_liu, dof, delta, _) = liu_sf(score, a, np.ones(q), np.zeros(q))
        p_val = p_val_liu

    return a, score, p_val, cc_par


def cygnet_davies_iter(y, X, G, E, kernel_lst, maxiter=100): # test
    """Run the Davies CYGNET test and return the REML iteration count."""
    y = np.asarray(y, float).reshape(-1, 1) # -1: non constraint on row number
    n = y.shape[0]

    X = np.asarray(X, float).reshape(n, -1) # -1: non constraint on col number
    # X = np.concatenate([np.ones((n, 1)), X], axis=1)
    G = np.asarray(G, float).reshape(n, 1) 
    X = np.concatenate([X, G], axis=1) 
    if _matrix_rank(X) != X.shape[1]:
        _raise_rank_error("X", X)
    q = E.shape[1]

    varcom, cc_par, iter = cygnet_null_multiK_checkiter(y, X, kernel_lst, maxiter=maxiter) # estimate variance component
    # print("Estimated variances: {}".format(varcom))

    # MME
    num_kernel = len(kernel_lst)
    posi_lst = [X.shape[1]]
    for kernel in kernel_lst:
        posi_lst.append(kernel.shape[1] + posi_lst[-1])
    W = np.concatenate([X] + kernel_lst, axis=1)
    CMnull = np.dot(W.T, W)
    CM = CMnull / varcom[-1]
    for k in range(num_kernel):
        start = posi_lst[k]
        end = posi_lst[k + 1]
        CM[start:end, start:end] += np.identity(end - start) / varcom[k]
    CMi = np.linalg.inv(CM)

    EG = E * G  # diag(G)E
    P_EG = EG / varcom[-1] - np.dot(W, np.dot(CMi, np.dot(W.T, EG))) / (varcom[-1] * varcom[-1])
    P_EGt_y = np.dot(P_EG.T, y)  # diag(g)EPy
    score = np.sum(np.dot(P_EGt_y.T, P_EGt_y))
    EGtPEG = np.dot(EG.T, P_EG)
    a = np.linalg.eigvalsh(EGtPEG)

    try:
        p_val = davies_pvalue(score, EGtPEG)
    
    except:
        # print("Davies method failed, switch to Liu's method")
        (p_val_liu, dof, delta, _) = liu_sf(score, a, np.ones(q), np.zeros(q))
        p_val = p_val_liu

    return a, score, p_val, cc_par, iter


def get_fourier(location,gamma_in=1.0,n_component=100, random_state=1):
    """Construct deterministic random Fourier features."""
    n = location.shape[0]
    location = np.asarray(location, float).reshape(n, -1)
    # This reproduces sklearn.kernel_approximation.RBFSampler's NumPy
    # RandomState draws and scaling. NumPy 2 on the affected Windows stack can
    # terminate inside safe_sparse_dot's BLAS dispatch, so only that platform
    # combination uses the safe einsum fallback. Other stacks retain the old
    # matrix-multiplication path and its performance.
    rng = np.random.RandomState(random_state)
    random_weights = np.sqrt(2.0 * gamma_in) * rng.normal(
        size=(location.shape[1], n_component)
    )
    random_offset = rng.uniform(0, 2 * np.pi, size=n_component)
    if os.name == "nt" and int(np.__version__.partition(".")[0]) >= 2:
        S = np.einsum("ij,jk->ik", location, random_weights, optimize=False)
    else:
        S = location @ random_weights
    S += random_offset
    np.cos(S, out=S)
    S *= np.sqrt(2.0 / n_component)
    S = np.asarray(S, float).reshape(n, -1)
    return S

def decompose_low_rank_blocks(blocks, tol=1e-10):
    """
    Takes a list of low-rank, positive semi-definite square matrices [a, b, c...].
    Returns the block diagonal matrix L with reduced columns.
    L @ L.T will perfectly reconstruct the block diagonal matrix of the inputs.
    """
    L_blocks = []
    
    for i, matrix in enumerate(blocks):
        matrix = np.asarray(matrix)
        
        # 1. Eigendecomposition (returns ascending eigenvalues)
        # eigh is specifically optimized for symmetric/Hermitian matrices
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        
        # 2. Filter out numerical noise (eigenvalues that are essentially 0)
        max_eig = np.max(eigenvalues)
        if max_eig <= tol:
            # If the whole block is just zeros, it has rank 0.
            # Create an (n x 0) empty matrix to represent no variance.
            L_block = np.empty((matrix.shape[0], 0))
        else:
            # Set a dynamic threshold relative to the largest eigenvalue
            threshold = tol * max_eig
            valid_mask = eigenvalues > threshold
            
            # 3. Keep only the positive valid eigenvalues & their vectors
            eig_vals_k = eigenvalues[valid_mask]
            eig_vecs_k = eigenvectors[:, valid_mask]
            
            # 4. Form L for this block: L = Eigenvectors * sqrt(Eigenvalues)
            # Broadcasting the multiplication here is much faster than np.diag()
            L_block = eig_vecs_k * np.sqrt(eig_vals_k) 
            
        L_blocks.append(L_block)
        
    # 5. Stitch L_a, L_b, L_c together diagonally
    L_final = block_diag(*L_blocks)
    
    return L_final

def reduce_and_stitch_L_blocks(L_blocks, tol=1e-10):
    """
    Takes a list of [L_a, L_b, L_c...] matrices that might have redundant columns.
    Reduces them to their absolute minimum rank form, and returns 
    the stitched block diagonal matrix L_final.
    """
    reduced_L_blocks = []
    
    for i, L_matrix in enumerate(L_blocks):
        L_matrix = np.asarray(L_matrix)
        
        # 1. Perform SVD directly on the L matrix (Highly numerically stable)
        # S is returned as a 1D array of singular values in descending order
        U, S, Vh = np.linalg.svd(L_matrix, full_matrices=False)
        
        # 2. Filter out numerical noise (singular values that are essentially 0)
        max_s = S[0] if len(S) > 0 else 0
        if max_s <= tol:
            # If the block has no variance, make an empty matrix wrapper
            L_reduced = np.empty((L_matrix.shape[0], 0))
        else:
            # Dynamic threshold relative to the highest singular value
            threshold = tol * max_s
            valid_mask = S > threshold
            
            # 3. Keep only the valid columns of U and valid singular values S
            U_k = U[:, valid_mask]
            S_k = S[valid_mask]
            
            # 4. Form the optimal minimal-column L block
            # Broadcasting U_k * S_k scales the columns natively and quickly
            L_reduced = U_k * S_k
            
        reduced_L_blocks.append(L_reduced)
        
    # 5. Stitch the minimized blocks together
    L_final = block_diag(*reduced_L_blocks)
    
    return L_final


def run_cygnet(y, X, x, E, null_kernels, full_kernels = None, maxiter=100):
    """Run the default CYGNET test and estimate full-model variance components."""
    # try:
    #     x = x.reshape(-1, 1)
    #     y = y.reshape(-1, 1)
    #     a, score, p_val = cygnet(y, X, x, E, null_kernels)
    # except Exception as e:
    #     a, score, p_val = None, None, e
    # return a, score, p_val

    # x = x.reshape(-1, 1)
    # y = y.reshape(-1, 1)
    # a, score, p_val, cc_par = cygnet_davies(y, X, x, E, null_kernels, maxiter)
    # varcom,_ = cygnet_null_multiK(y, np.concatenate([X, x], axis=1), full_kernels, maxiter)
    # return a, score ,p_val, varcom, cc_par

    try:
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        a, score, p_val, cc_par = cygnet_davies(y, X, x, E, null_kernels, maxiter)
        varcom,_ = cygnet_null_multiK(y, np.concatenate([X, x], axis=1), full_kernels, maxiter)
    # varcom = None
    except Exception as e:
        # print(e)
        a, score, p_val, varcom, cc_par = None, None, e, None, None
    return a, score ,p_val, varcom, cc_par

    # x = x.reshape(-1, 1)
    # a, score, p_val = cygnet(y, X, x, E, null_kernels)
    # varcom = cygnet_null_multiK(y, np.concatenate([X, x], axis=1), full_kernels)
    # varcom = None
    # return p_val, varcom


def run_cygnet_permu(y, X, x, E, null_kernels, full_kernels = None, maxiter=100):
    """Run the default CYGNET test for a permuted response without full variance components."""
    # try:
    #     x = x.reshape(-1, 1)
    #     y = y.reshape(-1, 1)
    #     a, score, p_val = cygnet(y, X, x, E, null_kernels)
    # except Exception as e:
    #     a, score, p_val = None, None, e
    # return a, score, p_val

    try:
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        a, score, p_val, cc_par = cygnet_davies(y, X, x, E, null_kernels, maxiter)
        varcom = None

    except Exception as e:
        # print(e)
        a, score, p_val, varcom, cc_par = None, None, e, None, None
    return a, score ,p_val, varcom, cc_par


def run_cygnet_iter(y, X, x, E, null_kernels, full_kernels = None, maxiter=100):
    """Run CYGNET and return the REML iteration count for diagnostics."""
    # try:
    #     x = x.reshape(-1, 1)
    #     y = y.reshape(-1, 1)
    #     a, score, p_val = cygnet(y, X, x, E, null_kernels)
    # except Exception as e:
    #     a, score, p_val = None, None, e
    # return a, score, p_val

    # try:
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    a, score, p_val, cc_par, iter = cygnet_davies_iter(y, X, x, E, null_kernels, maxiter)
    # varcom,_ = cygnet_null_multiK(y, np.concatenate([X, x], axis=1), full_kernels, maxiter)
    varcom = None
    # except Exception as e:
    #     print(e)
    #     a, score, p_val, varcom = None, None, None, None
    return a, score ,p_val, varcom, cc_par, iter

    # x = x.reshape(-1, 1)
    # a, score, p_val = cygnet(y, X, x, E, null_kernels)
    # varcom = cygnet_null_multiK(y, np.concatenate([X, x], axis=1), full_kernels)
    # varcom = None
    # return p_val, varcom


def construct_null_kernels(E, S, gamma=1.0, n=100):
    """Build random Fourier features for environment and spatial null kernels."""
    E = get_fourier(E, gamma, n)
    S = get_fourier(S, gamma, n)
    return E, S


def normalize_features(Z):
    """
    Scales the feature matrix Z so that the maximum diagonal of 
    its implied covariance matrix (Z @ Z.T) is exactly 1.0.
    This keeps the basis scale stable for the variance components solver.
    """
    Z = np.asarray(Z, dtype=float)
    if len(Z.shape) == 1:
        Z = Z.reshape(-1, 1)
        
    # Calculate the row-wise squared L2 norms (the diagonal of Z @ Z.T)
    row_sq_norms = np.sum(Z**2, axis=1)
    max_diag = np.max(row_sq_norms)
    
    if max_diag > 1e-8:
        return Z / np.sqrt(max_diag)
    return Z

def construct_rbf_kernel(
    X,
    gamma=None,
    n=100,
    random_state=1,
    gamma_method="median",
    gamma_max_samples=1000,
):
    """Build random Fourier features for an RBF kernel.

    Parameters
    ----------
    X : array-like
        Observations by features.
    gamma : float, optional
        Explicit RBF gamma. When supplied, ``gamma_method`` is ignored.
    n : int, default 100
        Number of random Fourier features.
    random_state : int, default 1
        Seed used for optional gamma subsampling and Fourier features.
    gamma_method : {"median", "median_half", "manuscript_range"}, default "median"
        Automatic gamma rule. ``"median"`` is the standard median heuristic
        ``1 / median(||x_i - x_j||^2)``. ``"median_half"`` uses
        ``1 / (2 * median(||x_i - x_j||^2))``.
        ``"manuscript_range"`` selects the coordinate-range setting used by
        most article analyses: ``1 / median(max(X_j) - min(X_j))``. The
        corrected large Xenium analysis uses ``"median_half"``.
    gamma_max_samples : int or None, default 1000
        Maximum observations used to estimate a distance-based gamma. The
        deterministic subsample prevents quadratic memory growth. Use
        ``None`` to use every observation. This option does not affect
        ``"manuscript_range"`` or an explicit ``gamma``.

    Returns
    -------
    tuple
        Random Fourier feature matrix and the resolved gamma.
    """
    X = np.asarray(X, float)
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)
    if gamma is None:
        gamma_method = str(gamma_method).lower()
        valid_methods = {"median", "median_half", "manuscript_range"}
        if gamma_method not in valid_methods:
            raise ValueError(
                "gamma_method must be 'median', 'median_half', or "
                "'manuscript_range'."
            )

        if gamma_method == "manuscript_range":
            column_ranges = np.max(X, axis=0) - np.min(X, axis=0)
            median_scale = float(np.median(column_ranges))
        else:
            if gamma_max_samples is not None:
                gamma_max_samples = int(gamma_max_samples)
                if gamma_max_samples < 2:
                    raise ValueError("gamma_max_samples must be at least 2 or None.")
            if gamma_max_samples is not None and X.shape[0] > gamma_max_samples:
                rng = np.random.default_rng(random_state)
                sample_idx = rng.choice(
                    X.shape[0], size=gamma_max_samples, replace=False
                )
                X_gamma = X[sample_idx]
            else:
                X_gamma = X
            squared_distances = pdist(X_gamma, metric="sqeuclidean")
            median_scale = (
                float(np.median(squared_distances))
                if squared_distances.size
                else 0.0
            )

        if not np.isfinite(median_scale) or median_scale <= 0:
            gamma = 1.0
        elif gamma_method == "median_half":
            gamma = 1.0 / (2.0 * median_scale)
        else:
            gamma = 1.0 / median_scale
    Z = get_fourier(X, gamma, n, random_state=random_state)
    return Z, gamma
    

def construct_inter_kernels(E, x):
    """Construct an interaction kernel factor by multiplying each row of E by x."""
    x = np.asarray(x, float).flatten()
    # diag_x = np.diag(x)
    # inter = diag_x @ E 
    inter = x[:, np.newaxis] * E
    return inter


def check_and_rearrange_dataframes(dfs, rearrange=False, only_common=False):
    """
    Check whether DataFrames have identical rows, potentially rearrange rows or 
    filter to common rows based on parameters.

    Args:
        dfs: List of pandas DataFrames to be checked.
        rearrange: Boolean indicating whether to rearrange DataFrames to have the same row order.
        only_common: Boolean indicating whether to filter DataFrames to only include common row names.

    Returns:
        A list of DataFrames that have been rearranged or filtered based on function arguments.
    """
    
    # Check if all DataFrames have the same row order
    reference_index = dfs[0].index
    if all(df.index.equals(reference_index) for df in dfs):
        print("All DataFrames have the same row order.")
        return dfs

    # Find common row names
    common_index = reference_index
    for df in dfs[1:]:
        common_index = common_index.intersection(df.index)

    # If only_common is True, filter all DataFrames by the common index
    if only_common:
        if common_index.empty:
            raise ValueError("There are no common rows among the DataFrames.")
        print("Filtering DataFrames to have only common row names.")
        filtered_dfs = [df.loc[common_index] for df in dfs]
        return filtered_dfs

    # Check if all DataFrames have the same number of rows if only_common is not True
    row_numbers = [df.shape[0] for df in dfs]
    if not all(row_numbers[0] == rn for rn in row_numbers):
        raise ValueError("DataFrames do not have the same row number.")

    # Rearrange DataFrames if requested
    if rearrange:
        if common_index.empty:
            raise ValueError("There are no common rows to rearrange among the DataFrames.")
        print("DataFrames do not have the same row names. Rearranging DataFrames to have the same row order based on common row names.")
        rearranged_dfs = [df.loc[common_index] for df in dfs]
        print("DataFrames have been rearranged to have the same row order based on common row names.")
        return rearranged_dfs
    else:
        raise ValueError("DataFrames do not have the same row names. Set rearrange=True to rearrange DataFrames to have the same row order based on common row names.")

    return dfs


def cygnet_davies_testing_E(y, X, G, E, kernel_lst, maxiter=100): # test
    """Run a Davies score test for the environment main-effect kernel."""
    y = np.asarray(y, float).reshape(-1, 1) # -1: non constraint on row number
    n = y.shape[0]

    X = np.asarray(X, float).reshape(n, -1) # -1: non constraint on col number
    # X = np.concatenate([np.ones((n, 1)), X], axis=1)
    G = np.asarray(G, float).reshape(n, 1) 
    X = np.concatenate([X, G], axis=1) 
    if _matrix_rank(X) != X.shape[1]:
        _raise_rank_error("X", X)
    q = E.shape[1]

    varcom, cc_par = cygnet_null_multiK(y, X, kernel_lst, maxiter=maxiter) # estimate variance component
    # print("Estimated variances: {}".format(varcom))

    # MME
    num_kernel = len(kernel_lst)
    posi_lst = [X.shape[1]]
    for kernel in kernel_lst:
        posi_lst.append(kernel.shape[1] + posi_lst[-1])
    W = np.concatenate([X] + kernel_lst, axis=1)
    CMnull = np.dot(W.T, W)
    CM = CMnull / varcom[-1]
    for k in range(num_kernel):
        start = posi_lst[k]
        end = posi_lst[k + 1]
        CM[start:end, start:end] += np.identity(end - start) / varcom[k]
    CMi = np.linalg.inv(CM)

    EG = E  # diag(G)E
    P_EG = EG / varcom[-1] - np.dot(W, np.dot(CMi, np.dot(W.T, EG))) / (varcom[-1] * varcom[-1])
    P_EGt_y = np.dot(P_EG.T, y)  # diag(g)EPy
    score = np.sum(np.dot(P_EGt_y.T, P_EGt_y))
    EGtPEG = np.dot(EG.T, P_EG)
    a = np.linalg.eigvalsh(EGtPEG)

    try:
        p_val = davies_pvalue(score, EGtPEG)
    
    except:
        # print("Davies method failed, switch to Liu's method")
        (p_val_liu, dof, delta, _) = liu_sf(score, a, np.ones(q), np.zeros(q))
        p_val = p_val_liu

    return a, score, p_val, cc_par


def cygnet_wald_testing_x(y, X, x, E, kernel_lst, maxiter=1000):
    """
    Performs an Association Test (Persistent Genetic/Main Effect) using a Wald Test 
    estimates from the Null Model (y ~ X + Random(kernel_lst)).

    Matches the signature of cygnet_davies_testing_E.

    Parameters
    ----------
    y : array
        Phenotype (N x 1)
    X : array
        Fixed covariates (N x C)
    x : array
        The Feature to test (N x 1). 
        (Note: x is treated as a Fixed Effect here).
    E : array
        Environment matrix. Kept for signature compatibility, 
        but actual random structure comes from kernel_lst.
    kernel_lst : list
        List of feature matrices for Random Effects (e.g. [E, S]).
    maxiter : int
        Maximum iterations for AI-REML.

    Returns
    -------
    a : array
        Weights for Chi2 distribution (Always [1.0] for Wald test df=1).
    score : float
        The Wald Statistic (beta^2 / var(beta)).
    p_val : float
        P-value of the association.
    cc_par : float
        Convergence parameter from the Null Model fit.
    """
    
    # 1. Reshape inputs
    y = np.asarray(y, float).reshape(-1, 1)
    n = y.shape[0]
    X = np.asarray(X, float).reshape(n, -1)
    x = np.asarray(x, float).reshape(n, -1)

    # 2. Check Rank of X just in case
    # (Optional, but good practice given previous code has rank checks)
    if _matrix_rank(X) != X.shape[1]:
        # Simple fallback or warning
        pass

    # 3. Estimate Variance Components for the Null Model
    # y ~ X + Random(kernel_lst)
    varcom, cc_par = cygnet_null_multiK(y, X, kernel_lst, maxiter=maxiter)
    
    # 4. Construct the Dense Covariance Matrix V
    # V = sum(sigma_k^2 * Z_k @ Z_k.T) + sigma_e^2 * I
    sigma_epsilon = varcom[-1]
    V = np.eye(n) * sigma_epsilon
    
    for k, factor_matrix in enumerate(kernel_lst):
        sigma_k = varcom[k]
        V += sigma_k * (factor_matrix @ factor_matrix.T)

    # 5. Invert V (GLS Preparation)
    try:
        Vi = np.linalg.inv(V)
    except np.linalg.LinAlgError:
        Vi = np.linalg.pinv(V)

    # 6. Perform Generalized Least Squares (GLS) for [X, G]
    # We want to solve: (D.T @ Vi @ D) * beta = D.T @ Vi @ y
    # Where D = [X, G]
    
    try:
        # -- Pre-compute X parts to save time (though for 1 SNP it's negligible) --
        XtVi = X.T @ Vi
        XtViX = XtVi @ X
        XtViy = XtVi @ y

        # -- Calculations for G --
        gtVi = x.T @ Vi
        gtVig = gtVi @ x
        gtViX = gtVi @ X  # Shape (1, Cols_X)

        # -- Build the Information Matrix (LHS) --
        # | XtViX   XtVig |
        # | gtViX   gtVig |
        LHS = np.block([
            [XtViX,   gtViX.T],
            [gtViX,   gtVig]
        ])

        # -- Build the RHS --
        gtViy = gtVi @ y
        RHS = np.concatenate([XtViy, gtViy], axis=0)

        # -- Solve for Beta --
        # We need the inverse of LHS to get the variance of the estimates
        InvLHS = np.linalg.inv(LHS)
        beta_all = np.dot(InvLHS, RHS)

        # -- Extract Statistics --
        # The parameter of interest (G) is the last one
        beta_g = beta_all[-1, 0]
        
        # Variance of the estimate is the bottom-right element of Inv(InformationMatrix)
        var_beta_g = InvLHS[-1, -1]

        # Wald Statistic: (beta / SE)^2
        score = (beta_g**2) / var_beta_g
        
        # P-value (Chi-squared with dof=1)
        p_val = chi2.sf(score, df=1)
        
        # 'a' is just [1.0] because Wald test stat ~ 1 * Chi2(1)
        a = np.array([1.0])

    except Exception:
        # Fallback for Singular Matrix or Solver errors
        score = 0.0
        p_val = 1.0
        a = np.array([1.0])

    return a, score, p_val, cc_par


def run_cygnet_testing_E(y, X, x, E, null_kernels, full_kernels = None, maxiter=100):
    """Run the environment main-effect CYGNET test and estimate full variance components."""
    # try:
    #     x = x.reshape(-1, 1)
    #     y = y.reshape(-1, 1)
    #     a, score, p_val = cygnet(y, X, x, E, null_kernels)
    # except Exception as e:
    #     a, score, p_val = None, None, e
    # return a, score, p_val

    try:
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        a, score, p_val, cc_par = cygnet_davies_testing_E(y, X, x, E, null_kernels, maxiter)
        varcom,_ = cygnet_null_multiK(y, np.concatenate([X, x], axis=1), full_kernels, maxiter)

    # varcom = None
    except Exception as e:
        # print(e)
        a, score, p_val, varcom, cc_par = None, None, e, None, None
    return a, score ,p_val, varcom, cc_par

    # x = x.reshape(-1, 1)
    # a, score, p_val = cygnet(y, X, x, E, null_kernels)
    # varcom = cygnet_null_multiK(y, np.concatenate([X, x], axis=1), full_kernels)
    # varcom = None
    # return p_val, varcom


def run_cygnet_testing_E_permu(y, X, x, E, null_kernels, full_kernels = None, maxiter=100):
    """Run the environment main-effect CYGNET test for permutation workflows."""
    # try:
    #     x = x.reshape(-1, 1)
    #     y = y.reshape(-1, 1)
    #     a, score, p_val = cygnet(y, X, x, E, null_kernels)
    # except Exception as e:
    #     a, score, p_val = None, None, e
    # return a, score, p_val

    try:
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        a, score, p_val, cc_par = cygnet_davies_testing_E(y, X, x, E, null_kernels, maxiter)
        # varcom,_ = cygnet_null_multiK(y, np.concatenate([X, x], axis=1), full_kernels, maxiter)
        varcom = None

    except Exception as e:
        # print(e)
        a, score, p_val, varcom, cc_par = None, None, e, None, None
    return a, score ,p_val, varcom, cc_par

    # x = x.reshape(-1, 1)
    # a, score, p_val = cygnet(y, X, x, E, null_kernels)
    # varcom = cygnet_null_multiK(y, np.concatenate([X, x], axis=1), full_kernels)
    # varcom = None
    # return p_val, varcom


def cygnet_davies_testing_E_celltype_specific(y, X, G, E, kernel_lst, maxiter=100):
    """Run a cell-type-specific environment score test without adding G to X."""
    y = np.asarray(y, float).reshape(-1, 1) # -1: non constraint on row number
    n = y.shape[0]

    X = np.asarray(X, float).reshape(n, -1) # -1: non constraint on col number
    # X = np.concatenate([np.ones((n, 1)), X], axis=1)
    # G = np.asarray(G, float).reshape(n, 1) 
    # X = np.concatenate([X, G], axis=1) 
    if _matrix_rank(X) != X.shape[1]:
        _raise_rank_error("X", X)
    q = E.shape[1]

    varcom, cc_par = cygnet_null_multiK(y, X, kernel_lst, maxiter=maxiter) # estimate variance component
    # print("Estimated variances: {}".format(varcom))

    # MME
    num_kernel = len(kernel_lst)
    posi_lst = [X.shape[1]]
    for kernel in kernel_lst:
        posi_lst.append(kernel.shape[1] + posi_lst[-1])
    W = np.concatenate([X] + kernel_lst, axis=1)
    CMnull = np.dot(W.T, W)
    CM = CMnull / varcom[-1]
    for k in range(num_kernel):
        start = posi_lst[k]
        end = posi_lst[k + 1]
        CM[start:end, start:end] += np.identity(end - start) / varcom[k]
    CMi = np.linalg.inv(CM)

    EG = E  # diag(G)E
    P_EG = EG / varcom[-1] - np.dot(W, np.dot(CMi, np.dot(W.T, EG))) / (varcom[-1] * varcom[-1])
    P_EGt_y = np.dot(P_EG.T, y)  # diag(g)EPy
    score = np.sum(np.dot(P_EGt_y.T, P_EGt_y))
    EGtPEG = np.dot(EG.T, P_EG)
    a = np.linalg.eigvalsh(EGtPEG)

    try:
        p_val = davies_pvalue(score, EGtPEG)
    
    except:
        # print("Davies method failed, switch to Liu's method")
        (p_val_liu, dof, delta, _) = liu_sf(score, a, np.ones(q), np.zeros(q))
        p_val = p_val_liu

    return a, score, p_val, cc_par


def run_cygnet_testing_E_celltype_specific(y, X, x, E, null_kernels, full_kernels = None, maxiter=100):
    """Run the cell-type-specific environment test and estimate full variance components."""
    # try:
    #     x = x.reshape(-1, 1)
    #     y = y.reshape(-1, 1)
    #     a, score, p_val = cygnet(y, X, x, E, null_kernels)
    # except Exception as e:
    #     a, score, p_val = None, None, e
    # return a, score, p_val

    try:
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        a, score, p_val, cc_par = cygnet_davies_testing_E_celltype_specific(y, X, x, E, null_kernels, maxiter)
        varcom,_ = cygnet_null_multiK(y, np.concatenate([X, x], axis=1), full_kernels, maxiter)
    # varcom = None

    except Exception as e:
        # print(e)
        a, score, p_val, varcom, cc_par = None, None, e, None, None
    return a, score ,p_val, varcom, cc_par


def run_cygnet_testing_E_celltype_specific_permu(y, X, x, E, null_kernels, full_kernels = None, maxiter=100):
    """Run the cell-type-specific environment test for permutation workflows."""
    # try:
    #     x = x.reshape(-1, 1)
    #     y = y.reshape(-1, 1)
    #     a, score, p_val = cygnet(y, X, x, E, null_kernels)
    # except Exception as e:
    #     a, score, p_val = None, None, e
    # return a, score, p_val

    try:
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        a, score, p_val, cc_par = cygnet_davies_testing_E_celltype_specific(y, X, x, E, null_kernels, maxiter)
        # varcom,_ = cygnet_null_multiK(y, np.concatenate([X, x], axis=1), full_kernels, maxiter)
        varcom = None

    except Exception as e:
        # print(e)
        a, score, p_val, varcom, cc_par = None, None, e, None, None
    return a, score ,p_val, varcom, cc_par


def _as_environment_frame(values):
    """Return an environment DataFrame without dropping input columns."""
    if isinstance(values, pd.Series):
        return values.to_frame()

    if isinstance(values, pd.DataFrame):
        if values.shape[1] < 1:
            raise ValueError("Environment input must have at least one column.")
        return values.copy()

    array = np.asarray(values)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2 or array.shape[1] < 1:
        raise ValueError("Environment input must be a vector or a matrix with at least one column.")
    return pd.DataFrame(array)


def _environment_column_names(columns):
    """Return stable string column names for transformed environment matrices."""
    names = []
    for i, column in enumerate(columns):
        name = str(column)
        names.append(name if name else f"env{i + 1}")
    return names


def _resolve_rscript(rscript=None):
    """Resolve the Rscript executable used for mgcv transforms."""
    if rscript:
        return rscript

    env_rscript = os.environ.get("CYGNET_RSCRIPT")
    if env_rscript:
        return env_rscript

    found = shutil.which("Rscript")
    if found:
        return found

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        conda_candidates = [
            os.path.join(conda_prefix, "Scripts", "Rscript.exe"),
            os.path.join(conda_prefix, "bin", "Rscript"),
        ]
        for candidate in conda_candidates:
            if os.path.isfile(candidate):
                return candidate

    posix_candidates = [
        "/usr/local/bin/Rscript",
        "/usr/bin/Rscript",
        "/opt/homebrew/bin/Rscript",
        "/opt/local/bin/Rscript",
    ]
    for candidate in posix_candidates:
        if os.path.isfile(candidate):
            return candidate

    windows_roots = [
        os.path.join(os.environ.get("ProgramFiles", ""), "R"),
        os.path.join(os.environ.get("ProgramFiles(x86)", ""), "R"),
    ]
    for root in windows_roots:
        if not root or not os.path.isdir(root):
            continue
        candidates = glob(os.path.join(root, "R-*", "bin", "x64", "Rscript.exe"))
        candidates += glob(os.path.join(root, "R-*", "bin", "Rscript.exe"))
        if candidates:
            return sorted(candidates)[-1]

    raise RuntimeError(
        "Rscript was not found. Install R with the mgcv package, or set CYGNET_RSCRIPT "
        "to the full path of Rscript."
    )


def mgcv_tp_transform(values, k=10, rscript=None):
    """Build the mgcv thin-plate spline basis for one or more environment columns.

    This calls R's ``mgcv::smoothCon(mgcv::s(..., k=k, fx=TRUE, bs='tp'))``
    and returns the resulting design matrix as a pandas DataFrame. Multiple
    input columns are passed to one multivariate thin-plate smooth.
    """
    env_df = _as_environment_frame(values)
    input_names = _environment_column_names(env_df.columns)
    r_names = [f"env_{i + 1}" for i in range(env_df.shape[1])]
    env_for_r = env_df.copy()
    env_for_r.columns = r_names

    r_code = r"""
args <- commandArgs(trailingOnly = TRUE)
input_path <- args[1]
output_path <- args[2]
k <- as.integer(args[3])
df <- read.csv(input_path, check.names = FALSE)
if (ncol(df) < 1) {
  stop("mgcv_tp_transform expects at least one environment column")
}
suppressPackageStartupMessages(library(mgcv))
smooth_expr <- paste0(
  "mgcv::s(",
  paste(colnames(df), collapse = ", "),
  ", k = ", k,
  ", fx = TRUE, bs = \"tp\")"
)
smooth_con <- mgcv::smoothCon(eval(parse(text = smooth_expr)), data = df)[[1]]
write.csv(as.data.frame(smooth_con$X), output_path, row.names = FALSE, quote = FALSE)
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "env.csv")
        output_path = os.path.join(tmpdir, "mgcv_env.csv")
        env_for_r.to_csv(input_path, index=False)

        completed = subprocess.run(
            [_resolve_rscript(rscript), "-e", r_code, input_path, output_path, str(k)],
            text=True,
            capture_output=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "mgcv thin-plate transform failed. "
                f"stderr:\n{completed.stderr.strip()}"
            )

        transformed = pd.read_csv(output_path)

    transformed.index = env_df.index
    if len(input_names) == 1:
        prefix = input_names[0]
    else:
        prefix = "_".join(input_names)
    transformed.columns = [f"{prefix}_mgcv_{i + 1}" for i in range(transformed.shape[1])]
    return transformed


def statsmodels_bspline_transform(values, k=10, degree=3):
    """Build statsmodels cubic B-spline bases for one or more environment columns."""
    env_df = _as_environment_frame(values)
    input_names = _environment_column_names(env_df.columns)
    x = env_df.to_numpy(dtype=float)

    from statsmodels.gam.smooth_basis import BSplines

    basis = BSplines(
        x,
        df=[k] * x.shape[1],
        degree=[degree] * x.shape[1],
        include_intercept=True,
    ).basis
    columns = []
    for input_name in input_names:
        columns.extend([f"{input_name}_bspline_{i + 1}" for i in range(k)])
    return pd.DataFrame(
        basis,
        index=env_df.index,
        columns=columns,
    )


def transform_environment(values, method="mgcv", k=10, rscript=None):
    """Transform environment columns with mgcv tp splines or statsmodels B-splines."""
    method_normalized = method.lower()
    if method_normalized in _NONLINEAR_ENV_TYPES or method_normalized == "r":
        return mgcv_tp_transform(values, k=k, rscript=rscript)
    if method_normalized in _STATSMODELS_ENV_TYPES:
        return statsmodels_bspline_transform(values, k=k)
    if method_normalized == "values":
        return _as_environment_frame(values)
    raise ValueError(
        "Unknown environment transform method. Use 'values', 'mgcv', 'non-linear', "
        "or 'statsmodels'."
    )


def load_simulation_data(
    simulation_dir,
    seed,
    normalized_type='sctransform',
    env_type='values',
    env_transform_backend=None,
    env_spline_k=10,
    rscript=None,
    mode='cell',
    celltype_process=None,
    celltype_clr_drop_column='auto',
):
    """
    Load simulation data from a specific simulation directory and seed.
    
    Parameters
    ----------
    simulation_dir : str
        Path to the simulation directory (e.g., 'simulation_results/simulation_1')
    seed : int
        Seed number to load
    
    Returns
    -------
    tuple
        (locations_df, celltype_df, normalized_counts_df, env_values_df)
        - locations_df: DataFrame with x, y coordinates
        - celltype_df: DataFrame with cell IDs and cell types
        - normalized_counts_df: DataFrame with normalized gene expression counts
        - env_values_df: DataFrame with environmental values
    """
    mode_normalized = mode.lower()
    if mode_normalized not in {"cell", "spot"}:
        raise ValueError("mode must be 'cell' or 'spot'.")

    # Select the prepared file matching ``env_type`` unless an in-memory
    # transformation backend is requested.
    transform_requested = env_transform_backend is not None
    env_file_type = 'values' if transform_requested else env_type
    transform_method = env_transform_backend

    # Initialize DataFrames
    locations_df = None
    celltype_df = None
    normalized_counts_df = None
    env_values_df = None
    
    # Get simulation index from directory name
    sim_dir_name = os.path.basename(os.path.normpath(simulation_dir))
    sim_index = sim_dir_name.split('_')[1]
    
    # Create file pattern for the specific seed
    pattern = f'sim_{sim_index}_{seed}_*.csv*'
    
    # Get all CSV files for this seed
    files = glob(os.path.join(simulation_dir, pattern))
    
    if not files:
        raise ValueError(f"No files found for simulation {sim_index}, seed {seed}")
    
    # Load each type of file
    for file in files:
        file_name = os.path.basename(file)
        if 'locations' in file_name:
            locations_df = pd.read_csv(file, index_col=0)
        elif 'celltypes' in file_name:
            celltype_df = pd.read_csv(file, index_col=0)
        elif 'normalized_counts' in file_name and normalized_type in file_name:
            normalized_counts_df = pd.read_csv(file, index_col=0)
        elif 'env' in file_name and env_file_type in file_name:
            env_values_df = pd.read_csv(file, index_col=0)

    if env_values_df is not None and not transform_requested and env_type == 'values':
        # A raw scalar environment uses its first input column.
        env_values_df = env_values_df.iloc[:, [0]]

    if env_values_df is not None and transform_requested:
        env_values_df = transform_environment(
            env_values_df,
            method=transform_method,
            k=env_spline_k,
            rscript=rscript,
        )

    # Check if all required data was loaded before type-specific preprocessing.
    if any(df is None for df in [locations_df, celltype_df, normalized_counts_df, env_values_df]):
        missing = []
        if locations_df is None: missing.append("locations")
        if celltype_df is None: missing.append("celltypes")
        if normalized_counts_df is None: missing.append("normalized_counts")
        if env_values_df is None: missing.append("env_values")
        raise ValueError(f"Missing required data files: {', '.join(missing)}")

    # change celltype_df to one-hot encoding
    celltype_df = pd.get_dummies(celltype_df, prefix='', prefix_sep='',dtype=float)
    process = (celltype_process or ("clr" if mode_normalized == "spot" else "onehot")).lower()
    if process in {"clr", "clr_transform", "clr-transformed"}:
        celltype_df = celltype_clr_transform_from_df(
            celltype_df,
            drop_column=celltype_clr_drop_column,
        )
    elif process not in {"onehot", "one-hot", "raw", "none"}:
        raise ValueError("celltype_process must be 'onehot', 'raw', 'none', or 'clr'.")

    return locations_df, celltype_df, normalized_counts_df, env_values_df


# Define processing functions at the top level for proper pickling
def process_real_chunk(genes_chunk, celltype_name, X_dropped, x, inter, E, S, normalized_counts_df):
    """Apply the default CYGNET test to a chunk of observed genes."""
    results = []
    for columnname in genes_chunk:
        y = normalized_counts_df[columnname].to_numpy()
        y = y - np.mean(y)
        a, score, p_val, v, delta = run_cygnet(y, X_dropped, x, E, [E, S], [inter, E, S], maxiter=100)
        results.append({
            'Gene': columnname,
            'Celltype': celltype_name,
            'p_value': p_val,
            'score': score,
            'varcom': v,
            'seed': 0,  
            'cc_par': delta
        })
    return results

def process_permutation_chunk(genes_chunk, celltype_name, X_dropped, x, inter, E, S, seed, normalized_counts_df):
    """Apply the default CYGNET test to a chunk of permuted genes."""
    results = []
    for columnname in genes_chunk:
        y = normalized_counts_df[columnname].to_numpy()
        y = y - np.mean(y)
        np.random.seed(seed)
        y_perm = np.random.permutation(y)
        a, score, p_val, v, delta = run_cygnet_permu(y_perm, X_dropped, x, E, [E, S], [inter, E, S], maxiter=100)
        results.append({
            'Gene': columnname,
            'Celltype': celltype_name,
            'p_value': p_val,
            'score': score,
            'varcom': v,
            'seed': seed,
            'cc_par': delta
        })
    return results


def process_permutation_chunk_on_x(genes_chunk, celltype_name, X_dropped, x, inter, E, S, seed, normalized_counts_df):
    """Apply CYGNET to observed genes while recording a permutation seed for x."""
    results = []
    for columnname in genes_chunk:
        y = normalized_counts_df[columnname].to_numpy()
        y = y - np.mean(y)
        a, score, p_val, v, delta = run_cygnet_permu(y, X_dropped, x, E, [E, S], [inter, E, S], maxiter=100)
        results.append({
            'Gene': columnname,
            'Celltype': celltype_name,
            'p_value': p_val,
            'score': score,
            'varcom': v,
            'seed': seed,
            'cc_par': delta
        })
    return results

def _resolve_clr_drop_column(input_df, drop_column):
    """Resolve which CLR-transformed cell-type column should be removed."""
    if drop_column is None or drop_column is False:
        return None
    if input_df.shape[1] <= 1:
        raise ValueError("Cannot drop a CLR column when the cell-type matrix has only one column.")
    if drop_column is True or str(drop_column).lower() == "auto":
        column_sums = input_df.sum(axis=0)
        return column_sums.idxmin()
    if drop_column not in input_df.columns:
        raise ValueError(f"CLR drop column not found in cell-type matrix: {drop_column}")
    return drop_column


def celltype_clr_transform_from_df(input_df, output_file=None, drop_column=None):
    """
    Transform cell-type composition data to CLR format.

    Parameters
    ----------
    input_df : pandas.DataFrame
        Non-negative cell-type composition matrix.
    output_file : str or path-like, optional
        Optional CSV path for the transformed matrix.
    drop_column : str, "auto", bool, or None, optional
        Column to remove after CLR transformation. ``"auto"`` or True removes
        the cell type with the smallest original column sum. None keeps all
        CLR columns.
    """
    drop_resolved = _resolve_clr_drop_column(input_df, drop_column)

    # These helper functions are from scikit-bio packages
    def closure(mat):
        mat = np.atleast_2d(mat)
        if np.any(mat < 0):
            raise ValueError("Cannot have negative proportions")
        if mat.ndim > 2:
            raise ValueError("Input matrix can only have two dimensions or less")
        if np.all(mat == 0, axis=1).sum() > 0:
            raise ValueError("Input matrix cannot have rows with all zeros")
        mat = mat / mat.sum(axis=1, keepdims=True)
        return mat.squeeze()

    def clr(mat):
        mat = closure(mat)
        lmat = np.log(mat)
        gm = lmat.mean(axis=-1, keepdims=True)
        return (lmat - gm).squeeze()
    
    def multi_replace(mat, delta=None):
        mat = closure(mat)
        z_mat = mat == 0

        num_feats = mat.shape[-1]
        tot = z_mat.sum(axis=-1, keepdims=True)

        if delta is None:
            delta = (1.0 / num_feats) ** 2

        zcnts = 1 - tot * delta
        if np.any(zcnts) < 0:
            raise ValueError(
                "The multiplicative replacement created negative "
                "proportions. Consider using a smaller `delta`."
            )
        mat = np.where(z_mat, delta, zcnts * mat)
        return mat.squeeze()

    df = input_df
    data = df.to_numpy()

    if not np.allclose(data.sum(axis=1), 1, atol=1e-6):
        #raise ValueError("Rows do not sum to 1. Check input data!")
        warnings.warn("Rows do not sum to 1. Applying closure transformation.")
    # check if all values are not 0 and positive
    if not np.all(data >= 0):
        raise ValueError("All values must be non-negative for CLR transformation.")
    elif not np.all(data > 0):
        warnings.warn("Some values are 0. Applying multi_replace transformation.")
        data = multi_replace(data)

    # CLR
    data = clr(data)
    clr_df = pd.DataFrame(data, index=df.index, columns=df.columns)
    if drop_resolved is not None:
        clr_df = clr_df.drop(columns=[drop_resolved])
    if output_file:
        clr_df.to_csv(output_file)
    return clr_df

def process_real_chunk_with_spatial_random_effect(genes_chunk, celltype_name, X_dropped, x, e_inter, E, S, normalized_counts_df, maxiter=100):
    """Apply CYGNET with a spatial interaction random effect to observed genes."""
    results = []
    s_inter = construct_inter_kernels(S, x)
    for columnname in genes_chunk:
        y = normalized_counts_df[columnname].to_numpy()
        y = y - np.mean(y)
        a, score, p_val, v, delta = run_cygnet(y, X_dropped, x, E, [s_inter, E, S], [e_inter, s_inter, E, S], maxiter=maxiter)
        results.append({
            'Gene': columnname,
            'Celltype': celltype_name,
            'p_value': p_val,
            'score': score,
            'varcom': v,
            'seed': 0,  
            'cc_par': delta
        })
    return results

def process_permutation_chunk_with_spatial_random_effect(genes_chunk, celltype_name, X_dropped, x, e_inter, E, S, seed, normalized_counts_df, maxiter=100):
    """Apply CYGNET with a spatial interaction random effect to permuted genes."""
    results = []
    s_inter = construct_inter_kernels(S, x)
    for columnname in genes_chunk:
        y = normalized_counts_df[columnname].to_numpy()
        y = y - np.mean(y)
        np.random.seed(seed)
        y_perm = np.random.permutation(y)
        a, score, p_val, v, delta = run_cygnet_permu(y_perm, X_dropped, x, E, [s_inter, E, S], [e_inter, s_inter, E, S], maxiter=maxiter)
        results.append({
            'Gene': columnname,
            'Celltype': celltype_name,
            'p_value': p_val,
            'score': score,
            'varcom': v,
            'seed': seed,
            'cc_par': delta
        })
    return results

def process_real_chunk_with_all_random_effect(genes_chunk, celltype_name, X_dropped, x, e_inter, E, S, normalized_counts_df, maxiter=100):
    """Apply CYGNET with spatial and residual interaction random effects."""
    results = []
    s_inter = construct_inter_kernels(S, x)
    identity_matrix = np.identity(x.shape[0])
    residual_inter = construct_inter_kernels(identity_matrix, x)
    for columnname in genes_chunk:
        y = normalized_counts_df[columnname].to_numpy()
        y = y - np.mean(y)
        a, score, p_val, v, delta = run_cygnet(y, X_dropped, x, E, [s_inter, residual_inter, E, S], [e_inter, s_inter, residual_inter, E, S], maxiter=maxiter)
        results.append({
            'Gene': columnname,
            'Celltype': celltype_name,
            'p_value': p_val,
            'score': score,
            'varcom': v,
            'seed': 0,  
            'cc_par': delta
        })
    return results

def process_permutation_chunk_with_all_random_effect(genes_chunk, celltype_name, X_dropped, x, e_inter, E, S, seed, normalized_counts_df, maxiter=100):
    """Apply the all-random-effects CYGNET model to permuted genes."""
    results = []
    s_inter = construct_inter_kernels(S, x)
    identity_matrix = np.identity(x.shape[0])
    residual_inter = construct_inter_kernels(identity_matrix, x)
    for columnname in genes_chunk:
        y = normalized_counts_df[columnname].to_numpy()
        y = y - np.mean(y)
        np.random.seed(seed)
        y_perm = np.random.permutation(y)
        a, score, p_val, v, delta = run_cygnet_permu(y_perm, X_dropped, x, E, [s_inter, residual_inter, E, S], [e_inter, s_inter, residual_inter, E, S], maxiter=maxiter)
        results.append({
            'Gene': columnname,
            'Celltype': celltype_name,
            'p_value': p_val,
            'score': score,
            'varcom': v,
            'seed': seed,
            'cc_par': delta
        })
    return results


def process_real_chunk_testing_E(genes_chunk, celltype_name, X_dropped, x, e_inter, E, S, normalized_counts_df, maxiter=100):
    """Apply the environment main-effect test to observed genes."""
    results = []
    for columnname in genes_chunk:
        y = normalized_counts_df[columnname].to_numpy()
        y = y - np.mean(y)
        a, score, p_val, v, delta = run_cygnet_testing_E(y, X_dropped, x, E, [S], [E, S], maxiter=100)
        results.append({
            'Gene': columnname,
            'Celltype': celltype_name,
            'p_value': p_val,
            'score': score,
            'varcom': v,
            'seed': 0,  
            'cc_par': delta
        })
    return results

def process_permutation_chunk_testing_E(genes_chunk, celltype_name, X_dropped, x, e_inter, E, S, seed, normalized_counts_df, maxiter=100):
    """Apply the environment main-effect test to permuted genes."""
    results = []
    for columnname in genes_chunk:
        y = normalized_counts_df[columnname].to_numpy()
        y = y - np.mean(y)
        np.random.seed(seed)
        y_perm = np.random.permutation(y)
        a, score, p_val, v, delta = run_cygnet_testing_E(y_perm, X_dropped, x, E, [S], [E, S], maxiter=maxiter)
        results.append({
            'Gene': columnname,
            'Celltype': celltype_name,
            'p_value': p_val,
            'score': score,
            'varcom': v,
            'seed': seed,
            'cc_par': delta
        })
    return results


def process_real_chunk_testing_E_celltype_specific(genes_chunk, celltype_name, X_dropped, x, e_inter, E, S, normalized_counts_df, maxiter=100):
    """Apply the cell-type-specific environment test to observed genes."""
    results = []
    # inter = construct_inter_kernels(E, x)
    for columnname in genes_chunk:
        y = normalized_counts_df[columnname].to_numpy()
        y = y - np.mean(y)
        a, score, p_val, v, delta = run_cygnet_testing_E_celltype_specific(y, X_dropped, x, E, [S], [E, S], maxiter=100)
        results.append({
            'Gene': columnname,
            'Celltype': celltype_name,
            'p_value': p_val,
            'score': score,
            'varcom': v,
            'seed': 0,  
            'cc_par': delta
        })
    return results

def process_permutation_chunk_testing_E_celltype_specific(genes_chunk, celltype_name, X_dropped, x, e_inter, E, S, seed, normalized_counts_df, maxiter=100):
    """Apply the cell-type-specific environment test to permuted genes."""
    results = []
    # inter = construct_inter_kernels(E, x)
    for columnname in genes_chunk:
        y = normalized_counts_df[columnname].to_numpy()
        y = y - np.mean(y)
        np.random.seed(seed)
        y_perm = np.random.permutation(y)
        a, score, p_val, v, delta = run_cygnet_testing_E_celltype_specific(y_perm, X_dropped, x, E, [S], [E, S], maxiter=100)
        results.append({
            'Gene': columnname,
            'Celltype': celltype_name,
            'p_value': p_val,
            'score': score,
            'varcom': v,
            'seed': seed,  
            'cc_par': delta
        })
    return results

def process_real_chunk_testing_margin_E_with_inter(genes_chunk, celltype_name, X_dropped, x, e_inter, E, S, normalized_counts_df, maxiter=100):
    """Apply a marginal environment test with an interaction kernel to observed genes."""
    results = []
    for columnname in genes_chunk:
        y = normalized_counts_df[columnname].to_numpy()
        y = y - np.mean(y)
        a, score, p_val, v, delta = run_cygnet(y, X_dropped, x, E, [e_inter, S], [E, e_inter, S], maxiter=100)
        results.append({
            'Gene': columnname,
            'Celltype': celltype_name,
            'p_value': p_val,
            'score': score,
            'varcom': v,
            'seed': 0,  
            'cc_par': delta
        })
    return results

def process_permutation_chunk_testing_margin_E_with_inter(genes_chunk, celltype_name, X_dropped, x, e_inter, E, S, seed, normalized_counts_df, maxiter=100):
    """Apply a marginal environment test with an interaction kernel to permuted genes."""
    results = []
    # inter = construct_inter_kernels(E, x)
    for columnname in genes_chunk:
        y = normalized_counts_df[columnname].to_numpy()
        y = y - np.mean(y)
        np.random.seed(seed)
        y_perm = np.random.permutation(y)
        a, score, p_val, v, delta = run_cygnet(y_perm, X_dropped, x, E, [e_inter, S], [E, e_inter, S], maxiter=100)
        results.append({
            'Gene': columnname,
            'Celltype': celltype_name,
            'p_value': p_val,
            'score': score,
            'varcom': v,
            'seed': seed,  
            'cc_par': delta
        })
    return results


if __name__ == "__main__":
    pass
