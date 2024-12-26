import numpy as np
import scipy as sc
from scipy.signal import fftconvolve


def calculate_log_probability(X, F, B, s):
    """
    Calculates log p(X_k|d_k,F,B,s) for all images X_k in X and
    all possible displacements d_k.

    Parameters
    ----------
    X : array, shape (H, W, K)
        K images of size H x W.
    F : array, shape (h, w)
        Estimate of villain's face.
    B : array, shape (H, W)
        Estimate of background.
    s : float
        Estimate of standard deviation of Gaussian noise.

    Returns
    -------
    ll : array, shape(H-h+1, W-w+1, K)
        ll[dh,dw,k] - log-likelihood of observing image X_k given
        that the villain's face F is located at displacement (dh, dw)
    """
    H, W, K = X.shape
    h, w = F.shape
    ll = np.sum((X - B[..., None]) ** 2, axis=(0, 1))[None, None, :] - \
         fftconvolve(B ** 2, np.ones((h, w)), mode="valid")[..., None] + 2 * fftconvolve(X * B[..., None],
                                                                                         np.ones((h, w, 1)),
                                                                                         mode="valid") + np.sum(
        F ** 2) - 2 * fftconvolve(X, F[::-1, ::-1, None], mode="valid")
    return ll * (-1 / (2 * s ** 2)) - H * W * np.log(2 * np.pi * s ** 2 + 1e-12) / 2


def calculate_lower_bound(X, F, B, s, A, q, use_MAP=False):
    """
    Calculates the lower bound L(q,F,B,s,A) for the marginal log likelihood.

    Parameters
    ----------
    X : array, shape (H, W, K)
        K images of size H x W.
    F : array, shape (h, w)
        Estimate of villain's face.
    B : array, shape (H, W)
        Estimate of background.
    s : float
        Estimate of standard deviation of Gaussian noise.
    A : array, shape (H-h+1, W-w+1)
        Estimate of prior on displacement of face in any image.
    q : array
        If use_MAP = False: shape (H-h+1, W-w+1, K)
            q[dh,dw,k] - estimate of posterior of displacement (dh,dw)
            of villain's face given image Xk
        If use_MAP = True: shape (2, K)
            q[0,k] - MAP estimates of dh for X_k
            q[1,k] - MAP estimates of dw for X_k
    use_MAP : bool, optional
        If true then q is a MAP estimates of displacement (dh,dw) of
        villain's face given image Xk.

    Returns
    -------
    L : float
        The lower bound L(q,F,B,s,A) for the marginal log likelihood.
    """
    log_full_likelihood = calculate_log_probability(X, F, B, s) + np.log(A + 1e-12)[..., None]

    if use_MAP:
        return np.einsum("ii -> ", log_full_likelihood[q[0], q[1]])  # E log q(d) = 1 * log(1) = 0
    return np.einsum("ijk, ijk ->", q, log_full_likelihood - np.log(q + 1e-12))


def run_e_step(X, F, B, s, A, use_MAP=False):
    """
    Given the current esitmate of the parameters, for each image Xk
    esitmates the probability p(d_k|X_k,F,B,s,A).

    Parameters
    ----------
    X : array, shape(H, W, K)
        K images of size H x W.
    F  : array_like, shape(h, w)
        Estimate of villain's face.
    B : array shape(H, W)
        Estimate of background.
    s : scalar, shape(1, 1)
        Eestimate of standard deviation of Gaussian noise.
    A : array, shape(H-h+1, W-w+1)
        Estimate of prior on displacement of face in any image.
    use_MAP : bool, optional,
        If true then q is a MAP estimates of displacement (dh,dw) of
        villain's face given image Xk.

    Returns
    -------
    q : array
        If use_MAP = False: shape (H-h+1, W-w+1, K)
            q[dh,dw,k] - estimate of posterior of displacement (dh,dw)
            of villain's face given image Xk
        If use_MAP = True: shape (2, K)
            q[0,k] - MAP estimates of dh for X_k
            q[1,k] - MAP estimates of dw for X_k
    """
    log_likelihood = calculate_log_probability(X, F, B, s)  # H-h+1, W-w+1, K
    log_prior = np.log(A + 1e-12)  # H-h+1, W-w+1
    posterior = sc.special.softmax(log_likelihood + log_prior[..., None], axis=(0, 1))
    if not use_MAP:
        return posterior
    else:
        ind = np.unravel_index(np.argmax(posterior.reshape(-1, posterior.shape[-1]), axis=0), posterior.shape[:2])
        return np.array(list(zip(ind[0], ind[1]))).T


def run_m_step(X, q, h, w, use_MAP=False):
    """
    Estimates F,B,s,A given esitmate of posteriors defined by q.

    Parameters
    ----------
    X : array, shape(H, W, K)
        K images of size H x W.
    q  :
        if use_MAP = False: array, shape (H-h+1, W-w+1, K)
           q[dh,dw,k] - estimate of posterior of displacement (dh,dw)
           of villain's face given image Xk
        if use_MAP = True: array, shape (2, K)
            q[0,k] - MAP estimates of dh for X_k
            q[1,k] - MAP estimates of dw for X_k
    h : int
        Face mask height.
    w : int
        Face mask width.
    use_MAP : bool, optional
        If true then q is a MAP estimates of displacement (dh,dw) of
        villain's face given image Xk.

    Returns
    -------
    F : array, shape (h, w)
        Estimate of villain's face.
    B : array, shape (H, W)
        Estimate of background.
    s : float
        Estimate of standard deviation of Gaussian noise.
    A : array, shape (H-h+1, W-w+1)
        Estimate of prior on displacement of face in any image.
    """
    H, W, K = X.shape
    if use_MAP:
        A = np.zeros((H - h + 1, W - w + 1, K))
        A[q[0], q[1], np.arange(K)] = 1
        A = np.sum(A, axis=-1) / K

        X = np.transpose(X, (2, 0, 1))
        F = np.sum(X[np.arange(K)[:, None, None], (q[0][:, None] + np.arange(h))[..., None], (q[1][:, None] + np.arange(
            w))[:, None, :]], axis=0)
        X = np.transpose(X, (1, 2, 0))
        F /= K

        q_conv = np.zeros((K, H, W)) - 1
        q_conv[np.arange(K)[:, None, None], (q[0][:, None] + np.arange(h))[..., None], (q[1][:, None] + np.arange(w))[:,
                                                                                       None, :]] = np.zeros((K, h, w))
        q_conv = np.transpose(q_conv, (1, 2, 0))
        numerator = np.sum(X * q_conv, axis=-1)
        denominator = np.sum(q_conv, axis=-1)
        mask = denominator != 0
        B = np.zeros((H, W))
        B[mask] = numerator[mask] / denominator[mask]

        ll = np.sum((X - B[..., None]) ** 2, axis=(0, 1))[None, None, :] - \
             fftconvolve(B ** 2, np.ones((h, w)), mode="valid")[..., None] + 2 * fftconvolve(X * B[..., None],
                                                                                             np.ones((h, w, 1)),
                                                                                             mode="valid") + np.sum(
            F ** 2) - 2 * fftconvolve(X, F[::-1, ::-1, None], mode="valid")
        s = np.sum(ll[q[0], q[1], np.arange(K)]) / (H * W * K)

    else:
        A = np.sum(q, axis=-1)
        A /= K

        F = np.sum(fftconvolve(X, q[::-1, ::-1], mode="valid", axes=(0, 1)), axis=-1)
        F /= K

        q_conv = fftconvolve(q, np.ones((h, w, 1))) - 1
        numerator = np.sum(X * q_conv, axis=-1)
        denominator = np.sum(q_conv, axis=-1)
        mask = numerator == 0
        B = np.zeros((H, W))
        B[~mask] = numerator[~mask] / denominator[~mask]

        ll = np.sum((X - B[..., None]) ** 2, axis=(0, 1))[None, None, :] - \
             fftconvolve(B ** 2, np.ones((h, w)), mode="valid")[..., None] + 2 * fftconvolve(X * B[..., None],
                                                                                             np.ones((h, w, 1)),
                                                                                             mode="valid") + np.sum(
            F ** 2) - 2 * fftconvolve(X, F[::-1, ::-1, None], mode="valid")
        s = np.einsum("ijk, ijk ->", q, ll) / (H * W * K)
    return F, B, np.sqrt(s + 1e-12), A


def run_EM(X, h, w, F=None, B=None, s=None, A=None, tolerance=0.001,
           max_iter=50, use_MAP=False):
    """
    Runs EM loop until the likelihood of observing X given current
    estimate of parameters is idempotent as defined by a fixed
    tolerance.

    Parameters
    ----------
    X : array, shape (H, W, K)
        K images of size H x W.
    h : int
        Face mask height.
    w : int
        Face mask width.
    F : array, shape (h, w), optional
        Initial estimate of villain's face.
    B : array, shape (H, W), optional
        Initial estimate of background.
    s : float, optional
        Initial estimate of standard deviation of Gaussian noise.
    A : array, shape (H-h+1, W-w+1), optional
        Initial estimate of prior on displacement of face in any image.
    tolerance : float, optional
        Parameter for stopping criterion.
    max_iter  : int, optional
        Maximum number of iterations.
    use_MAP : bool, optional
        If true then after E-step we take only MAP estimates of displacement
        (dh,dw) of villain's face given image Xk.

    Returns
    -------
    F, B, s, A : trained parameters.
    LL : array, shape(number_of_iters,)
        L(q,F,B,s,A) after each EM iteration (1 iteration = 1 e-step + 1 m-step); 
        number_of_iters is actual number of iterations that was done.
    """
    H, W, K = X.shape
    if F is None:
        F = np.random.rand(h, w)
    if B is None:
        B = np.random.rand(H, W)
    if s is None:
        s = 1.0
    if A is None:
        A = np.ones((H - h + 1, W - w + 1))
        A /= np.sum(A)
    LL = []
    for i in range(max_iter):
        q = run_e_step(X, F, B, s, A, use_MAP)
        F, B, s, A = run_m_step(X, q, h, w, use_MAP)
        lower_bound = calculate_lower_bound(X, F, B, s, A, q, use_MAP)
        if len(LL) and lower_bound - LL[-1] <= tolerance:
            LL.append(lower_bound)
            break
        else:
            LL.append(lower_bound)
    return F, B, s, A, np.array(LL)


def run_EM_with_restarts(X, h, w, tolerance=0.001, max_iter=50, use_MAP=False,
                         n_restarts=10):
    """
    Restarts EM several times from different random initializations
    and stores the best estimate of the parameters as measured by
    the L(q,F,B,s,A).

    Parameters
    ----------
    X : array, shape (H, W, K)
        K images of size H x W.
    h : int
        Face mask height.
    w : int
        Face mask width.
    tolerance, max_iter, use_MAP : optional parameters for EM.
    n_restarts : int
        Number of EM runs.

    Returns
    -------
    F : array,  shape (h, w)
        The best estimate of villain's face.
    B : array, shape (H, W)
        The best estimate of background.
    s : float
        The best estimate of standard deviation of Gaussian noise.
    A : array, shape (H-h+1, W-w+1)
        The best estimate of prior on displacement of face in any image.
    L : float
        The best L(q,F,B,s,A).
    """
    best_L, best_F, best_B, best_s, best_A = None, None, None, None, None
    for i in range(n_restarts):
        F, B, s, A, LL = run_EM(X, h, w, tolerance=tolerance, max_iter=max_iter, use_MAP=use_MAP)
        if best_L is None or best_L < LL[-1]:
            best_L = LL[-1]
            best_F = F
            best_B = B
            best_s = s
            best_A = A
    return best_F, best_B, best_s, best_A, best_L
