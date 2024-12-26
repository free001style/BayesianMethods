import numpy as np
from scipy.stats import binom
from scipy.stats import poisson


def pa(params, model):
    val = np.arange(params["amin"], params["amax"] + 1)
    prob = 1 / (params["amax"] - params["amin"] + 1) * np.ones_like(val)
    return prob, val


def pb(params, model):
    val = np.arange(params["bmin"], params["bmax"] + 1)
    prob = 1 / (params["bmax"] - params["bmin"] + 1) * np.ones_like(val)
    return prob, val


def pc_ab(params, model, a, b):
    val = np.arange(params["amax"] + params["bmax"] + 1)
    if model == 3:
        # немного изменил логику, чтобы считалось быстрее
        a_binom = binom.pmf(np.arange(params["amax"] + 1)[:, None], a, params["p1"])
        b_binom = binom.pmf(np.arange(params["bmax"] + 1)[:, None], b, params["p2"])
        prob = np.empty((len(val), len(a), len(b)))
        for i in range(len(a)):
            for j in range(len(b)):
                prob[:, i, j] = np.convolve(a_binom[:, i], b_binom[:, j])
    elif model == 4:
        prob = poisson.pmf(
            val[:, None, None], a[:, None] * params["p1"] + b[None, :] * params["p2"]
        )  # c x a x b
    return prob, val


def pc(params, model):
    a_prob, a_val = pa(params, model)
    b_prob, b_val = pb(params, model)
    c_ab_prob, c_val = pc_ab(params, model, a_val, b_val)
    prob = c_ab_prob.sum((1, 2)) * a_prob[0] * b_prob[0]
    return prob, c_val


def pd_c(params, model, c):
    val = np.arange((params["amax"] + params["bmax"]) * 2 + 1)
    prob = binom.pmf(val[:, None] - c[None, :], c, params["p3"])  # d x c
    return prob, val


def pd(params, model):
    c_prob, c_val = pc(params, model)
    d_c_prob, d_val = pd_c(params, model, c_val)
    prob = (d_c_prob * c_prob[None, :]).sum(axis=1)
    return prob, d_val


def generate(N, a, b, params, model):
    seed = 0
    if model == 3:
        # binom.rvs не проходит в контесте
        a_part = np.random.binomial(a[None, :], params["p1"], size=(N, len(a)))
        b_part = np.random.binomial(b[None, :], params["p2"], size=(N, len(b)))
        c = a_part[..., None] + b_part[:, None, :]  # N x a x b
    elif model == 4:
        c = np.random.poisson(
            a[None, :, None] * params["p1"] + b[None, None, :] * params["p2"],
            size=(N, len(a), len(b)),
        )  # N x a x b
    return c + binom.rvs(c, params["p3"], random_state=seed)


def pd_ab(params, model, a, d=None):
    _, b_val = pb(params, model)
    c_ab_prob, c_val = pc_ab(params, model, a, b_val)
    d_c_prob, d_val = pd_c(params, model, c_val)
    if d is None:
        d_ab_prob = np.einsum("ij, jkl -> ikl", d_c_prob, c_ab_prob)
    else:
        d_ab_prob = np.einsum("ijk, klm -> ijlm", d_c_prob[d, :], c_ab_prob)
    return d_ab_prob, d_val


def pb_d(d, params, model):
    k_d, N = d.shape
    a_prob, a_val = pa(params, model)
    b_prob, b_val = pb(params, model)
    d_b_prob, d_val = pd_ab(params, model, a_val, d)
    numerator = np.prod(d_b_prob, axis=1).sum(1) * b_prob[0] * a_prob[0]
    denominator = numerator.sum(-1)
    b_d_prob = numerator / denominator[:, None]
    return b_d_prob.T, b_val


def pb_ad(a, d, params, model):
    k_d, N = d.shape
    d_ab_prob, _ = pd_ab(params, model, a)
    b_prob, b_val = pb(params, model)
    d_ab_prob_wrt_a_d = d_ab_prob[d.reshape(k_d * N), ...].reshape(k_d, N, len(a), -1)
    numerator = np.prod(d_ab_prob_wrt_a_d, axis=1) * b_prob[0]
    denominator = numerator.sum((1, 2))
    b_ad_prob = numerator / denominator[:, None, None]
    return b_ad_prob.transpose(2, 1, 0), b_val


a = np.arange(80, 82)
b = np.arange(500, 503)
d = np.tile(np.arange(40, 44), [7, 1])
params = {'amin': 75, 'amax': 90, 'bmin': 500, 'bmax': 600,
          'p1': 0.1, 'p2': 0.01, 'p3': 0.3}
print(pb_d(d, params, 4)[0])
