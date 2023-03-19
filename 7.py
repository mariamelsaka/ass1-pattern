from scipy.optimize import minimize, Bounds, LinearConstraint
import scipy.special as sc
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

Lmin, Lnom, Lmax = 0.8035, 2.3811, 3.084
L10, L20 = 2.3, 2.3
# one step look ahead
# Eindf = [1116, 1160, 841, 1116]
# Dindf = [44, 861, 275]
# Ldf = [4.2, 5.2, 4.8, 4.4]
Eindf = [1116, 1160, 841]
Dindf = [44, 861]
Ldf = [4.2, 5.2, 4.8]

R10, R20 = 0.1803, 0.1803
Rfc1, Rfc2 = R10 + 0.003, R20 + 0.0

if Rfc1 > Rfc2:
    w1 = (Rfc1 * np.exp((Rfc1 - Rfc2)) + Rfc1) / (Rfc2 + np.exp((Rfc1 - Rfc2)) * Rfc1 + Rfc1)
    w2 = Rfc2 / (Rfc2 + np.exp((Rfc1 - Rfc2)) * Rfc1 + Rfc1)
else:
    w1 = (Rfc1) / (Rfc2 + np.exp((Rfc1 - Rfc2)) * Rfc1 + Rfc1)
    w2 = (Rfc2 * np.exp((Rfc2 - Rfc1) + Rfc2)) / (Rfc2 * np.exp((Rfc2 - Rfc1)) + Rfc2 + Rfc1)

β = 0.00873

Ratio_VtoR = 5.39
Dv_ΔL = 5.93e-7
RatioD_ΔL = 20
Dr_ΔL = RatioD_ΔL * Dv_ΔL * Ratio_VtoR
k = (R10 * Dr_ΔL) / (Lmax - Lmin)


def alpha_beta(x):
    # a, b are params fitted for parabola functions
    a = 0.0019727939
    b = 0.0078887
    Lmin, Lnom, Lmax = 0.8035, 2.3811, 3.084
    return np.piecewise(x, [np.logical_and(Lmin <= x, x < Lnom),
                            np.logical_and(Lnom <= x, x <= Lmax)],
                        [lambda x: a * ((x - Lnom) ** 2) + 0.006226,
                         lambda x: b * ((x - Lnom) ** 2) + 0.006226, 0])


def func1(Dind):
    crit = lambda x: (k * (w1 * abs(x[0] - L10) + w2 * abs(x[1] - L20)) + \
                      β * Dind[0] * (w1 * alpha_beta(x[0]) + w2 * alpha_beta(x[1])) + β * Dind[1] * (
                                  w1 * alpha_beta(x[2]) + w2 * alpha_beta(x[3])) + \
                      k * (w1 * abs(x[2] - x[0]) + w1 * abs(x[4] - x[2]) + w2 * abs(x[3] - x[1]) + w2 * abs(
                x[5] - x[3])))
    return crit


def func2(Dind):
    crit = lambda x: (k * (w1 * abs(x[0] - L10) + w2 * abs(x[1] - L20)) + β * Dind[1] * (
                w1 * alpha_beta(x[2]) + w2 * alpha_beta(x[3])) + \
                      β * Dind[0] * (w1 * alpha_beta(x[0]) + w2 * alpha_beta(x[1])) + \
                      k * (w1 * abs(x[2] - x[0]) + w1 * abs(x[4] - x[2]) + w2 * abs(x[3] - x[1]) + w2 * abs(
                x[5] - x[3])))
    return crit


def func(Dind):
    Cr1 = lambda x: (k * (w1 * abs(x[0] - L10) + w2 * abs(x[1] - L20)) + \
                     sum((β * Dind[i] * (w1 * alpha_beta(x[2 * i]) + w2 * alpha_beta(x[2 * i + 1])) + \
                          k * (w1 * abs(x[2 * (i + 1)] - x[2 * i]) + w2 * abs(x[2 * (i + 1) + 1] - x[2 * i + 1]))) for i
                         in range(len(Dind))))
    return Cr1


bounds = Bounds([Lmin] * 2 * len(Ldf), [Lmax] * 2 * len(Ldf))

A_mat = np.zeros((len(Ldf), 2 * len(Ldf)))
for i in range(len(Ldf)):
    A_mat[i, 2 * i] = 1
    A_mat[i, 2 * i + 1] = 1

linear_cons = LinearConstraint(A_mat, np.array(Ldf), (1.08 * np.array(Ldf)))

x0 = np.repeat(Ldf, 2) / 2

res = minimize(func(Dindf), x0, method='trust-constr', constraints=linear_cons, options={'disp': 0, 'xtol': 1e-5,
                                                                                         'maxiter': 80, 'gtol': 1e-7,
                                                                                         'verbose': 0}, bounds=bounds)
print(f'func:{res.x}')

res = minimize(func1(Dindf), x0, method='trust-constr', constraints=linear_cons, options={'disp': 0, 'xtol': 1e-5,
                                                                                          'maxiter': 80, 'gtol': 1e-7,
                                                                                          'verbose': 0}, bounds=bounds)
print(f'func1:{res.x}')

res = minimize(func2(Dindf), x0, method='trust-constr', constraints=linear_cons, options={'disp': 0, 'xtol': 1e-5,
                                                                                          'maxiter': 80, 'gtol': 1e-7,
                                                                                          'verbose': 0}, bounds=bounds)
print(f'func2:{res.x}')