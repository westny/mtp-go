import torch
from torchdiffeq import odeint


def forward_euler(model, X, u, static_f, h):
    """ Forward Euler (first-order) method"""
    return X + h * model(X, u, static_f)


def midpoint(model, X, u, static_f, h):
    """ Explicit midpoint (second-order) method """
    k1 = model(X, u, static_f)
    k2 = model(X + h / 2 * k1, u, static_f)
    return X + h * k2


def heun(model, X, u, static_f, h):
    """ Heun's (second-order) method """
    k1 = model(X, u, static_f)
    k2 = model(X + h * k1, u, static_f)
    return X + h / 2 * (k1 + k2)


def rk3(model, X, u, static_f, h):
    """ Kutta's third-order method """
    k1 = model(X, u, static_f)
    k2 = model(X + h / 2 * k1, u, static_f)
    k3 = model(X + h * (2 * k2 - k1), u, static_f)
    return X + h / 6 * (k1 + 4 * k2 + k3)


def ssprk3(model, X, u, static_f, h):
    """ Third-order Strong Stability Preserving Runge-Kutta (SSPRK3) """
    k1 = model(X, u, static_f)
    k2 = model(X + h * k1, u, static_f)
    k3 = model(X + h / 4 * (k1 + k2), u, static_f)
    return X + h / 6 * (k1 + k2 + 4 * k3)


def rk4(model, X, u, static_f, h):
    """ Classic fourth-order method """
    k1 = model(X, u, static_f)
    k2 = model(X + h / 2 * k1, u, static_f)
    k3 = model(X + h / 2 * k2, u, static_f)
    k4 = model(X + h * k3, u, static_f)
    return X + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


# Solver implementations used in torchdiffeq package: https://github.com/rtqichen/torchdiffeq
# presented in "Neural Ordinary Differential Equations": https://arxiv.org/abs/1806.07366

def diff_eq(t, X, model):
    x, u, s = X
    dx = model(x, u, s)
    du = torch.zeros_like(u)
    ds = torch.zeros_like(s)
    return dx, du, ds


def rk45(model, X, inp, static_f, h):
    f = lambda t, x0: diff_eq(t, x0, model)
    dx, du, ds = odeint(f, (X, inp, static_f), t=torch.tensor([0., h], device=X.device), method='dopri5')
    return dx[-1]


def impl_adam(model, X, inp, static_f, h):
    f = lambda t, x0: diff_eq(t, x0, model)
    dx, du, ds = odeint(f, (X, inp, static_f), t=torch.tensor([0., h], device=X.device),
                        method='implicit_adams', options=dict(step_size=h))
    return dx[-1]


solvers = {}
solvers['ef'] = forward_euler
solvers['mp'] = midpoint
solvers['heun'] = heun
solvers['rk3'] = rk3
solvers['ssprk3'] = ssprk3
solvers['rk4'] = rk4
solvers['dopri5'] = rk45
solvers['impl_adam'] = impl_adam
