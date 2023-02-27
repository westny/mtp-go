def forward_euler(model, X, u, static_f, h):
    """ Forward Euler (first-order) method"""
    return X + h * model(X, u, static_f)


def rk4(model, X, u, static_f, h):
    """ Classic fourth-order method """
    k1 = model(X, u, static_f)
    k2 = model(X + h / 2 * k1, u, static_f)
    k3 = model(X + h / 2 * k2, u, static_f)
    k4 = model(X + h * k3, u, static_f)
    return X + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


solvers = {}
solvers['ef'] = forward_euler
solvers['rk4'] = rk4
