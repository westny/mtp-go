import math
import torch
import torch.nn as nn
from .ode_solvers import *
from torch.func import vmap, jacrev


class MotionModelBase(nn.Module):
    def __init__(self, solver='rk4', dt=4e-2, n_states=4, mixtures=8, static_f_dim=6,
                 *u_lims):
        super(MotionModelBase, self).__init__()
        self.dt = dt
        self.mixtures = mixtures
        self.n_states = n_states
        self.static_f_dim = static_f_dim
        self.n_inputs = len(u_lims)
        self.u_constrain = []
        self.solver = solvers[solver]

        for u_lim in u_lims:
            self.u_constrain.append(nn.Hardtanh(-u_lim, u_lim))
        self.G = nn.Parameter(self.build_inp_transition_matrix(), requires_grad=False)

    def build_inp_transition_matrix(self):
        G = torch.zeros(1, self.mixtures, self.n_states, self.n_inputs)
        idx_offset = self.n_states - self.n_inputs
        G[..., idx_offset, 0] = torch.ones(1, self.mixtures) * self.dt
        G[..., idx_offset + 1, 1] = torch.ones(1, self.mixtures) * self.dt
        return G

    def model_update(self, X, u, static_f):
        return X

    @torch.inference_mode(False)
    def state_transition_matrix(self, X, inp, static_f):
        N, m, feat_dim = X.shape
        fx = lambda state, inputs, static_input: solvers['ef'](self.model_update, state,
                                                               inputs, static_input, self.dt)
        jacobian_rev = vmap(jacrev(fx, argnums=0))(X.flatten(0, 1), inp.flatten(0, 1), static_f.flatten(0, 1))
        F = jacobian_rev.view(N, m, feat_dim, feat_dim)
        return F, torch.transpose(F, dim0=-2, dim1=-1)

    def input_transition_matrix(self, X, u):
        batch_size = X.size(0)
        G = self.G.clone().expand(batch_size, -1, -1, -1)
        return G, torch.transpose(G, dim0=-2, dim1=-1)

    def forward(self, past_state, inputs, static_f):
        n_inputs = inputs.size(-1)
        input_clamped = [self.u_constrain[i](inputs[..., i]) for i in range(n_inputs)]
        input_clamped = torch.stack(input_clamped, dim=-1)
        next_state = self.solver(self.model_update, past_state, input_clamped,
                                 static_f, self.dt)
        return next_state, input_clamped


class FirstOrderNeuralODE(MotionModelBase):
    """
        x, y = NN(x, y, u1, u2)
    """

    def __init__(self, solver='rk4', dt=4e-2, n_states=2, mixtures=8, static_f_dim=6,
                 n_hidden=32, n_layers=2, u1_lim=10, u2_lim=10):
        super().__init__(solver, dt, n_states, mixtures, static_f_dim, u1_lim, u2_lim)
        # n_inputs = 2
        self.n_states = n_states
        self.f = nn.ModuleList([self.create_net(n_states, 1 + self.static_f_dim,
                                                n_hidden, n_layers) for _ in range(n_states)])

    @staticmethod
    def create_net(n_states, n_inputs, n_hidden, n_layers=2):
        f = nn.Sequential()
        n_inputs += n_states
        n_outputs = 1
        i = 0
        while i < n_layers:
            if i == 0:
                f.append(nn.Linear(n_inputs, n_hidden))
            else:
                f.append(nn.Linear(n_hidden, n_hidden))
            f.append(nn.ELU(inplace=True))
            i += 1
        else:
            if not len(f):
                f.append(nn.Linear(n_inputs, n_outputs))
            else:
                f.append(nn.Linear(n_hidden, n_outputs))
        return f

    def model_update(self, X, u, static_f):
        u_x = u[..., 0:1]
        u_y = u[..., 1:2]

        if self.static_f_dim > 0:
            inp_x_cat = (X, u_x, static_f)
            inp_y_cat = (X, u_y, static_f)
        else:
            inp_x_cat = (X, u_x)
            inp_y_cat = (X, u_y)

        inp_x = torch.cat(inp_x_cat, dim=-1)
        inp_y = torch.cat(inp_y_cat, dim=-1)
        dx = self.f[0](inp_x)
        dy = self.f[1](inp_y)
        dX = torch.cat((dx, dy), dim=-1)
        return dX


class SecondOrderNeuralODE(MotionModelBase):
    """
        x = vx
        y = vy
        vx, vy = NN(vx, vy, u1, u2)
    """

    def __init__(self, solver='rk4', dt=4e-2, n_states=4, mixtures=8, static_f_dim=6,
                 n_hidden=32, n_layers=2, u1_lim=10, u2_lim=10):
        super().__init__(solver, dt, n_states, mixtures, static_f_dim, u1_lim, u2_lim)
        # n_inputs = 2
        self.n_states = n_states
        self.f = nn.ModuleList([self.create_net(n_states // 2, 1 + self.static_f_dim,
                                                n_hidden, n_layers) for _ in range(n_states // 2)])

        # x, y
        self.m0 = nn.Parameter(torch.zeros(n_states, n_states // 2), requires_grad=False)
        self.m0[2, 0] = self.m0[3, 1] = 1.

        # vy
        self.m1 = nn.Parameter(torch.zeros(6, 3), requires_grad=False)
        self.m1[2, 0] = self.m1[3, 1] = self.m1[4, 2] = 1.

        # vx
        self.m2 = nn.Parameter(torch.zeros(6, 3), requires_grad=False)
        self.m2[2, 0] = self.m2[3, 1] = self.m2[5, 2] = 1.

    @staticmethod
    def create_net(n_states, n_inputs, n_hidden, n_layers=2):
        f = nn.Sequential()
        n_inputs += n_states
        i = 0
        while i < n_layers:
            if i == 0:
                f.append(nn.Linear(n_inputs, n_hidden))
            else:
                f.append(nn.Linear(n_hidden, n_hidden))
            f.append(nn.ELU(inplace=True))
            i += 1
        else:
            if not len(f):
                f.append(nn.Linear(n_inputs, 1))
            else:
                f.append(nn.Linear(n_hidden, 1))
        return f

    def model_update(self, X, u, static_f):
        inp = torch.cat((X, u), dim=-1)

        if self.static_f_dim > 0:
            # Append static features
            inp_vx = torch.cat((inp @ self.m2, static_f), dim=-1)
            inp_vy = torch.cat((inp @ self.m1, static_f), dim=-1)
        else:
            inp_vx = inp @ self.m2
            inp_vy = inp @ self.m1

        dvx = self.f[0](inp_vx)
        dvy = self.f[1](inp_vy)
        dX = torch.cat((X @ self.m0, dvx, dvy), dim=-1)
        return dX


class KinematicSingleTrack(MotionModelBase):
    """
        beta = atan((lr / (lr + lf) * tan(u1)))
        dx = v * cos(psi + beta)
        dy = v * sin(psi + beta)
        dv = u2
        dpsi = v / lr * sin(beta)
    """

    def __init__(self, solver='rk4', dt=2e-1, n_states=4, mixtures=8,
                 u1_lim=math.pi / 4, u2_lim=10):
        super().__init__(solver, dt, n_states, mixtures, 2, u1_lim, u2_lim)
        self.t_states = n_states - 1

    def model_update(self, X, inp, static_f):
        df = inp[..., 0]
        dv = inp[..., 1]

        lf = static_f[..., 0]
        lr = static_f[..., 1]

        beta = torch.atan((lr / (lf + lr) * torch.tan(df)))

        dx = X[..., 2] * torch.cos(X[..., 3] + beta)
        dy = X[..., 2] * torch.sin(X[..., 3] + beta)
        dpsi = X[..., 2] * torch.sin(beta) / lr

        dX = torch.stack((dx, dy, dv, dpsi), dim=-1)
        return dX


class Unicycle(MotionModelBase):
    """
        x = v * cos(psi)
        y = v * sin(psi)
        psi = u1
        v = u2
    """

    def __init__(self, solver='rk4', dt=2e-1, n_states=4, mixtures=8, u1_lim=math.pi, u2_lim=10):
        super().__init__(solver, dt, n_states, mixtures, 0, u1_lim, u2_lim)

        self.t_states = n_states - 1

    def model_update(self, X, u, static_f):
        dx = X[..., 2] * torch.cos(X[..., 3])
        dy = X[..., 2] * torch.sin(X[..., 3])
        dpsi = u[..., 0]
        dvx = u[..., 1]
        return torch.stack((dx, dy, dvx, dpsi), dim=-1)


class Curvature(MotionModelBase):
    """
        x = v * cos(psi)
        y = v * sin(psi)
        psi = u1 * v
        v = u2
    """

    def __init__(self, solver='rk4', dt=2e-1, n_states=4, mixtures=8, u1_lim=0.5, u2_lim=10):
        super().__init__(solver, dt, n_states, mixtures, 0, u1_lim, u2_lim)

        self.t_states = n_states - 1

    def model_update(self, X, u, static_f):
        dx = X[..., 2] * torch.cos(X[..., 3])
        dy = X[..., 2] * torch.sin(X[..., 3])
        dpsi = u[..., 0] * X[..., 2]
        dvx = u[..., 1]
        return torch.stack((dx, dy, dvx, dpsi), dim=-1)


class CurviLinear(MotionModelBase):
    """
        x = v * cos(psi)
        y = v * sin(psi)
        psi = u1 / v
        v = u2
    """

    def __init__(self, solver='rk4', dt=2e-1, n_states=4, mixtures=8, u1_lim=10, u2_lim=8):
        super().__init__(solver, dt, n_states, mixtures, 0, u1_lim, u2_lim)
        self.eps = 1e-5
        self.t_states = n_states - 1

    def model_update(self, X, u, static_f):
        dx = X[..., 2] * torch.cos(X[..., 3])
        dy = X[..., 2] * torch.sin(X[..., 3])
        dpsi = u[..., 0] / (X[..., 2] + self.eps)
        dpsi = dpsi * (X[..., 2] > 1e-1)
        dvx = u[..., 1]
        return torch.stack((dx, dy, dvx, dpsi), dim=-1)


class SingleIntegrator(MotionModelBase):
    """
        x = u1
        y = u2
    """

    def __init__(self, solver='rk4', dt=2e-1, n_states=2, mixtures=8, u1_lim=12, u2_lim=60):
        super().__init__(solver, dt, n_states, mixtures, 0, u1_lim, u2_lim)
        self.m1 = nn.Parameter(torch.zeros(n_states, n_states), requires_grad=False)
        self.m1[0, 1] = self.m1[1, 0] = 1.

        self.jac_solver = solvers['ef']

    def model_update(self, X, u, static_f):
        """
        dx = u[..., 1]
        dy = u[..., 0]
        """
        dX = u @ self.m1
        return dX


class DoubleIntegrator(MotionModelBase):
    """
        x = vx
        y = vy
        vx = u1
        vy = u2
    """

    def __init__(self, solver='rk4', dt=2e-1, n_states=4, mixtures=8, u1_lim=8, u2_lim=10):
        super().__init__(solver, dt, n_states, mixtures, 0, u1_lim, u2_lim)
        self.m0 = nn.Parameter(torch.zeros(n_states, n_states // 2), requires_grad=False)
        self.m0[2, 0] = self.m0[3, 1] = 1.

        self.m1 = nn.Parameter(torch.zeros(n_states // 2, n_states // 2), requires_grad=False)
        self.m1[0, 1] = self.m1[1, 0] = 1.

        self.jac_solver = solvers['ef']

    def model_update(self, X, u, static_f):
        """
            dx = X[..., 2]
            dy = X[..., 3]
            dvx = u[..., 1]
            dvy = u[..., 0]
        """
        dX = torch.cat((X @ self.m0, u @ self.m1), dim=-1)
        return dX


class TripleIntegrator(MotionModelBase):
    """
        x = vx
        y = vy
        vx = ax
        vy = ay
        ax = u2
        ay = u1
    """

    def __init__(self, solver='rk4', dt=2e-1, n_states=6, mixtures=8, u1_lim=18, u2_lim=22):
        super().__init__(solver, dt, n_states, mixtures, 0, u1_lim, u2_lim)

        self.m0 = nn.Parameter(torch.zeros(n_states, n_states - 2), requires_grad=False)
        self.m0[2, 0] = self.m0[3, 1] = self.m0[4, 2] = self.m0[5, 3] = 1.

        self.m1 = nn.Parameter(torch.zeros(2, 2), requires_grad=False)
        self.m1[0, 1] = self.m1[1, 0] = 1.

        self.jac_solver = solvers['ef']

    def model_update(self, X, u, static_f):
        """
        dx = X[..., 2]
        dy = X[..., 3]

        dvx = X[..., 4]
        dvy = X[..., 5]

        dax = u[..., 1]
        day = u[..., 0]
        """
        dX = torch.cat((X @ self.m0, u @ self.m1), dim=-1)
        return dX
