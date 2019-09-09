# -*- coding: utf-8 -*-
"""
A modified iterative learning control (ILC) algorithm is demonstrated for single input single output (siso) linear.
The modification is an error shaping (key idea of ILC) and smooths the response improving convergence in the case of
nonlinear dynamics. Script contains one example each of the control linear and nonlinear dynamical systems.

Author: Kaivalya Bakshi
Date: 15 Apr 2019
"""

import math
import numpy as np
import matplotlib.pyplot as plt

class performILC:
    def __init__(self, T, K, gamma, example):
        self.example = example
        if example == 'linear':
            self.stateTrans = np.array([[-0.8, -0.22], [1, 0]])
            self.ctrlTrans = np.array([0.5, 1]).reshape(2, 1)
            self.obsTrans = np.array([1, 0.5]).reshape(1, 2)
            self.x0 = np.array([0, 0])
        else:
            self.x0 = [np.pi / 2]  # initial state
        self.gamma = gamma  # learning rate of Arimoto ILC
        self.T = T  # total time of process
        self.k = 0  # initializing episode number
        self.K = K  # maximum number of episodes
        self.y0 = 0  # initial output
        self.u_0 = [0]*self.T  # baseline policy

    def transDyn(self, x, u, example):  # state transition dynamics x(t+1) = f(x(t), u(t)) and output equation y(t) = g(x(t))
        if example == 'linear':  # 2d state and 1d control
            x = (self.stateTrans @ x.reshape(2, 1)).reshape(2, 1) + (self.ctrlTrans * u).reshape(2, 1)
            y = float(self.obsTrans @ x)
        elif example == 'nonlinear':
            x = np.cos(x) + u
            y = x
        return x, y

    def ctrlComp(self, u, e):
        u = u + self.gamma*e  # Arimoto u_(k+1) = u_k + gamma*e_k
        return u

    def trajComp(self, u_prev, e_prev):
        k = self.k;  T = self.T;
        x0 = self.x0;  u_0 = self.u_0
        t = int(0); x = x0
        y = [None]*T;  yd = [None]*T;  e = [None]*T; u = [None]*T
        for t in np.arange(T):
            if k == 0:
                u[t] = u_0[t]
            elif t == T-1:
                u[t] = self.ctrlComp(u_prev[t], e_prev[t] + (e_prev[t] - e_prev[t-1]))
                # modified error shaping of the Arimoto ILC algorithm to use the predictive capability of derivative
                # action to learn from the previous episode at the last time step since we don't have access to e_k(T+1)
            else:
                u[t] = self.ctrlComp(u_prev[t], (e_prev[t] + e_prev[t + 1])/2)
                # modified error shaping of the Arimoto ILC algorithm using a one step moving average smoothing, to
                # smooth the response and allow good convergence for nonlinear dynamics
            x, y[t] = self.transDyn(x, u[t], self.example)
            yd[t] = np.sin(8*t/100)
            e[t] = yd[t] - y[t]
            if t == T:
                yd.pop();  y.pop();  u.pop()
            t += 1
            print('e(t) = ' + str(e[t-1]) + ' at ' + 'time step ' + str(t) + ' in iteration ' + str(self.k))
        return yd, y, e, u

    def ILC(self):
        k = self.k;  K= self.K; T = self.T
        traj = [None] *(K+1);
        while k <= K:
            if k == 0:
                e_prev = [None]*self.T;  u_prev = self.u_0
            # breakpoint()
            yd, traj[k], e, u = self.trajComp(u_prev, e_prev)
            e_prev, u_prev = e, u
            # making a nice simulation in time over all learning episodes
            plt.figure(0)
            if k == 0:
                plt.plot(np.arange(self.T), yd, 'k-.', markersize=2, label='reference')
            for t in range(T):
                plt.plot(t, traj[k][t], 'r_', markersize=1, alpha=(0.75 + 0.25*(k+1)/(K+1)))
                plt.legend()
                # plt.pause(1/T**3)
            label = 'episode ' + str(k)
            plt.plot(np.arange(T), traj[k], label=label)
            k += 1
            self.k = k  # required to communicate the iteration number to trajComp(self, u_prev, e_prev)
            plt.title('Iterative learning control over all episodes')
            plt.xlabel('time step')
            plt.ylabel('output')
        return yd, traj

if __name__ == '__main__':
    plt.close('all')
    # siso_linear = performILC(T=100, K=10, gamma=0.5, example='linear')
    # yd, traj = siso_linear.ILC()
    # plt.grid()
    # plt.savefig('lin_sisoILC')
    # plt.clf()
    siso_nonlinear = performILC(T=100, K=11, gamma=0.5, example='nonlinear')
    yd, traj = siso_nonlinear.ILC()
    plt.savefig('nonlin_sisoILC')
    plt.grid()