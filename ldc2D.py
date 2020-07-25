#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
   Author: Kamil Rocki
   Date: 5/13/2020

   Lid Driven Cavity 2D

   Solving N-S equations for incompressible flow via
   Finite-Volume Method and SIMPLE (Semi-Implicit
   Method for Pressure Linked Equations) algorithm

   Patankar, Suhas V: "Numerical heat transfer and fluid flow"
"""

import numpy as np
import scipy.sparse as sp
import time
import argparse
import matplotlib.pyplot as plt
import matplotlib

from cg import cg, cgs, bicg, bicgstab
#matplotlib.use('Agg')

np.set_printoptions(precision=3, suppress=True, linewidth=100)

dtype = 'float64'

### parameters ###

max_iters = 110
lid_velocity = 1

imax=21          # grid size in x-direction
jmax=21          # grid size in y-direction
mu = 0.02        # viscosity
rho = 1          # density
alpha_U = 0.7    # velocity under-relaxation
alpha_P = 0.1    # pressure under-relaxation
tol = 1e-5       # max residual

# derived grid cell sizes
dx = 1/(imax-1)
dy = 1/(jmax-1)

# set up grid
x = np.arange(dx/2, 1+dx/2, dx)
y = np.arange(0,    1+dy,   dy)

# set up arrays
p = np.zeros((imax,   jmax  ), dtype) # pressure
u = np.zeros((imax+1, jmax  ), dtype) # horizontal velocity
v = np.zeros((imax,   jmax+1), dtype) # vertical velocity

# boundary condition
u[:, -1] = lid_velocity

### code for momentum
# convective coefficients
Dw = mu * dy/dx
De = mu * dy/dx
Ds = mu * dx/dy
Dn = mu * dx/dy

# 5.86 - power law scheme
def coeff(F, D):
    return np.maximum(0, (1 - 0.1 * np.abs(F/D))**5)

def compute_u_star(u, v, p, alpha, mu, dx, dy):

  u_star = np.zeros_like(u)
  d_u = np.zeros_like(u) # correction coefficient

  for i in range(1, imax):
    for j in range(0, jmax):

      Fw = .5 * rho * dy * (u[i-1, j  ] + u[i  , j  ]) if i>0      else 0 # 6.9a
      Fe = .5 * rho * dy * (u[i+1, j  ] + u[i  , j  ]) if i<imax   else 0 # 6.9b
      Fs = .5 * rho * dx * (v[i  , j  ] + v[i-1, j  ]) if j>0      else 0 # 6.9c
      Fn = .5 * rho * dx * (v[i  , j+1] + v[i-1, j+1]) if j<jmax-1 else 0 # 6.9d

      # power-law differencing
      aW = Dw * coeff(Fw, Dw) + np.maximum( Fw, 0) if i>0      else 0
      aE = De * coeff(Fe, De) + np.maximum(-Fe, 0) if i<imax   else 0
      aS = Ds * coeff(Fs, Ds) + np.maximum( Fs, 0) if j>0      else 0
      aN = Dn * coeff(Fn, Dn) + np.maximum(-Fn, 0) if j<jmax-1 else 0

      aP = aE + aW + aN + aS + (Fe-Fw) + (Fn-Fs)

      pressure_term = dy * (p[i-1, j] - p[i, j])

      # 6.36
      # only do inner cells
      if j>0 and j<jmax-1 and i>0 and i<imax:
        # stencil
        u_star[i, j] = alpha/aP * (
                    pressure_term +
                    aW * u[i-1, j  ] +
                    aE * u[i+1, j  ] +
                    aS * u[i  , j-1] +
                    aN * u[i  , j+1]
                  ) + (1 - alpha) * u[i, j]

      d_u[i, j] = alpha * dy / aP

  # boundary conditions for outer cells
  u_star[0,  :] = -u_star[ 1, :] # left wall
  u_star[-1, :] = -u_star[-2, :] # right wall
  u_star[:,  0] = 0              # bottom wall
  u_star[:, -1] = lid_velocity   # top (no-wall)

  return u_star, d_u

# this is almost the same as compute_u_star
def compute_v_star(u, v, p, alpha, mu, dx, dy):

  v_star = np.zeros_like(v)
  d_v = np.zeros_like(v) # correction coefficient

  for i in range(0, imax):
    for j in range(1, jmax):

      Fw = .5 * rho * dy * (u[i  , j-1] + u[i  , j  ]) if i>0      else 0
      Fe = .5 * rho * dy * (u[i+1, j  ] + u[i+1, j-1]) if i<imax-1 else 0
      Fs = .5 * rho * dx * (v[i  , j-1] + v[i  , j  ]) if j>0      else 0
      Fn = .5 * rho * dx * (v[i  , j  ] + v[i  , j+1]) if j<jmax   else 0

      # power-law differencing
      aW = Dw * coeff(Fw, Dw) + np.maximum( Fw, 0) if i>0          else 0
      aE = De * coeff(Fe, De) + np.maximum(-Fe, 0) if i<imax-1     else 0
      aS = Ds * coeff(Fs, Ds) + np.maximum( Fs, 0) if j>0          else 0
      aN = Dn * coeff(Fn, Dn) + np.maximum(-Fn, 0) if j<jmax       else 0

      aP = aE + aW + aN + aS + (Fe-Fw) + (Fn-Fs)

      pressure_term = dx * (p[i, j-1] - p[i, j])

      # only do inner cells
      if j>0 and j<jmax and i>0 and i<imax-1:
        # stencil
        v_star[i, j] = alpha/aP * (
                        pressure_term +
                        aW * v[i-1, j  ] +
                        aE * v[i+1, j  ] +
                        aS * v[i  , j-1] +
                        aN * v[i  , j+1]
                      ) + (1 - alpha) * v[i, j]

      d_v[i, j] = alpha * dx / aP

  # boundary conditions
  v_star[ 0, :] = 0 # left wall
  v_star[-1, :] = 0 # right wall
  v_star[ :, 0] = -v_star[ :, 1] # bottom
  v_star[ :,-1] = -v_star[ :,-2] # lid

  return v_star, d_v

### code for pressure
# to calculate rhs vector of the pressure Poisson matrix
def get_rhs(u_star, v_star, rho, dx, dy):

  nx = u_star.shape[1]
  ny = v_star.shape[0]

  n = nx * ny
  stride = ny

  # vector of RHS for solving pressure corrections
  bp = np.zeros((n, 1), dtype)

  for j in range(0, ny):
    for i in range(0, nx):
      idx = j + i * stride
      bp[idx, 0] = rho * (
        (u_star[i, j] - u_star[i+1, j]) * dy +
        (v_star[i, j] - v_star[i, j+1]) * dx )

  # p(0,0) is set to be zero, it has no pressure correction
  bp[0,0] = 0

  return bp

# build the A matrix
def get_coeff_mat(imax, jmax, dx, dy, rho, d_u, d_v):

  N = imax * jmax
  s = jmax # stride

  Ap = np.zeros((N, N), dtype)

  for j in range(jmax):
    for i in range(imax):

      pos = i + j * s

      aE, aW, aN, aS = 0, 0, 0, 0

      if i==0 and j==0:
        Ap[pos, pos] = 1 # no correction
      else:
        if i>0:
          Ap[pos, pos-1] = -rho * d_u[i  , j  ] * dy # sub diagonal
          aW = -Ap[pos, pos-1]

        if i<imax-1:
          Ap[pos, pos+1] = -rho * d_u[i+1, j  ] * dy # upper diagonal
          aE = -Ap[pos, pos+1]

        if j>0:
          Ap[pos, pos-s] = -rho * d_v[i  , j  ] * dx # sub-sub diagonal
          aS = -Ap[pos, pos-s]

        if j<jmax-1:
          Ap[pos, pos+s] = -rho * d_v[i  , j+1] * dx # upper-upper diagonal
          aN = -Ap[pos, pos+s]

        # main diagonal
        aP = aE + aN + aW + aS
        Ap[pos, pos] = aP

  return Ap

# replace with BiCGStab
def lin_solve(A, b):
  # test various linear solvers
  # numpy default
  #return np.linalg.solve(A, b)
  # simple CG
  #return cg(A, b, np.zeros_like(b))[0]
  # CG Squared
  #return cgs(A, b, np.zeros_like(b))[0]
  # BiCG
  #return bicg(A, b, np.zeros_like(b))[0]
  # BiCGStab
  return bicgstab(A, b, np.zeros_like(b))[0]

def pressure_correct(imax, jmax, rhsp, Ap, p, alpha):

  pressure = np.zeros_like(p)
  p_prime_interior = lin_solve(Ap, rhsp);

  p_prime = np.reshape(p_prime_interior, (imax, jmax))

  pressure = p + alpha * p_prime
  pressure[0, 0] = 0

  return pressure, p_prime

def update_velocity(imax, jmax, u_star, v_star, p_prime, d_u, d_v, lid_vel):

  u = np.zeros_like(u_star)
  v = np.zeros_like(v_star)

  for i in range(1, imax):
    for j in range(1, jmax):
      u[i, j] = u_star[i, j] + d_u[i, j] * (p_prime[i-1][j] - p_prime[i, j])

  for i in range(1, imax):
    for j in range(1, jmax):
      v[i, j] = v_star[i, j] + d_v[i, j] * (p_prime[i][j-1] - p_prime[i, j])


  # update boundary conditions
  v[0 , :] = 0         # left wall
  v[-1, :] = 0         # right wall
  v[: , 0] = -v[:,  1] # bottom wall
  v[: ,-1] = -v[:, -2] # top

  u[0 , :] = -u[ 1, :] # left
  u[-1, :] = -u[-2, :] # right
  u[: , 0] = 0         # bottom
  u[: ,-1] = lid_vel   # lid

  return u, v

def check_divergence_free(imax, jmax, dx, dy, u, v):

  div = np.zeros((imax, jmax), dtype)

  for i in range(imax-1):
    for j in range(jmax-1):
      div[i, j] = (1/dx) * (u[i, j] - u[i+1, j]) + (1/dy) * (v[i, j] - v[i, j+1])

  return div


def get_kinetic_energy(imax, jmax, u, v, dx, dy):

  energy = 0
  for i in range(imax):
    for j in range(jmax):
      energy += 0.5 * dx * dy * (u[i,j]**2 + v[i,j]**2)

  return energy

def find_nnz(a):
  x = a.flatten()
  nnz_idx = np.nonzero(x)
  nnz_val = x[nnz_idx]
  return nnz_idx, nnz_val

if __name__ == "__main__":

  parser = argparse.ArgumentParser(description='')
  parser.add_argument('-o', type=str, default='plot', help='output file prefix')

  opt = parser.parse_args()

  X, Y = np.meshgrid(y, y)

  # initial guess for p_star
  p_star = p.copy()

  iteration, t0 = 0, time.perf_counter()

  residuals, energies = [], []
  plt.ion()

  while iteration < max_iters:

    # SIMPLE loop

    # solve momentum equation for intermediate u_star velocity
    u_star, d_u = compute_u_star(u, v, p, alpha_U, mu, dx, dy)
    v_star, d_v = compute_v_star(u, v, p, alpha_U, mu, dx, dy)

    # calculate rhs vector of the pressure Poisson matrix
    # rhsp should be of size [imax * jmax, 1]
    rhsp = get_rhs(u_star, v_star, rho, dx, dy)

    # Form the Pressure Poisson coefficient matrix
    Ap = get_coeff_mat(imax, jmax, dx, dy, rho, d_u, d_v)

    ## Solve pressure correction implicitly and update pressure
    p, p_prime = pressure_correct(imax, jmax, rhsp, Ap.T, p_star, alpha_P)

    old_u, old_v = u.copy(), v.copy()

    # Update velocity based on pressure correction
    u, v = update_velocity(imax, jmax, u_star, v_star, p_prime, d_u, d_v, lid_velocity)

    # check if velocity field is divergence free
    divergence = check_divergence_free(imax, jmax, dx, dy, u, v)

    # use p as p_star for the next iteration
    p_star = p.copy()

    # find maximum residual in the domain
    residual_u = np.abs(u - old_u)
    residual_v = np.abs(v - old_v)
    max_residual_u = np.max(residual_u)
    max_residual_v = np.max(residual_v)
    max_residual = np.maximum(max_residual_u, max_residual_v)

    # end SIMPLE loop
    t1 = time.perf_counter()

    residuals.append(max_residual)

    # kinetic energy
    energy = get_kinetic_energy(imax, jmax, u, v, dx, dy)
    energies.append(energy)

    # this is for plotting
    #########################
    # if iteration % 10 == 0:

      # fig = plt.figure(figsize=(10, 8), dpi=100)
      # plt.subplot(221)
      # # plotting the pressure field as a contour
      # plt.title('iteration {}, time {:.3f} s'.format(iteration, t1-t0 ))
      # plt.contourf(X, Y, p.T, alpha=0.5, cmap='viridis')
      # plt.contour(X, Y, p.T, cmap='viridis')
      # #plt.streamplot(X, Y, u[:41, :].T, v[:, :41].T)
      # plt.quiver(X[::2, ::2], Y[::2, ::2], u.T[::2, ::2], v.T[::2, ::2])
      # plt.xlabel('X')
      # plt.ylabel('Y')
      #
      # plt.subplot(222)
      # # plotting the pressure field as a contour
      # plt.contourf(X, Y, p.T, alpha=0.5, cmap='viridis')
      # plt.contour(X, Y, p.T, cmap='viridis')
      # plt.streamplot(X, Y, u[:-1, :].T, v[:, :-1].T)
      # #plt.colorbar()
      # plt.xlim((0,1))
      # plt.ylim((0,1))
      # #plt.quiver(X[::2, ::2], Y[::2, ::2], u.T[::2, ::2], v.T[::2, ::2])
      # plt.xlabel('X')
      # plt.ylabel('Y')
      #
      # plt.subplot(223)
      # plt.plot(energies)
      # #plt.yscale('log')
      # plt.grid(linestyle='-', linewidth='0.2', color='gray')
      # plt.grid(which='minor', linestyle=':', linewidth='0.2', color='black')
      # plt.xlabel('iteration')
      # plt.ylabel('kinetic energy')
      # plt.minorticks_on()
      #
      # plt.subplot(224)
      # plt.plot(residuals)
      # plt.yscale('log')
      # plt.grid(linestyle='-', linewidth='0.2', color='gray')
      # plt.grid(which='minor', linestyle=':', linewidth='0.2', color='black')
      # plt.xlabel('iteration')
      # plt.ylabel('max residual')
      # plt.minorticks_on()
      # plt.draw()
      # if max_residual <= tol:
      #   plt.savefig("{:}_final.pdf".format(opt.o), dpi=300)
      #   break
      # if iteration % 100 == 0:
      #   plt.savefig("{:}_{:}.pdf".format(opt.o, iteration), dpi=300)
      # plt.pause(0.0001)
      # plt.clf()

    ##################
    # plotting end

    iteration += 1

    print('i={:4d}, residual={:.6f}, divergence={:.6f}, time elapsed = {:.6f} s'.format(iteration, max_residual, np.linalg.norm(divergence), t1-t0))
