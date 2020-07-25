#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
   Author: Kamil Rocki
   Date: 5/18/2020

   Lid Driven Cavity 3D

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
from matplotlib.animation import FuncAnimation

from cg import cg, cgs, bicg, bicgstab
matplotlib.use('Agg')

#%matplotlib inline
np.set_printoptions(precision=3, suppress=True, linewidth=100)

dtype = 'float64'

### parameters ###

max_iters = 1000
lid_velocity = 1

imax=5           # grid size in x-direction
jmax=5           # grid size in y-direction
kmax=5           # grid size in z-direction
mu = 0.1         # viscosity
rho = 1          # density
alpha_U = 0.7    # velocity under-relaxation
alpha_P = 0.1    # pressure under-relaxation
tol = 1e-5       # max residual

# derived grid cell sizes
dx = 1/(imax-1)
dy = 1/(jmax-1)
dz = 1/(kmax-1)

# set up grid
x = np.arange(dx/2, 1+dx/2, dx)
y = np.arange(0,    1+dy,   dy)
z = np.arange(0,    1+dz,   dz)

# set up arrays
p = np.zeros((kmax,   imax,   jmax  ), dtype) # pressure
u = np.zeros((kmax,   imax+1, jmax  ), dtype) # horizontal velocity
v = np.zeros((kmax,   imax,   jmax+1), dtype) # vertical velocity
w = np.zeros((kmax+1, imax,   jmax  ), dtype) # w velocity

# boundary condition
u[:, :, -1] = lid_velocity

### code for momentum
# convective coefficients
Dw = mu * (dz * dy)/dx
De = mu * (dz * dy)/dx
Ds = mu * (dz * dx)/dy
Dn = mu * (dz * dx)/dy
Df = mu * (dx * dy)/dz
Db = mu * (dx * dy)/dz

# 5.86 - power law scheme
def coeff(F, D): return np.maximum(0, (1 - 0.1 * np.abs(F/D))**5)

def compute_u_star(u, v, w, p, alpha, mu, dx, dy, dz):

  u_star = np.zeros_like(u)
  d_u = np.zeros_like(u) # correction coefficient

  for k in range(0, kmax):
    for i in range(1, imax):
      for j in range(0, jmax):

        Fw = .5 * rho * dz * dy * (u[k,  i,   j  ] + u[k,  i-1, j  ]) if i>0      else 0
        Fe = .5 * rho * dz * dy * (u[k,  i,   j  ] + u[k,  i+1, j  ]) if i<imax   else 0
        Fs = .5 * rho * dz * dx * (v[k,  i  , j  ] + v[k,  i-1, j  ]) if j>0      else 0
        Fn = .5 * rho * dz * dx * (v[k,  i  , j+1] + v[k,  i-1, j+1]) if j<jmax-1 else 0
        Ff = .5 * rho * dx * dy * (w[k,  i  , j  ] + w[k,  i-1, j  ]) if k>0      else 0
        Fb = .5 * rho * dx * dy * (w[k+1,i  , j  ] + w[k+1,i-1, j  ]) if k<kmax-1 else 0

        # power-law differencing
        aW = Dw * coeff(Fw, Dw) + np.maximum( Fw, 0) if i>0      else 0
        aE = De * coeff(Fe, De) + np.maximum(-Fe, 0) if i<imax   else 0
        aS = Ds * coeff(Fs, Ds) + np.maximum( Fs, 0) if j>0      else 0
        aN = Dn * coeff(Fn, Dn) + np.maximum(-Fn, 0) if j<jmax-1 else 0
        aF = Df * coeff(Ff, Df) + np.maximum( Ff, 0) if k>0      else 0
        aB = Db * coeff(Fb, Db) + np.maximum(-Fb, 0) if k<kmax-1 else 0

        aP = aF + aB + aE + aW + aN + aS + (Fb-Ff) + (Fe-Fw) + (Fn-Fs)

        pressure_term = dz * dy * (p[k, i-1, j] - p[k, i, j])

        # only do inner cells
        if k>0 and k<kmax-1 and j>0 and j<jmax-1 and i>0 and i<imax:
          u_star[k, i, j] = alpha/aP * (
                        pressure_term +
                        aW * u[k,  i-1, j  ] +
                        aE * u[k,  i+1, j  ] +
                        aS * u[k,  i  , j-1] +
                        aN * u[k,  i  , j+1] +
                        aF * u[k-1,i  , j  ] +
                        aB * u[k+1,i  , j  ]
                    ) + (1 - alpha) * u[k, i, j]

        d_u[k, i, j] = alpha * (dz * dy) / aP

    # boundary conditions for outer cells
    u_star[0,  :, :] = 0 # front
    u_star[-1, :, :] = 0 # back
    u_star[:, 0,  :] = -u_star[:,  1, :] # left wall
    u_star[:, -1, :] = -u_star[:, -2, :] # right wall
    u_star[:, :,  0] = 0              # bottom wall
    u_star[:, :, -1] = lid_velocity   # top (no-wall)

  return u_star, d_u

# this is almost the same as compute_u_star
def compute_v_star(u, v, w, p, alpha, mu, dx, dy, dz):

  v_star = np.zeros_like(v)
  d_v = np.zeros_like(v) # correction coefficient

  for k in range(0, kmax):
    for i in range(0, imax):
      for j in range(1, jmax):

        Fw = .5 * rho * dz * dy * (u[k  , i  , j  ] + u[k,  i  , j-1]) if i>0      else 0
        Fe = .5 * rho * dz * dy * (u[k  , i+1, j  ] + u[k,  i+1, j-1]) if i<imax-1 else 0
        Fs = .5 * rho * dz * dx * (v[k  , i  , j  ] + v[k,  i  , j-1]) if j>0      else 0
        Fn = .5 * rho * dz * dx * (v[k  , i  , j  ] + v[k,  i  , j+1]) if j<jmax   else 0
        Ff = .5 * rho * dx * dy * (w[k  , i  , j  ] + w[k,  i  , j-1]) if k>0      else 0
        Fb = .5 * rho * dz * dy * (w[k+1, i  , j  ] + w[k+1,i  , j-1]) if k<kmax-1 else 0

        # power-law differencing
        aW = Dw * coeff(Fw, Dw) + np.maximum( Fw, 0) if i>0          else 0
        aE = De * coeff(Fe, De) + np.maximum(-Fe, 0) if i<imax-1     else 0
        aS = Ds * coeff(Fs, Ds) + np.maximum( Fs, 0) if j>0          else 0
        aN = Dn * coeff(Fn, Dn) + np.maximum(-Fn, 0) if j<jmax       else 0
        aF = Df * coeff(Ff, Df) + np.maximum( Ff, 0) if k>0          else 0
        aB = Db * coeff(Fb, Db) + np.maximum(-Fb, 0) if k<kmax-1     else 0

        aP = aF + aB + aE + aW + aN + aS + (Fb-Ff) + (Fe-Fw) + (Fn-Fs)

        pressure_term = dz * dx * (p[k, i, j-1] - p[k, i, j])

        # only do inner cells
        if k>0 and k<kmax-1 and j>0 and j<jmax and i>0 and i<imax-1:
          v_star[k, i, j] = alpha/aP * (
                            pressure_term +
                            aW * v[k  , i-1, j  ] +
                            aE * v[k  , i+1, j  ] +
                            aS * v[k  , i  , j-1] +
                            aN * v[k  , i  , j+1] +
                            aF * v[k-1, i  , j  ] +
                            aB * v[k+1, i  , j  ]
                        ) + (1 - alpha) * v[k, i, j]

        d_v[k, i, j] = alpha * (dz * dx) / aP

    # boundary conditions
    v_star[0,  :, :] = 0 # front
    v_star[-1, :, :] = 0 # back
    v_star[:,  0, :] = 0 # left wall
    v_star[:, -1, :] = 0 # right wall
    v_star[:,  :, 0] = -v_star[:, :, 1] # bottom
    v_star[:,  :,-1] = -v_star[:, :,-2] # lid

  return v_star, d_v

def compute_w_star(u, v, w, p, alpha, mu, dx, dy, dz):

  w_star = np.zeros_like(w)
  d_w = np.zeros_like(w) # correction coefficient

  for k in range(1, kmax):
    for i in range(0, imax):
      for j in range(0, jmax):

        Fw = .5 * rho * dz * dy * (u[k  , i  , j  ] + u[k-1,i  , j  ]) if i>0      else 0
        Fe = .5 * rho * dz * dy * (u[k  , i+1, j  ] + u[k-1,i+1, j  ]) if i<imax-1 else 0
        Fs = .5 * rho * dz * dx * (v[k  , i  , j  ] + v[k-1,i  , j  ]) if j>0      else 0
        Fn = .5 * rho * dz * dx * (v[k  , i  , j+1] + v[k-1,i  , j+1]) if j<jmax   else 0
        Ff = .5 * rho * dx * dy * (w[k  , i  , j  ] + w[k-1,i  , j  ]) if k>0      else 0
        Fb = .5 * rho * dz * dy * (w[k  , i  , j  ] + w[k+1,i  , j  ]) if k<kmax-1 else 0

        # power-law differencing
        aW = Dw * coeff(Fw, Dw) + np.maximum( Fw, 0) if i>0          else 0
        aE = De * coeff(Fe, De) + np.maximum(-Fe, 0) if i<imax-1     else 0
        aS = Ds * coeff(Fs, Ds) + np.maximum( Fs, 0) if j>0          else 0
        aN = Dn * coeff(Fn, Dn) + np.maximum(-Fn, 0) if j<jmax       else 0
        aF = Df * coeff(Ff, Df) + np.maximum( Ff, 0) if k>0          else 0
        aB = Db * coeff(Fb, Db) + np.maximum(-Fb, 0) if k<kmax-1     else 0

        aP = aF + aB + aE + aW + aN + aS + (Fb-Ff) + (Fe-Fw) + (Fn-Fs)

        pressure_term = dy * dx * (p[k-1, i, j] - p[k, i, j])

        # only do inner cells
        if k>0 and k<kmax-1 and j>0 and j<jmax-1 and i>0 and i<imax-1:
          w_star[k, i, j] = alpha/aP * (
                            pressure_term +
                            aW * w[k  , i-1, j  ] +
                            aE * w[k  , i+1, j  ] +
                            aS * w[k  , i  , j-1] +
                            aN * w[k  , i  , j+1] +
                            aF * w[k-1, i  , j  ] +
                            aB * w[k+1, i  , j  ]
                        ) + (1 - alpha) * w[k, i, j]

        d_w[k, i, j] = alpha * (dy * dx) / aP

    # boundary conditions
    w_star[:, 0 , :] = 0 # left wall
    w_star[:, -1, :] = 0 # right wall
    w_star[-1,:, :] = -w_star[-2,:,:] # back wall
    w_star[0, :, :] = -w_star[ 1,:,:] # front wall
    w_star[:,  :, 0] = 0 # bottom
    w_star[:,  :,-1] = 0 # lid

  return w_star, d_w

### code for pressure
# to calculate rhs vector of the pressure Poisson matrix
def get_rhs(u_star, v_star, w_star, rho, dx, dy, dz):

  nx = u_star.shape[2]
  ny = v_star.shape[1]
  nz = u_star.shape[0]

  n = nz * nx * ny

  # vector of RHS for solving pressure corrections
  bp = np.zeros((n, 1), dtype)

  for k in range(0, nz):
    for j in range(0, ny):
      for i in range(0, nx):
        idx = k * ny * nx + j + i * ny
        bp[idx, 0] = rho * (
          (u_star[k, i, j] - u_star[k, i+1, j]) * dz * dy +
          (v_star[k, i, j] - v_star[k, i, j+1]) * dz * dx +
          (w_star[k, i, j] - w_star[k+1, i, j]) * dx * dy )

  # p(0,0) is set to be zero, it has no pressure correction
  bp[0,0] = 0

  return bp

# build the A matrix
def get_coeff_mat(imax, jmax, kmax, dx, dy, dz, rho, d_u, d_v, d_w):

  N = imax * jmax * kmax

  Ap = np.zeros((N, N), dtype)

  for k in range(kmax):
    for j in range(jmax):
      for i in range(imax):

        pos = k * (jmax * imax) + i + j * (jmax)

        aE, aW, aN, aS, aF, aB = 0, 0, 0, 0, 0, 0

        if k==0 and i==0 and j==0:
          Ap[pos, pos] = 1 # no correction
        else:
          if i>0:
            Ap[pos, pos-1] = -rho * d_u[k, i  , j  ] * dz * dy # sub diagonal
            aW = -Ap[pos, pos-1]

          if i<imax-1:
            Ap[pos, pos+1] = -rho * d_u[k, i+1, j  ] * dz * dy # upper diagonal
            aE = -Ap[pos, pos+1]

          if j>0:
            Ap[pos, pos-jmax] = -rho * d_v[k, i  , j  ] * dz * dx # sub-sub diagonal
            aS = -Ap[pos, pos-jmax]

          if j<jmax-1:
            Ap[pos, pos+jmax] = -rho * d_v[k, i  , j+1] * dz * dx # upper-upper diagonal
            aN = -Ap[pos, pos+jmax]

          if k>0:
            Ap[pos, pos-(jmax*kmax)] = -rho * d_w[k, i  , j  ] * dy * dx # sub-sub diagonal
            aF = -Ap[pos, pos-(kmax*jmax)]

          if k<kmax-1:
            Ap[pos, pos+(jmax*kmax)] = -rho * d_w[k+1, i  , j] * dy * dx # upper-upper diagonal
            aB = -Ap[pos, pos+(jmax*kmax)]

          # main diagonal
          aP = aE + aN + aW + aS + aF + aB
          Ap[pos, pos] = aP

  return Ap

def lin_solve(A, b): return bicgstab(A, b, np.zeros_like(b))[0]

def pressure_correct(imax, jmax, kmax, rhsp, Ap, p, alpha):

  pressure = np.zeros_like(p)
  p_prime_interior = lin_solve(Ap, rhsp);

  p_prime = np.reshape(p_prime_interior, (kmax, imax, jmax))

  pressure = p + alpha * p_prime
  pressure[0, 0] = 0

  return pressure, p_prime

def update_velocity(imax, jmax, kmax, u_star, v_star, w_star, p_prime, d_u, d_v, d_w, lid_vel):

  u = np.zeros_like(u_star)
  v = np.zeros_like(v_star)
  w = np.zeros_like(w_star)

  for k in range(1,kmax):
    for i in range(1, imax):
      for j in range(1, jmax):
        u[k, i, j] = u_star[k, i, j] + d_u[k, i, j] * (p_prime[k, i-1, j] - p_prime[k, i, j])

    for i in range(1, imax):
      for j in range(1, jmax):
        v[k, i, j] = v_star[k, i, j] + d_v[k, i, j] * (p_prime[k, i, j-1] - p_prime[k, i, j])

    for i in range(1, imax):
      for j in range(1, jmax):
        w[k, i, j] = w_star[k, i, j] + d_w[k, i, j] * (p_prime[k-1, i, j] - p_prime[k, i, j])

  # update boundary conditions
  v[-1,: , :] = 0 # back wall
  v[0, : , :] = 0 # front wall
  v[:, 0 , :] = 0 # left wall
  v[:, -1, :] = 0 # right wall
  v[:, : , 0] = -v[:, :, 1] # bottom wall
  v[:, : ,-1] = -v[:, :, -2] # top

  u[:, 0 , :] = -u[:,  1, :] # left
  u[:, -1, :] = -u[:, -2, :] # right
  u[:, : , 0] = 0 # bottom
  u[:, : ,-1] = lid_vel # lid
  u[-1,: , :] = 0 # back wall
  u[0, : , :] = 0 # front wall

  w[:, 0 , :] = 0 # left wall
  w[:, -1, :] = 0 # right wall
  w[-1,: , :] = -w[-2,:,:] # back wall
  w[0, : , :] = -w[ 1,:,:] # front wall
  w[:,  :, 0] = 0 # bottom
  w[:,  :,-1] = 0 # lid

  return u, v, w

def check_divergence_free(imax, jmax, kmax, dx, dy, dz, u, v, w):

  div = np.zeros((kmax, imax, jmax), dtype)

  for k in range(kmax):
    for i in range(imax-1):
      for j in range(jmax-1):
        div[k, i, j] = (1/dx) * (u[k, i, j] - u[k, i+1, j]) + (1/dy) * (v[k, i, j] - v[k, i, j+1])

  return div


def get_kinetic_energy(imax, jmax, kmax, u, v, w, dx, dy, dz):

  energy = 0

  for k in range(kmax):
    for i in range(imax):
      for j in range(jmax):
        energy += 0.5 * dx * dy * (u[k,i,j]**2 + v[k,i,j]**2)

  return energy

if __name__ == "__main__":

  parser = argparse.ArgumentParser(description='')
  parser.add_argument('-o', type=str, default='plot', help='output file prefix')

  opt = parser.parse_args()

  X, Y, Z = np.meshgrid(y, y, y)

  # initial guess for p_star
  p_star = p.copy()

  iteration, t0 = 0, time.perf_counter()

  residuals, energies, div_norms = [], [], []
  #plt.ion()

  #fig = plt.figure(figsize=(10, 8), dpi=100)
  #ax = plt.gca(projection='3d')

  while iteration < max_iters:

    # SIMPLE loop

    # solve momentum equation for intermediate u_star velocity
    u_star, d_u = compute_u_star(u, v, w, p, alpha_U, mu, dx, dy, dz)
    v_star, d_v = compute_v_star(u, v, w, p, alpha_U, mu, dx, dy, dz)
    w_star, d_w = compute_w_star(u, v, w, p, alpha_U, mu, dx, dy, dz)

    # calculate rhs vector of the pressure Poisson matrix
    # rhsp should be of size [imax * jmax, 1]
    rhsp = get_rhs(u_star, v_star, w_star, rho, dx, dy, dz)

    # Form the Pressure Poisson coefficient matrix
    Ap = get_coeff_mat(imax, jmax, kmax, dx, dy, dz, rho, d_u, d_v, d_w)

    ### Solve pressure correction implicitly and update pressure
    p, p_prime = pressure_correct(imax, jmax, kmax, rhsp, Ap.T, p_star, alpha_P)

    old_u, old_v, old_w = u.copy(), v.copy(), w.copy()

    ## Update velocity based on pressure correction
    u, v, w = update_velocity(imax, jmax, kmax, u_star, v_star, w_star, p_prime, d_u, d_v, d_w, lid_velocity)

    # check if velocity field is divergence free
    divergence = check_divergence_free(imax, jmax, kmax, dx, dy, dz, u, v, w)

    # use p as p_star for the next iteration
    p_star = p.copy()

    # find maximum residual in the domain
    residual_u = np.abs(u - old_u)
    residual_v = np.abs(v - old_v)
    residual_w = np.abs(w - old_w)

    div_norm = np.linalg.norm(divergence)
    max_residual = np.max([np.max(residual_u), np.max(residual_v), np.max(residual_w)])

    # end SIMPLE loop
    t1 = time.perf_counter()

    residuals.append(max_residual)

    ## kinetic energy
    energy = get_kinetic_energy(imax, jmax, kmax, u, v, w, dx, dy, dz)
    energies.append(energy)

    div_norms.append(div_norm)

    # this is for plotting
    #########################
    #if iteration % 10 == 0:

    #  # plotting the pressure field as a contour
    #  fig = plt.figure(figsize=(10, 8), dpi=100)
    #  ax = plt.gca(projection='3d')
    #  ax.quiver(X[:,:,:-1], Y[:,:,:-1], Z[:,:,:-1], u[:,:-1,:-1], v[:,:,:-2], w[:-1,:,:-1], color='g', linewidth=0.4, length=0.3)
    #  plt.title('i={:4d}, residual={:.6f}, divergence={:.6f}, time elapsed = {:.6f} s'.format(iteration, max_residual, np.linalg.norm(divergence), t1-t0))
    #  if max_residual <= tol:
    #    plt.savefig("{:}_final.pdf".format(opt.o), dpi=300)
    #    break
    #  if iteration % 10 == 0:
    #    plt.savefig("{:}_{:}.pdf".format(opt.o, iteration), dpi=300)
    #  plt.pause(0.0001)
    #  plt.clf()

    ##################
    # plotting end

    iteration += 1

    print('i={:4d}, residual={:.6f}, divergence={:.6f}, time elapsed = {:.6f} s'.format(iteration, max_residual, np.linalg.norm(divergence), t1-t0))
