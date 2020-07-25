#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
   Author: Kamil Rocki

   Solve Ax = b
   Numpy implementations of:

   Simple Conjugate Gradient [1]
   Biconjugate Gradient [2]
   Conjugate Gradient Squared [3]
   BiCGStab - Biconjugate gradient stabilized [4]

  [1]: https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf

  [2]: Fletcher, R. (1976). Watson, G. Alistair (ed.). "Conjugate gradient methods for indefinite systems

  [3]: Generalized conjugate gradient squared Diederik R. Fokkema*, Gerard L.G. Sleijpen, Henk A. Van der Vorst

  [4]: Van der Vorst, H. A. (1992). "Bi-CGSTAB: A Fast and Smoothly Converging Variant of Bi-CG for the Solution of Nonsymmetric Linear Systems". SIAM J. Sci. Stat. Comput. 13

"""

import numpy as np
import time

# Ax = b
# x - initial solution
# tol - err tolerance
# max_iter

# vanilla conjugate gradient
def cg(A, b, x, tol=1e-6, max_iter=100):

  r = b - np.dot(A, x) # residual
  err = np.dot(r.T, r) # error
  i, p = 0, r

  while np.linalg.norm(err) > tol:

    Ap = np.dot(A, p)
    pAp = np.dot(p.T, Ap)
    alpha = err / pAp

    x = x + alpha * p
    r = r - alpha * Ap

    new_err = np.dot(r.T, r)
    beta = new_err / err
    p = r + beta * p # direction
    err = new_err
    i += 1
    if max_iter and i > max_iter: break

  return x, err, i

# conjugate gradient squared
def cgs(A, b, x, tol=1e-6, max_iter=100):

  r = b - np.dot(A, x) # residual
  err = np.dot(r.T, r) # error
  i, p, u = 0, r, r
  r0 = r

  while np.linalg.norm(err) > tol:

    Ap = np.dot(A, p)
    pAp = np.dot(r0.T, Ap)
    alpha = err / pAp

    q = u - alpha * Ap
    x = x + alpha * (u + q)
    r = r - alpha * np.dot(A, (u + q))

    new_err = np.dot(r0.T, r)
    beta = new_err / err
    u = r + beta * q
    p = u + beta * (q + beta * p) # direction
    err = new_err
    i += 1
    if max_iter and i > max_iter: break

  return x, err, i

# Biconjugate Gradient Stabilized (BICGSTAB)
def bicgstab(A, b, x, tol=1e-6, max_iter=100):

  r = b - np.dot(A, x) # residual
  err = np.dot(r.T, r) # error
  i, p  = 0, r
  r0 = r

  while np.linalg.norm(r) > tol:

    Ap = np.dot(A, p)
    pAp = np.dot(r0.T, Ap)
    alpha = err / pAp

    s = r - alpha * Ap
    As = np.dot(A, s)
    omega = np.dot(s.T, As) / (np.dot(As.T, As))
    x = x + alpha * p + omega * s
    r = s - omega * As

    new_err = np.dot(r0.T, r)
    beta = (alpha / omega) * (new_err / err)
    p = r + beta * (p - omega * Ap)
    err = new_err
    i += 1
    if max_iter and i > max_iter: break

  return x, err, i

# biconjugate gradient
def bicg(A, b, x, tol=1e-6, max_iter=100):

  r = b - np.dot(A, x)   # residual
  q = b - np.dot(A.T, x) # residual

  err = np.dot(q.T, r) # error
  i, p, s, y = 0, r, q, x

  while np.linalg.norm(err) > tol:

    Ap = np.dot(A, p)
    pAp = np.dot(s.T, Ap)
    alpha = err / pAp

    x = x + alpha * p
    y = y + alpha * s

    r = r - alpha * Ap
    q = q - alpha * np.dot(A.T, s)

    new_err = np.dot(q.T, r)
    beta = new_err / err
    p = r + beta * p # direction
    s = q + beta * s
    err = new_err
    i += 1
    if max_iter and i > max_iter: break

  return x, err, i
