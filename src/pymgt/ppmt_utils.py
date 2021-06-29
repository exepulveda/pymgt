"""Projection Pursuit Multivariate Transform using moments as projection index
"""

import numpy as np
import scipy.stats
import scipy
from scipy.optimize import differential_evolution

from .utils import spherical2cartesian

def find_next_best_direction(x,
                             index_func,
                             is_maximising_better=True,
                             maxiter=100,
                             popsize=None,
                             trace=False):
    """Find the next best direction by differential evolution optimisation
    """
    _, q = x.shape

    # use DE to minimise the projection_index
    sign = -1 if is_maximising_better else 1

    ndim = q-1 # using spherical coordinates requires 1 less dimension

    bounds = [(0.0, np.pi)]*ndim
    bounds[-1] = (0.0, 2*np.pi)

    if popsize is None:
        popsize = min(int(ndim*np.sqrt(ndim)), 50)
    popsize = max(15, popsize)

    def min_func(a):
        # a contains the angles in spherical system
        direction = spherical2cartesian(a)

        # np.testing.assert_almost_equal(np.linalg.norm(direction), 1.0)

        projection = x@direction
        #print(projection.shape)

        #pdb.set_trace()
        pi = index_func(projection)
        return sign*pi #the negative is because we want to maximise the projection index but this is a minimisation solver

    res = differential_evolution(min_func, bounds, maxiter=maxiter, popsize=popsize, disp=trace)
    #res = differential_evolution(min_func, maxiter=maxiter, popsize=popsize)

    #print(-res.fun,res.x)
    return spherical2cartesian(res.x), sign*res.fun

def legendre_poly(r, degree=10):
    """Compute the Legendre polynomials for the `r` transformed values
    """
    n = len(r)
    p = np.zeros((n, degree+1))

    p[:, 0] = 1.0  # Legendre 0
    p[:, 1] = r # Legendre 1
    for j in range(2, degree+1): #j --> i
        p[:, j] = ((2.0*j - 1.0) * r * p[:, j-1] - (j-1)*p[:, j-2])/j  # Legendre >=2

    return p

def legendre_poly_deriv_projection(poly, r):
    """Compute the Legendre polynomials derivate used in the projection
    """
    ndata = poly.shape[0]
    degree = poly.shape[1] - 1

    poly_deriv = np.zeros((ndata, degree))

    poly_deriv[:, 0] = 1.0  
    for i in range(1, degree):
        poly_deriv[:, i] = (r*poly_deriv[:, i-1] + (i+1)*poly[:, i-1]) #* w(:); !Legendre >=1

    return poly_deriv 

def legendre_poly_gradient(direction, data, projection, poly, r):
    """Compute the Legendre polynomials gradient
    """
    exp1 = np.exp(-projection**2/2.0)

    ndim = len(direction)
    degree = poly.shape[1] - 1

    poly_deriv = legendre_poly_deriv_projection(poly, r)

    gradient = np.zeros(ndim)
    for i in range(ndim):
        for j in range(degree):
            e1 = np.mean(poly[:, j])
            e2 = np.mean(poly_deriv[:, j] * exp1 * (data[:, i]-direction[i]*projection))
            gradient[i] += (2.0*(j+1)+1.0) * e1 * e2

    gradient = gradient*(2.0/np.sqrt(2.0*np.pi))

    return gradient


def legendre_poly_deriv(data, direction, degree=10):
    """Compute the Legendre polynomials derivate
    """
    ndata, ndim = data.shape

    # project
    projection = data@direction

    r = r_transform(projection)

    #print(projection[:10])
    #print(projection[-10:])
    #print(r[:10])
    #print(r[-10:])

    poly = legendre_poly(r, degree=degree)
    poly_deriv = np.zeros((ndata, degree))

    poly_deriv[:, 0] = 1.0

    for i in range(degree):
        poly_deriv[:, i] = (r*poly_deriv[:, i-1] + (i+1)*poly[:, i-1]) #* w(:); !Legendre >=1

    # projection index
    pi = friedman_index_internal(poly)

    # gradient
    exp1 = np.exp(-projection**2/2.0)

    gradient = np.zeros(ndim)
    for i in range(ndim):
        for j in range(degree):
            e1 = np.mean(poly[:, j])
            e2 = np.mean(poly_deriv[:, j] * exp1 * (data[:, i]-direction[i]*projection))
            gradient[i] += (2.0*(j+1)+1.0) * e1 * e2

    gradient = gradient*(2.0/np.sqrt(2.0*np.pi))

    return pi, gradient

def find_next_best_direction_gd(x, degree=10, maxiter=100, trace=False, step=0.01):
    """Find the next best direction by gradient descent
    """
    _, ndim = x.shape

    # first calculate the index to marginals
    if trace: print("calculate the index to marginals")
    best_pi = friedman_index(x[:, 0], degree=degree)
    best_direction = np.zeros(ndim)
    best_direction[0] = 1.0

    if trace: print("new best_direction: 0", best_pi, best_direction)

    for i in range(1, ndim):
        pi = friedman_index(x[:, i], degree=degree)
        if pi > best_pi:
            best_pi = pi
            best_direction.fill(0.0)
            best_direction[i] = 1.0      
            if trace: print("new best_direction:", i, best_pi, best_direction, np.linalg.norm(best_direction))

    if trace:
        print("=======")
        print("Stage 2")
        print("=======")

    # rough optimisation
    e = np.zeros(ndim)
    c1 = 1.0/np.sqrt(2.0)
    for _ in range(10):
        pi = best_pi
        for i in range(ndim):
            e.fill(0.0)
            e[i] = 1.0

            if 1.0 + best_direction[i] > 0.0:
                pupper = c1*(best_direction + e)/np.sqrt(1.0 + best_direction[i])
                fupper = friedman_index(x@pupper, degree=degree)
            else:   
                fupper = None

            if 1.0 - best_direction[i] > 0.0:
                plower = c1*(best_direction - e)/np.sqrt(1.0 - best_direction[i])
                flower = friedman_index(x@plower, degree=degree)
            else:
                flower = None

            if fupper is not None and flower is not None:
                if fupper > flower:
                    f = fupper
                    s = +1
                else:
                    f = flower
                    s = -1
            else:
                continue

            if f > best_pi:
                best_direction = c1*(best_direction+s*e)/np.sqrt(1+s*best_direction[i])
                pi_val = friedman_index(x@best_direction)
                best_pi = f
                if trace:
                    print("best_direction:", best_direction, np.linalg.norm(best_direction))
                    print("best_pi:", best_pi)
                    print("pi_val:", pi_val)


        if np.abs(pi - best_pi) < 1e-10: 
            if trace: print("Not more improvement")
            break

    if trace:        
        print("=======")
        print("Stage 3")
        print("=======")

        print("best_direction:", best_direction, np.linalg.norm(best_direction))

    # gradient decend
    alpha = step
    direction = np.copy(best_direction)
    for i in range(maxiter):
        projection = x@direction
        pi, poly, r = friedman_index(projection, return_full=True)
        gradient = legendre_poly_gradient(direction, x, projection, poly, r)

        new_direction = direction + alpha*gradient
        new_direction /= np.linalg.norm(new_direction)

        new_pi = friedman_index(x@new_direction)

        if trace: 
            print("at:", i, pi, new_pi, best_pi) #, new_direction, gradient)

        if new_pi > best_pi:
            best_pi = new_pi
            best_direction = new_direction
            direction = np.copy(new_direction)
            #if trace: print("new best:: ", best_pi, " dir:: ", best_direction, np.linalg.norm(best_direction))
        elif np.abs(new_pi - best_pi) < 1e-10:
            if trace: print("Not more improvement")
            break
        else:
            if trace: print("Not more improvement")
            break


    best_direction /= np.linalg.norm(best_direction)
    best_pi = friedman_index(x@best_direction)

    return best_direction, best_pi

def r_transform(x):
    """Compute the r transform for the Friedman index
    """
    return 2.0*scipy.stats.norm.cdf(x)-1.0

def friedman_index(x, degree=10, return_full=False):
    """Compute the Friedman index
    """
    r = r_transform(x)

    #pdb.set_trace()

    p = legendre_poly(r, degree=degree)
    if return_full:
        return friedman_index_internal(p), p, r

    return friedman_index_internal(p)

def friedman_index_internal(p):
    """Compute the Friedman index given the polynomial expansion `p`
    """
    _, degree = p.shape
    e2 = np.mean(p[:, 1:], axis=0)**2
    idx = 0.0
    for i in range(1, degree):
        idx += (2.0*i + 1.0)*e2[i-1]/2.0

    return idx
