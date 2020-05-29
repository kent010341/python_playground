import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing.pool import ThreadPool
from xlwings.constants import LineStyle

def pso(func, lb, ub, ieqcons=[], f_ieqcons=None, args=(), kwargs={},
        swarmsize=100, maxomega=0.9, minomega=0.4, phip=2, phig=2, maxiter=100,
        minstep=1e-8, minfunc=1e-18, debug=False, GS_ans=None):
    """
    Perform a particle swarm optimization (PSO)

    Parameters
    ==========
    func : function
        The function to be minimized
    lb : array
        The lower bounds of the design variable(s)
    ub : array
        The upper bounds of the design variable(s)

    Optional
    ========
    ieqcons : list
        A list of functions of length n such that ieqcons[j](x,*args) >= 0.0 in
        a successfully optimized problem (Default: [])
    f_ieqcons : function
        Returns a 1-D array in which each element must be greater or equal
        to 0.0 in a successfully optimized problem. If f_ieqcons is specified,
        ieqcons is ignored (Default: None)
    args : tuple
        Additional arguments passed to objective and constraint functions
        (Default: empty tuple)
    kwargs : dict
        Additional keyword arguments passed to objective and constraint
        functions (Default: empty dict)
    swarmsize : int
        The number of particles in the swarm (Default: 100)
    omega : scalar
        Particle velocity scaling factor (Default: 0.5)
    phip : scalar
        Scaling factor to search away from the particle's best known position
        (Default: 0.5)
    phig : scalar
        Scaling factor to search away from the swarm's best known position
        (Default: 0.5)
    maxiter : int
        The maximum number of iterations for the swarm to search (Default: 100)
    minstep : scalar
        The minimum stepsize of swarm's best position before the search
        terminates (Default: 1e-8)
    minfunc : scalar
        The minimum change of swarm's best objective value before the search
        terminates (Default: 1e-8)
    debug : boolean
        If True, progress statements will be displayed every iteration
        (Default: False)

    Returns
    =======
    g : array
        The swarm's best known position (optimal design)
    f : scalar
        The objective value at ``g``

    """

    print("run pso")

    assert len(lb) == len(ub), 'Lower- and upper-bounds must be the same length'
    assert hasattr(func, '__call__'), 'Invalid function handle'
    lb = np.array(lb)
    ub = np.array(ub)
#     assert np.all(ub > lb), 'All upper-bound values must be greater than lower-bound values'

    vhigh = np.abs(ub - lb) * 0.1
    vlow = -vhigh

    # Check for constraint function(s) #########################################
    obj = lambda x: func(x, *args, **kwargs)
    if f_ieqcons is None:
        if not len(ieqcons):
            if debug:
                print('No constraints given.')
            cons = lambda x: np.array([0])
        else:
            if debug:
                print('Converting ieqcons to a single constraint function')
            cons = lambda x: np.array([y(x, *args, **kwargs) for y in ieqcons])
    else:
        if debug:
            print('Single constraint function given in f_ieqcons')
        cons = lambda x: np.array(f_ieqcons(x, *args, **kwargs))

    def is_feasible(x):
        check = np.all(cons(x) >= 0)
        return check

    # Initialize the particle swarm ############################################
    # np.random.seed(5)
    #np.random.seed(rdseed)
    S = swarmsize
    D = len(lb)  # the number of dimensions each particle has
    x = np.random.rand(S, D)  # particle positions
    if isinstance(GS_ans, list) or isinstance(GS_ans, np.ndarray):
        x[np.random.randint(swarmsize)] = np.array(GS_ans)
    v = np.zeros_like(x)  # particle velocities
    p = np.zeros_like(x)  # best particle positions
    fp = np.zeros(S)  # best particle function values
    fxi = np.zeros(S)  # each particle function values
    g = []  # best swarm position
    fg = 1e100  # artificial best swarm position starting value

    for i in range(S):
        # Initialize the particle's position
        x[i, :] = lb + x[i, :] * (ub - lb)

        # Initialize the particle's best known position
        p[i, :] = x[i, :].copy()

        # Calculate the objective's value at the current particle's
        fp[i] = obj(p[i, :])

        # At the start, there may not be any feasible starting point, so just
        # give it a temporary "best" point since it's likely to change
        if i == 0:
            g = p[0, :].copy()

        # If the current particle's position is better than the swarm's,
        # update the best swarm position
        if fp[i] < fg and is_feasible(p[i, :]):
            fg = fp[i]
            g = p[i, :].copy()

        # Initialize the particle's velocity
        v[i, :] = vlow + np.random.rand(D) * (vhigh - vlow)

    # Iterate until termination criterion met ##################################
    it = 1
    omega = maxomega
    vlb = vlow
    vub = vhigh
    # print(vlb, vub)
    fgtotal = []
    gcopy = g.copy()
    
    
    while it <= maxiter:
        rp = np.random.uniform(size=(S, D))
        rg = np.random.uniform(size=(S, D))
        for i in range(S):

            # Update the particle's velocity
            v[i, :] = omega * v[i, :] + phip * rp[i, :] * (p[i, :] - x[i, :]) + \
                      phig * rg[i, :] * (gcopy - x[i, :])

            mark1 = v[i, :] < vlb
            mark2 = v[i, :] > vub
            v[i, mark1] = vlb[mark1]
            v[i, mark2] = vub[mark2]
            # Update the particle's position, correcting lower and upper bound
            # violations, then update the objective function value
            x[i, :] = x[i, :] + v[i, :]

            mark1 = x[i, :] < lb
            mark2 = x[i, :] > ub
            x[i, mark1] = lb[mark1]
            x[i, mark2] = ub[mark2]
            fx = obj(x[i, :])
            fxi[i] = fx

            # Compare particle's best position (if constraints are satisfied)
            if fx < fp[i] and is_feasible(x[i, :]):
                p[i, :] = x[i, :].copy()
                fp[i] = fx

                # Compare swarm's best position to current particle's position
                # (Can only get here if constraints are satisfied)
                if fx < fg:
                    #if debug:
                    #    print('New best for swarm at iteration {:}: {:} {:}'.format(it, x[i, :], fx))

                    tmp = x[i, :].copy()
                    stepsize = np.sqrt(np.sum((g - tmp) ** 2))
                    if np.abs(fg - fx) <= minfunc:
                        print('Stopping search: Swarm best objective change less than {:}'.format(minfunc))
                        return tmp, fx
                    elif stepsize <= minstep:
                        print('Stopping search: Swarm best position change less than {:}'.format(minstep))
                        return tmp, fx
                    else:
                        g = tmp.copy()
                        fg = fx

        if debug:
            print('Best after iteration {:}: {:} {:}'.format(it, g, fg))
       
        fgtotal.append(fg)
        # np.savetxt("output/output_0.1_%s.csv" % it, x, delimiter=",")
        # np.savetxt("output/fitness_0.1_%s.csv" % it, fxi, delimiter=",")
        omega = maxomega - (maxomega - minomega) * it / maxiter
        gcopy = g.copy()
        it += 1

    print('Stopping search: maximum iterations reached --> {:}'.format(maxiter))
   
    if not is_feasible(g):
        print("However, the optimization couldn't find a feasible design. Sorry")
   

    return g, fg, fgtotal



