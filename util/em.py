import numpy as np
import datetime
from scipy.stats import uniform, exponnorm, norm, lognorm, gennorm, halfnorm, expon, rv_continuous
from scipy.optimize import fmin, fmin_bfgs, fmin_powell 
from scipy import optimize
import scipy.stats._continuous_distns as scd
from scipy.stats._distn_infrastructure import (argsreduce)
from scipy.stats._constants import _XMAX
import time
import random

STOP_THRESHOLD = 1e-4


def get_theta_from_peak(component, peak=None):
    name = component['name']
    if name == 'reactive':
        scale = random.uniform(0.05, 0.1)
        K = random.uniform(0.1, 2)
        shift = np.sqrt(K) * scale/np.sqrt(scale)
        loc = peak - shift
        return (K, loc, scale)
    elif name == 'anticipatory':
        loc = peak
        scale = random.uniform(0.05, 0.1)
        return (loc, scale)



def get_peak_value(data):
    hist = np.histogram(data, bins=np.arange(-2.5, 2.5, (1/60)))
    max_idx = hist[0].argmax()
    interval = (hist[1][max_idx], hist[1][max_idx+1])
    return (interval[0] + interval[1])/2



def set_starting_values(X, components):
    likelihoods = np.zeros((len(X), len(components)))
        
    for i, c in enumerate(components):
        name, pdf, theta = c['name'], c['pdf'], c['theta']
        if name == 'reactive':
            peak = get_peak_value(X[X>0])
            c['theta'] = get_theta_from_peak(c, peak)
            likelihoods[:,i] = np.asarray(pdf(X, K=theta[0], loc=theta[1], scale=theta[2]))
        elif name == 'anticipatory':
            peak = get_peak_value(X[X<0])
            c['theta'] = get_theta_from_peak(c, peak)
            likelihoods[:,i] = np.asarray(pdf(X))
        elif name == 'irrelevant':
            likelihoods[:,i] = np.asarray(irrelevant_gen(c['data']).pdf(X))
    
    return components



def execute_em(X, components):
    """Update responsibility of observed data.
    """

    # initialize weight
    t0 = time.time()
    weights = np.ones((len(components))) / len(components)
    params = None
    prev_loglikelihood = 0

    result = {
        'weights': weights,
        'params': params,
        'llh': 0,
        'itr_num' : 0
    }

    step = 0
    while(1):

        # Expectation Step
        # Calculate Responsibility
        likelihoods = np.zeros((len(X), len(components)))
        responsibilties = np.zeros((len(X), len(components)))

        for i, c in enumerate(components):
            name, pdf, theta = c['name'], c['pdf'], c['theta']
            if name == 'reactive':
                likelihoods[:,i] = np.asarray(pdf(X, K=theta[0], loc=theta[1], scale=theta[2]))
            elif name == 'anticipatory':
                likelihoods[:,i] = np.asarray(pdf(X, loc=theta[0], scale=theta[1]))
            elif name == 'irrelevant':
                likelihoods[:,i] = np.asarray(irrelevant_gen(c['data']).pdf(X))
        
        # Prevent zero-value to avoid numerical errors
        likelihoods[likelihoods<1e-15] = 1e-15
        weights[weights<1e-15] = 1e-15
        responsibilties = ((likelihoods * weights).T / (np.sum(likelihoods * weights, axis=1))).T

        # Maximization Step
        # Find MAP
        for i, c in enumerate(components):
            if c['name'] == 'irrelevant':
                c['theta'] = c['theta']
            else:
                c['theta'] = c['map'](X, responsibilties[:,i])

            # When the weights are calculated, we exclude the nan and inf value
            _responsibilities = responsibilties[:,i]
            _responsibilities = _responsibilities[(~np.isnan(_responsibilities) & ~np.isinf(_responsibilities))]
            weights[i] = np.mean(_responsibilities)

        params = list(components[0]['theta']) + list(components[1]['theta']) + list(components[2]['theta'])# + list(components[3]['theta'])
        params = np.array(params)
        
        # stopping rule
        llh = np.log(likelihoods*responsibilties)
        loglikelihood = llh[(~np.isnan(llh) & ~np.isinf(llh))].sum()
        diff = np.abs(loglikelihood - prev_loglikelihood)
        prev_loglikelihood = loglikelihood
        
        if (diff < STOP_THRESHOLD) | (step >= 250):
            result['llh'] = loglikelihood
            result['itr_num'] = step
            params = list(components[0]['theta']) + list(components[1]['theta']) + list(components[2]['theta'])
            result['params'] = np.array(params)
            break

        step += 1
    
    result['time'] = time.time() - t0
    return result



def tuple2dict(data, keys):
    dic = {}
    for i, value in enumerate(data):
        dic[keys[i]] = value
    return dic



class Reactive(scd.exponnorm_gen):

    def _prior(self, *args):
        '''
        Calculate a prior probability
        '''
        loc = args[0]
        scale = args[1]
        K = args[2]

        # calculate priors
        p_loc = norm.pdf(loc, loc=1, scale=0.5)
        p_scale = norm.pdf(scale, loc=0.1, scale=0.1)
        
        return np.e * p_scale * p_loc

    def fit_map(self, data, responsibilities=None, *args, **kwds):
        # copy from rv.continous_gen
        Narg = len(args)
        if Narg > self.numargs:
            raise TypeError("Too many input arguments.")

        if not np.isfinite(data).all():
            raise RuntimeError("The data contains non-finite values.")

        if responsibilities is None:
            responsibilities = np.ones(data.shape)

        # initialize
        start = [None]*2
        if (Narg < self.numargs) or not ('loc' in kwds and
                                         'scale' in kwds):
            # get distribution specific starting locations
            start = self._fitstart(data)
            args += start[Narg:-2]
        loc = kwds.pop('loc', start[-2])
        scale = kwds.pop('scale', start[-1])
        args += (loc, scale)
        x0, func, restore, args = self._reduce_func_map(args, kwds)

        optimizer = kwds.pop('optimizer', optimize.fmin)
        # convert string to function in scipy.optimize
        if not callable(optimizer) and isinstance(optimizer, str):
            if not optimizer.startswith('fmin_'):
                optimizer = "fmin_"+optimizer
            if optimizer == 'fmin_':
                optimizer = 'fmin'
            try:
                optimizer = getattr(optimize, optimizer)
            except AttributeError:
                raise ValueError("%s is not a valid optimizer" % optimizer)

        # by now kwds must be empty, since everybody took what they needed
        if kwds:
            raise TypeError("Unknown arguments: %s." % kwds)

        vals = optimizer(func, x0, args=(np.ravel(data),responsibilities), disp=0)
        if restore is not None:
            vals = restore(args, vals)
        vals = tuple(vals)
        return vals

    def _reduce_func_map(self, args, kwds):
        """
        Return the (possibly reduced) function to optimize in order to find MLE
        estimates for the .fit method.
        """
        # Convert fixed shape parameters to the standard numeric form: e.g. for
        # stats.beta, shapes='a, b'. To fix `a`, the caller can give a value
        # for `f0`, `fa` or 'fix_a'.  The following converts the latter two
        # into the first (numeric) form.
        if self.shapes:
            shapes = self.shapes.replace(',', ' ').split()
            for j, s in enumerate(shapes):
                val = kwds.pop('f' + s, None) or kwds.pop('fix_' + s, None)
                if val is not None:
                    key = 'f%d' % j
                    if key in kwds:
                        raise ValueError("Duplicate entry for %s." % key)
                    else:
                        kwds[key] = val

        args = list(args)
        Nargs = len(args)
        fixedn = []
        names = ['f%d' % n for n in range(Nargs - 2)] + ['floc', 'fscale']
        x0 = []
        for n, key in enumerate(names):
            if key in kwds:
                fixedn.append(n)
                args[n] = kwds.pop(key)
            else:
                x0.append(args[n])

        if len(fixedn) == 0:
            func = self._penalized_nnlf_map
            restore = None
        else:
            if len(fixedn) == Nargs:
                raise ValueError(
                    "All parameters fixed. There is nothing to optimize.")

            def restore(args, theta):
                # Replace with theta for all numbers not in fixedn
                # This allows the non-fixed values to vary, but
                #  we still call self.nnlf with all parameters.
                i = 0
                for n in range(Nargs):
                    if n not in fixedn:
                        args[n] = theta[i]
                        i += 1
                return args

            def func(theta, x, responsibilities):
                newtheta = restore(args[:], theta)
                return self._penalized_nnlf_map(newtheta, x, responsibilities)

        return x0, func, restore, args

    def _penalized_nnlf_map(self, theta, x, responsibilities):
        ''' Return penalized negative loglikelihood function,
        i.e., - sum (log pdf(x, theta), axis=0) + penalty
           where theta are the parameters (including loc and scale)
        '''
        loc, scale, args = self._unpack_loc_scale(theta)
        if not self._argcheck(*args) or scale <= 0:
            return np.inf
        x = np.asarray((x-loc) / scale)
        n_log_scale = np.sum(responsibilities) * np.log(scale) # the number of membership * np.log(scale)
        outcome = (self._nnlf_and_penalty(x, responsibilities, args) + n_log_scale)/np.sum(responsibilities) + \
             (-np.log(self._prior(loc, scale, args)))

        return outcome

    def fit(self, data, responsibilities=None, *args, **kwds):
        # copy from rv.continous_gen
        Narg = len(args)
        if Narg > self.numargs:
            raise TypeError("Too many input arguments.")

        if not np.isfinite(data).all():
            raise RuntimeError("The data contains non-finite values.")

        if responsibilities is None:
            responsibilities = np.ones(data.shape)

        # initialize
        start = [None]*2
        if (Narg < self.numargs) or not ('loc' in kwds and
                                         'scale' in kwds):
            # get distribution specific starting locations
            start = self._fitstart(data)
            args += start[Narg:-2]
        loc = kwds.pop('loc', start[-2])
        scale = kwds.pop('scale', start[-1])
        args += (loc, scale)
        x0, func, restore, args = self._reduce_func(args, kwds)

        optimizer = kwds.pop('optimizer', optimize.fmin)
        # convert string to function in scipy.optimize
        if not callable(optimizer) and isinstance(optimizer, str):
            if not optimizer.startswith('fmin_'):
                optimizer = "fmin_"+optimizer
            if optimizer == 'fmin_':
                optimizer = 'fmin'
            try:
                optimizer = getattr(optimize, optimizer)
            except AttributeError:
                raise ValueError("%s is not a valid optimizer" % optimizer)

        # by now kwds must be empty, since everybody took what they needed
        if kwds:
            raise TypeError("Unknown arguments: %s." % kwds)

        vals = optimizer(func, x0, args=(np.ravel(data),responsibilities), disp=0)
        if restore is not None:
            vals = restore(args, vals)
        vals = tuple(vals)
        return vals

    def _nnlf_and_penalty(self, x, responsibilities, args):
        cond0 = ~self._support_mask(x, *args)
        n_bad = np.count_nonzero(cond0, axis=0)
        if n_bad > 0:
            x = argsreduce(~cond0, x)[0]
            responsibilities = argsreduce(~cond0, responsibilities)[0]
        logpdf = self._logpdf(x, *args) * responsibilities
        finite_logpdf = np.isfinite(logpdf)
        n_bad += np.sum(~finite_logpdf, axis=0)
        if n_bad > 0:
            penalty = n_bad * np.log(_XMAX) * 100
            return -np.sum(logpdf[finite_logpdf], axis=0) + penalty
        return -np.sum(logpdf, axis=0)

    def _penalized_nnlf(self, theta, x, responsibilities):
        ''' Return penalized negative loglikelihood function,
        i.e., - sum (log pdf(x, theta), axis=0) + penalty
           where theta are the parameters (including loc and scale)
        '''
        loc, scale, args = self._unpack_loc_scale(theta)
        if not self._argcheck(*args) or scale <= 0:
            return np.inf
        x = np.asarray((x-loc) / scale)
        n_log_scale = np.sum(responsibilities) * np.log(scale) # the number of membership * np.log(scale)
        return self._nnlf_and_penalty(x, responsibilities, args) + n_log_scale



class Proactive(scd.norm_gen):

    def _prior(self, *args):
        '''
        Calculate a prior probability
        '''

        loc = args[0]
        scale = args[1]
        
        loc_mu, loc_sigma = -0.05, 1 #0.05
        # scale_s, scale_mu, scale_sigma = 1, 0, 0.35
        # scale_beta, scale_mu, scale_sigma = 10, 2.5, 2.5
        # scale_loc, scale_scale = 0, 1 #r36
        scale_s, scale_loc, scale_scale = 2, 0, 0.75 #r36

        p_mu = norm.pdf(loc, loc=loc_mu, scale=loc_sigma)
        # p_sigma = lognorm.pdf(scale, s=scale_s, loc=scale_mu, scale=scale_sigma)
        p_sigma = norm.pdf(scale, loc=0.1, scale=0.1)
        # p_sigma = lognorm.pdf(scale, s=scale_s, loc=scale_loc, scale=scale_scale)


        return np.e * p_sigma * p_mu

    def fit_map(self, data, responsibilities=None, *args, **kwds):
        # copy from rv.continous_gen
        Narg = len(args)
        if Narg > self.numargs:
            raise TypeError("Too many input arguments.")

        if not np.isfinite(data).all():
            raise RuntimeError("The data contains non-finite values.")

        if responsibilities is None:
            responsibilities = np.ones(data.shape)

        # initialize
        start = [None]*2
        if (Narg < self.numargs) or not ('loc' in kwds and
                                         'scale' in kwds):
            # get distribution specific starting locations
            start = self._fitstart(data)
            args += start[Narg:-2]
        loc = kwds.pop('loc', start[-2])
        scale = kwds.pop('scale', start[-1])
        args += (loc, scale)
        x0, func, restore, args = self._reduce_func_map(args, kwds)

        optimizer = kwds.pop('optimizer', optimize.fmin)
        # convert string to function in scipy.optimize
        if not callable(optimizer) and isinstance(optimizer, str):
            if not optimizer.startswith('fmin_'):
                optimizer = "fmin_"+optimizer
            if optimizer == 'fmin_':
                optimizer = 'fmin'
            try:
                optimizer = getattr(optimize, optimizer)
            except AttributeError:
                raise ValueError("%s is not a valid optimizer" % optimizer)

        # by now kwds must be empty, since everybody took what they needed
        if kwds:
            raise TypeError("Unknown arguments: %s." % kwds)

        vals = optimizer(func, x0, args=(np.ravel(data),responsibilities), disp=0)
        if restore is not None:
            vals = restore(args, vals)
        vals = tuple(vals)
        return vals

    def _reduce_func_map(self, args, kwds):
        """
        Return the (possibly reduced) function to optimize in order to find MLE
        estimates for the .fit method.
        """
        # Convert fixed shape parameters to the standard numeric form: e.g. for
        # stats.beta, shapes='a, b'. To fix `a`, the caller can give a value
        # for `f0`, `fa` or 'fix_a'.  The following converts the latter two
        # into the first (numeric) form.
        if self.shapes:
            shapes = self.shapes.replace(',', ' ').split()
            for j, s in enumerate(shapes):
                val = kwds.pop('f' + s, None) or kwds.pop('fix_' + s, None)
                if val is not None:
                    key = 'f%d' % j
                    if key in kwds:
                        raise ValueError("Duplicate entry for %s." % key)
                    else:
                        kwds[key] = val

        args = list(args)
        Nargs = len(args)
        fixedn = []
        names = ['f%d' % n for n in range(Nargs - 2)] + ['floc', 'fscale']
        x0 = []
        for n, key in enumerate(names):
            if key in kwds:
                fixedn.append(n)
                args[n] = kwds.pop(key)
            else:
                x0.append(args[n])

        if len(fixedn) == 0:
            func = self._penalized_nnlf_map
            restore = None
        else:
            if len(fixedn) == Nargs:
                raise ValueError(
                    "All parameters fixed. There is nothing to optimize.")

            def restore(args, theta):
                # Replace with theta for all numbers not in fixedn
                # This allows the non-fixed values to vary, but
                #  we still call self.nnlf with all parameters.
                i = 0
                for n in range(Nargs):
                    if n not in fixedn:
                        args[n] = theta[i]
                        i += 1
                return args

            def func(theta, x, responsibilities):
                newtheta = restore(args[:], theta)
                return self._penalized_nnlf_map(newtheta, x, responsibilities)

        return x0, func, restore, args

    def _penalized_nnlf_map(self, theta, x, responsibilities):
        ''' Return penalized negative loglikelihood function,
        i.e., - sum (log pdf(x, theta), axis=0) + penalty
           where theta are the parameters (including loc and scale)
        '''
        loc, scale, args = self._unpack_loc_scale(theta)
        if not self._argcheck(*args) or scale <= 0:
            return np.inf
        x = np.asarray((x-loc) / scale)
        n_log_scale = np.sum(responsibilities) * np.log(scale) # the number of membership * np.log(scale)

        outcome = (self._nnlf_and_penalty(x, responsibilities, args) + n_log_scale)/np.sum(responsibilities) \
            + (-np.log(self._prior(loc, scale)))

        return outcome

    def fit(self, data, responsibilities=None, *args, **kwds):
        # copy from rv.continous_gen
        Narg = len(args)
        if Narg > self.numargs:
            raise TypeError("Too many input arguments.")

        if not np.isfinite(data).all():
            raise RuntimeError("The data contains non-finite values.")

        if responsibilities is None:
            responsibilities = np.ones(data.shape)

        # initialize
        start = [None]*2
        if (Narg < self.numargs) or not ('loc' in kwds and
                                         'scale' in kwds):
            # get distribution specific starting locations
            start = self._fitstart(data)
            args += start[Narg:-2]
        loc = kwds.pop('loc', start[-2])
        scale = kwds.pop('scale', start[-1])
        args += (loc, scale)
        x0, func, restore, args = self._reduce_func(args, kwds)

        optimizer = kwds.pop('optimizer', optimize.fmin)
        # convert string to function in scipy.optimize
        if not callable(optimizer) and isinstance(optimizer, str):
            if not optimizer.startswith('fmin_'):
                optimizer = "fmin_"+optimizer
            if optimizer == 'fmin_':
                optimizer = 'fmin'
            try:
                optimizer = getattr(optimize, optimizer)
            except AttributeError:
                raise ValueError("%s is not a valid optimizer" % optimizer)

        # by now kwds must be empty, since everybody took what they needed
        if kwds:
            raise TypeError("Unknown arguments: %s." % kwds)

        vals = optimizer(func, x0, args=(np.ravel(data),responsibilities), disp=0)
        if restore is not None:
            vals = restore(args, vals)
        vals = tuple(vals)
        return vals

    def _nnlf_and_penalty(self, x, responsibilities, args):
        cond0 = ~self._support_mask(x, *args)
        n_bad = np.count_nonzero(cond0, axis=0)
        if n_bad > 0:
            x = argsreduce(~cond0, x)[0]
            responsibilities = argsreduce(~cond0, responsibilities)[0]
        logpdf = self._logpdf(x, *args) * responsibilities
        finite_logpdf = np.isfinite(logpdf)
        n_bad += np.sum(~finite_logpdf, axis=0)
        if n_bad > 0:
            penalty = n_bad * np.log(_XMAX) * 100
            return -np.sum(logpdf[finite_logpdf], axis=0) + penalty
        return -np.sum(logpdf, axis=0)

    def _penalized_nnlf(self, theta, x, responsibilities):
        ''' Return penalized negative loglikelihood function,
        i.e., - sum (log pdf(x, theta), axis=0) + penalty
           where theta are the parameters (including loc and scale)
        '''
        loc, scale, args = self._unpack_loc_scale(theta)
        if not self._argcheck(*args) or scale <= 0:
            return np.inf
        x = np.asarray((x-loc) / scale)
        n_log_scale = np.sum(responsibilities) * np.log(scale) # the number of membership * np.log(scale)
        return self._nnlf_and_penalty(x, responsibilities, args) + n_log_scale



class irrelevant_gen(rv_continuous):

    def __init__(self, data):
        super().__init__()
        self.data = data
        self.c = self._calculate_c()

    def _calculate_c(self):
        '''Calculate the normalization factor
        1. Get the summation of areas bewteen bi_(i-1) and bi_(i), 
            which is a multiplication of count(height) and width
        2. Return the reciprocal of the summatinon
        '''
        X = self.data
        summation = 0

        for i in range(1, len(X)):
            w = X[i] - X[i-1]
            h = len(X[X[i] < X]) if X[i] > 0 else len(X[X[i] >= X])
            summation += w * h

        return 1 / summation

    def _pdf(self, X):
        if type(self.data) == None:
            raise("You should set data before using this function.")

        if type(X) == np.ndarray:
            H = np.zeros(len(X))
            for i, x in enumerate(X):
                H[i] = len(self.data[x < self.data]) if x > 0 else len(self.data[x > self.data])
            return H * self.c
        else:
            x = X
            h = len(self.data[x < self.data]) if x > 0 else len(self.data[x > self.data])
            return h * self.c


    def _cdf(self, x):
        if type(self.data) == None:
            raise("You should set data before using this function.")

        if type(x) == np.ndarray:
            results = np.zeros(len(x))
            for i in range(len(x)):
                X = np.append(self.data, x[i])
                X.sort()

                for j in range(1, len(X)):
                    if X[j] > x[i]:
                        break
                    w = X[j] - X[j-1]
                    h = len(X[X[j] < X]) if X[j] > 0 else len(X[X[j] > X])
                    if (X[j] > 0) * (X[j-1]<0):
                        if h < 10:
                            print("{}\t{}\t{}\t{}".format(X[j], X[j-1], h, x[i]))
                    results[i] += w * h
                    
            return results * self.c

        else:
            result = 0
            X = np.append(self.data[self.data<x], x[i])
            for i in range(1, len(X)):
                w = X[i] - X[i-1]
                h = len(X[X[i] < X]) if X[i] > 0 else len(X[X[i] > X])
                summation += w * h
            return result * self.c


proactive = Proactive()
reactive = Reactive(name='exponnorm')