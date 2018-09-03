from __future__ import print_function, division
"""
Statistical analysis for:

"Emergence of Music Universals through Iterated Pitch Learning: A Pilot Study"
by Jack Goffinet (2018)

Unfortunately, there seems to be a bug in PyMC version 2.3.6 that prevents me
from setting a random seed in the linear model (line 85).
"""
__author__ = "Jack Goffinet"
__date__ = "April, September 2018"


import numpy as np
import pymc as pm
from pymc.Matplot import plot


SIGMA_SQUARED = 10.0 # for priors
ITERATIONS = 220000
BURN_IN = 20000


gen_7_avg_intervals = np.array([1.7778, 2.8889, 4.5556, 1.7778, 5.3333, \
                                        1.7778, 2.5556, 6.3333])

constant_data = {
        'mean interval diffs':np.array([-0.5, -1.5139, -0.5972, 0.375, \
                                        -0.4861, -0.0694, -0.25]),
        'mean radius diffs':np.array([-0.475, -0.6125, -0.17, 0.095, -0.2275, \
                                        -0.025, -0.1125]),
        'unison interval content':np.array([0.0, 0.0, 0.2, -0.3333, 0.5, 0.0, \
                                        -0.1667]),
}

linear_data = {
        'pitches by avg interval' : np.array([10.0, 5.0, 7.0, 7.0, 5.0, 10.0, \
                                        6.0, 7.0]),
        'pcs by avg interval' : np.array([10.0, 5.0, 7.0, 7.0, 4.0, 10.0, 6.0, \
                                        6.0]),
        'accuracy by gen':np.array([-0.5645, -1.0986, 0.5108, -0.2513, \
                                        -0.3536, 1.5506, 0.8473]),
        'contour accuracy by gen':np.array([1.7130, 0.3939, 1.2528, 0.5108, \
                                        0.9555, 2.8332, 1.9459]),
        'descending contour gen 0':np.array([1.3, 1.3, 1.425, 2.425, 0.3, \
                                        -1.45, -2.575, 0.675, -0.45, -2.95]),
        'descending contour gen 7':np.array([0.825, 2.575, 0.325, 1.325, \
                                        -0.425, 1.7, 0.075, -0.425, -1.675, \
                                        -4.3]),
        'pitches by gen':np.array([67.0, 66.0, 63.0, 58.0, 61.0, 55.0, 53.0, \
                                        57.0]),
        'pitch classes by gen':np.array([58.0, 53.0, 58.0, 52.0, 52.0, 51.0, \
                                        49.0, 55.0])
}

quadratic_data = {
        'arched contour gen 0':np.array([1.3, 1.3, 1.425, 2.425, 0.3, -1.45, \
                                        -2.575, 0.675, -0.45, -2.95]),
        'arched contour gen 7':np.array([0.825, 2.575, 0.325, 1.325, -0.425, \
                                        1.7, 0.075, -0.425, -1.675, -4.3])
}


def build_constant_model(data, sigma_squared=10.0):
    tau = 1.0/sigma_squared
    B_0 = pm.Normal('B_0', mu=0.0, tau=tau, doc='mean', rseed=0)
    SI = pm.HalfNormal('SI', tau=tau, doc='sigma', rseed=0)
    OUT = pm.Normal('accuracies', mu=B_0, tau=1.0/SI**2, value=data, \
                                        observed=True, rseed=0)
    return [B_0, SI, OUT]


def build_linear_model(x_vals, data, sigma_squared=10.0):
    tau = 1.0/sigma_squared
    B_0 = pm.Normal('B_0', mu=0.0, tau=tau, doc='line slope', rseed=0)
    B_1 = pm.Normal('B_1', mu=0.0, tau=tau, doc='line intercept', rseed=0)
    SI = pm.HalfNormal('SI', tau=tau, doc='line sigma', rseed=0)
    MU = pm.Deterministic(
            name = 'mus',
            eval = lambda B_0, B_1 : B_0*x_vals + B_1,
            parents = {'B_0':B_0, 'B_1':B_1},
            doc='mu for line',
            plot=False,
            # rseed=0  NOTE: PyMC version 2.3.6 throws an error here.
    )
    OUT = pm.Normal('accuracies', mu=MU, tau=1.0/SI**2, value=data, \
                                        observed=True, rseed=0)
    return [B_0, B_1, SI, MU, OUT]


def build_quadratic_model(x_vals, data, sigma_squared=10.0):
    tau = 1.0/sigma_squared
    B_0 = pm.Normal('B_0', mu=0.0, tau=tau, doc='first coefficient', rseed=0)
    B_1 = pm.Normal('B_1', mu=0.0, tau=tau, doc='second coefficient', rseed=0)
    B_2 = pm.Normal('B_2', mu=0.0, tau=tau, doc='third coefficient', rseed=0)
    SI = pm.HalfNormal('SI', tau=tau, doc='sigma', rseed=0)
    MU = pm.Deterministic(
            name = 'mus',
            eval = lambda B_0, B_1, B_2 : B_0*x_vals**2 + B_1*x_vals + B_2,
            parents = {'B_0':B_0, 'B_1':B_1, 'B_2':B_2},
            doc='mu for parabola',
            plot=False,
            rseed=0
    )
    OUT = pm.Normal('accuracies', mu=MU, tau=1.0/SI**2, value=data, \
                                        observed=True, rseed=0)
    return [B_0, B_1, B_2, SI, MU, OUT]


def get_bf_less_than(trace, value):
    p = sum(1.0 if temp < value else 0.0 for temp in trace)
    p /= len(trace)
    return p/(1-p), (1-p)/p, p


groups = [constant_data, linear_data, quadratic_data]
models = [build_constant_model, build_linear_model, build_linear_model]


if __name__=='__main__':
    """Run all statistical tests."""
    output_file = "results.txt"
    to_write = [["test name", "2.5q,norm", "97.5q,norm", "2.5q", "97.5q", \
                                        "bf(<0)", "bf(>0)", "probability"]]
    for i, group, model in zip(range(len(groups)), groups, models):
        for label, data in sorted(group.iteritems()):
            pm.numpy.random.seed(0)
            np.random.seed(0)
            temp_write = [label]
            # First, z-score y values.
            y_mean = None
            if i > 0:
                y_mean = np.mean(data)
                data -= y_mean
            y_std = np.std(data)
            data /= y_std
            # Define x values and z-score them as well.
            if label in ["pitches by avg interval", "pcs by avg interval"]:
                x_vals = np.copy(gen_7_avg_intervals)
            else:
                x_vals = np.arange(float(len(data)))
            x_mean = np.mean(x_vals)
            x_vals -= x_mean
            x_std = np.std(x_vals)
            x_vals /= x_std
            # Build a model with a weakly informative prior.
            if i == 0:
                stochastics = model(data, sigma_squared=SIGMA_SQUARED)
            else:
                stochastics = model(x_vals, data, sigma_squared=SIGMA_SQUARED)
            M = pm.MCMC(stochastics)
            if i == 0:
                M.use_step_method(pm.AdaptiveMetropolis, stochastics[:-1])
            else:
                M.use_step_method(pm.AdaptiveMetropolis, stochastics[:-2])
            # Sample from the posterior.
            M.sample(iter=ITERATIONS, burn=BURN_IN, thin=1, verbose=0)
            # Get stats.
            quantiles = M.stats()['B_0']['quantiles']
            temp_write.append("{:.4f}".format(quantiles[2.5]))
            temp_write.append("{:.4f}".format(quantiles[97.5]))
            for q in [2.5, 97.5]:
                temp = quantiles[q]
                if i == 0:
                    temp *= y_std
                elif i == 1:
                    temp = temp * y_std / x_std
                else: # i == 2
                    temp = temp * y_std / x_std**2
                temp_write.append("{:.4f}".format(temp))
            trace = M.trace('B_0', chain=-1)[:]
            bf, bf_alt, prob = get_bf_less_than(trace, 0.0)
            temp_write.append("{:.4f}".format(bf))
            temp_write.append("{:.4f}".format(bf_alt))
            temp_write.append("{:.6f}".format(prob))
            to_write.append(temp_write)
            # Plot traces.
            pm.Matplot.plot(M, verbose=0)
            print("check traces")

    # Write stats to <output_file>.
    col_width_1 = max(len(row[0]) for row in to_write) + 2
    col_width_2 = max(len(word) for row in to_write for word in row[1:]) + 2
    temp_str = ""
    for row in to_write:
        temp_str += "".join(row[0].ljust(col_width_1))
        temp_str += "".join(word.ljust(col_width_2) for word in row[1:])
        temp_str += "\n"
    with open(output_file, 'a') as text_file:
        text_file.write(temp_str)
