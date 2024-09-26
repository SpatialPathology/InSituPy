from math import sqrt
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from warnings import warn

import numpy as np
import pandas as pd
import scipy
from scipy import stats
from scipy.spatial import distance as dist
from skmisc.loess import loess
from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess


class confidence_intervals:
    '''
    Class to store confidence intervals of lowess prediction.
    '''
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper

class lowess_prediction:
    '''
    Class to store results from lowess prediction.
    '''
    def __init__(self, values, stderr, smooths):
        self.values = values
        self.stderr = stderr
        self.smooths = smooths

    def confidence(self, alpha=0.05, percentile_method=False):
        if percentile_method:
            # use 2.5 and 97.5% percentiles to calculate the 95% confidence interval
            # This approach is mentioned here: https://acclab.github.io/bootstrap-confidence-intervals.html
            # However I am not sure if it is also valid for low numbers of bootstrapping cycles.
            lower = np.nanpercentile(self.smooths, 2.5, axis=1) #2.5 percent
            upper = np.nanpercentile(self.smooths, 97.5, axis=1) # 97.5 percent
        else:
            # calculate 95% CI use formula for confidence interval
            self.smooths_mean = np.nanmean(self.smooths, axis=1)
            lower, upper = stats.norm.interval(1-alpha, loc=self.smooths_mean, scale=self.stderr)

        return confidence_intervals(lower, upper)


class lowess:
    '''
    Function to perform LOWESS regression and optionally calculate the confidence intervals using bootstrapping.

    Adapted from: https://james-brennan.github.io/posts/lowess_conf/
    '''
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.fitted = False

    def predict(self, newdata, stderror=False, verbose=False, K=100, **kwargs):
        # make sure the fit() function was run before
        assert self.fitted, "Values have not been fitted yet. Run .fit() first."

        # regularly sample it onto the grid (statsmodel does not provide the solution on interpolated values)
        # for the statistics later it is important that the same grid of x values is used
        if not verbose:
            # suppress runtime error for division by nan
            np.seterr(invalid='ignore')

        values = scipy.interpolate.interp1d(self.pred_x, self.pred_y,
                                            fill_value='extrapolate')(newdata)

        if stderror:
            self.calc_stderror(newdata, K=K, **kwargs)
        else:
            self.stderr = None
            self.bootstrap_result = None

        return lowess_prediction(values, self.stderr, self.bootstrap_result)

    def bootstrap(self, x, y, newdata, sample_frac=0.5):

        samples = np.random.choice(len(x), int(len(x)*sample_frac), replace=True)

        y_s = y[samples]
        x_s = x[samples]
        y_sm = sm_lowess(y_s, x_s, frac=self.frac, it=5,
                         return_sorted = False)
        # regularly sample it onto the grid (statsmodel does not provide the solution on interpolated values)
        # for the statistics later it is important that the same grid of x values is used
        return scipy.interpolate.interp1d(x_s, y_sm, fill_value='extrapolate')(newdata)

    def calc_stderror(self, newdata, sample_frac=0.5, K=100, **kwargs):
        '''
        Interesting input on topic from:
            - https://acclab.github.io/bootstrap-confidence-intervals.html
            - https://stackoverflow.com/questions/28242593/correct-way-to-obtain-confidence-interval-with-scipy
        '''
        # calculate confidence interval using bootstrapping approach
        self.bootstrap_result = np.stack([self.bootstrap(self.x, self.y, newdata, sample_frac=sample_frac, **kwargs) for k in range(K)]).T

        # calc mean and stderr of smooths
        self.stderr = np.nanstd(self.bootstrap_result, axis=1, ddof=0) # OR std

    def fit(self, frac=0.3, **kwargs):
        self.pred_x, self.pred_y = sm_lowess(endog=self.y, exog=self.x, frac=frac,
                               it=3, return_sorted=True, **kwargs).T

        # save that object was fitted
        self.fitted = True
        self.frac = frac # save frac setting for functions run later


def bootstrap_loess(x, y, newdata, sample_frac=0.5):

    # subsample data
    samples = np.random.choice(len(x), int(len(x)*sample_frac), replace=True)
    y_s = y[samples]
    x_s = x[samples]

    # fit on subsampled data and predict for new data using LOESS
    ls = loess(x_s, y_s)
    pred = ls.predict(newdata)
    return pred.values

def calc_loess_stderror_by_bootstrap(x, y, newdata, sample_frac=0.5, K=50, **kwargs):
    '''
    Interesting input on topic from:
        - https://acclab.github.io/bootstrap-confidence-intervals.html
        - https://stackoverflow.com/questions/28242593/correct-way-to-obtain-confidence-interval-with-scipy
    '''
    # calculate confidence interval using bootstrapping approach
    bootstrap_result = np.stack([bootstrap_loess(x, y, newdata, sample_frac=sample_frac, **kwargs) for k in range(K)]).T

    # calc mean and stderr of smooths
    stderr = np.nanstd(bootstrap_result, axis=1, ddof=0) # OR std

    return stderr

class bootstrap_loess:
    '''
    Class to calculate standard error and confidence intervals of LOESS regression using bootstrapping.
    '''
    def __init__(self,
                 loess_object: loess,
                 ):
        self.x = loess_object.inputs.x
        self.y = loess_object.inputs.y

    def _single_bootstrap_loess(self, newdata, sample_frac=0.5, assert_minmax: bool = True):

        # get indices of max and min in x
        max_id = np.argmax(self.x)
        min_id = np.argmin(self.x)

        # subsample data
        samples = np.random.choice(len(self.x), int(len(self.x)*sample_frac), replace=False)

        if assert_minmax:
            # make sure that the minimum and maximum values stay
            if max_id not in samples:
                samples = np.append(samples, max_id)
            if min_id not in samples:
                samples = np.append(samples, min_id)

        # do subsampling
        y_s = self.y[samples]
        x_s = self.x[samples]

        # fit on subsampled data and predict for new data using LOESS
        ls = loess(x_s, y_s)
        pred = ls.predict(newdata)
        return pred.values

    def calc_loess_stderror_by_bootstrap(self, newdata, sample_frac=0.5, K=100, **kwargs):
        '''
        Interesting input on topic from:
            - https://acclab.github.io/bootstrap-confidence-intervals.html
            - https://stackoverflow.com/questions/28242593/correct-way-to-obtain-confidence-interval-with-scipy
        '''
        # calculate confidence interval using bootstrapping approach
        self.bootstrap_result = np.stack([self._single_bootstrap_loess(newdata, sample_frac=sample_frac, **kwargs) for k in range(K)]).T

        # calc mean and stderr of smooths
        self.stderr = np.nanstd(self.bootstrap_result, axis=1, ddof=0) # OR std

    def confidence(self, alpha=0.05, percentile_method=False):
        if percentile_method:
            # use 2.5 and 97.5% percentiles to calculate the 95% confidence interval
            # This approach is mentioned here: https://acclab.github.io/bootstrap-confidence-intervals.html
            # However I am not sure if it is also valid for low numbers of bootstrapping cycles.
            lower = np.nanpercentile(self.bootstrap_result, 2.5, axis=1) #2.5 percent
            upper = np.nanpercentile(self.bootstrap_result, 97.5, axis=1) # 97.5 percent
        else:
            # calculate 95% CI use formula for confidence interval
            self.smooths_mean = np.nanmean(self.bootstrap_result, axis=1)
            lower, upper = stats.norm.interval(1-alpha, loc=self.smooths_mean, scale=self.stderr)

        return confidence_intervals(lower, upper)


def smooth_fit(xs: np.ndarray, ys: np.ndarray,
    #x_to_fit_on: Optional[List] = None,
    #dist_thrs: Optional[float] = None,
    xmin: Optional[float] = None,
    xmax: Optional[float] = None,
    #stepsize: Optional[float] = None,
    nsteps: Optional[float] = None,
    method: Literal["lowess", "loess"] = "loess",
    stderr: bool = True,
    loess_bootstrap: bool = True,
    K: int = 100
    ):

    """Smooth curve using loess

    will perform curve fitting using skmisc.loess,
    points above 'dist_thrs' will be excluded.
    Parameters:
    ----------
    xs : np.ndarray
        x values
    ys : np.ndarray
        y values
    min: Optional[float]
        exclude (x,y) tuples were x < min
    max: Optional[float]
        exclude (x,y) tuples were x > max
    nsteps = Optional[float]
        Number of steps x is divided into for the prediction.
    method: str
        Which method to use for the smoothing. Options: "loess" or "lowess".
        "loess": Uses `skmisc.loess`. This is not implemented for Windows.
        "lowess": Uses `statsmodels.nonparametric.smoothers_lowess.sm_lowess` which is available on all tested platforms.
    stderr: bool
        Whether to calculate standard deviation of the prediction. Depends on the method used:
        "loess": Standard deviation returned by package is used.
        "lowess" Bootstrapping is used to estimate a confidence interval.
    K: int
        Only needed for `method="lowess"`. Determines number of bootstrapping cycles.

    Returns:
    -------
    A tuple with included x and y-values (xs',ys'), as well
    as fitted y-values (ys_hat) together with associated
    standard errors. The tuple is on the form
    (xs',ys',y_hat,std_err)

    From: https://github.com/almaan/ST-mLiver
    """

    # check method
    if method == "loess":
        use_loess = True
        if (stderr == True) & (bootstrap_loess == False):
            warn("When using the LOESS method and `stderr=True`, `bootstrap_loess` should be set True. Otherwise it could be that the kernel is crashing due to the high number of data points in Xenium experiments.")
    elif method == "lowess":
        use_loess = False
    else:
        raise ValueError('Invalid `method`. Expected is one of: ["loess", "lowess"')

    # sort x values
    srt = np.argsort(xs)
    xs = xs[srt]
    ys = ys[srt]

    # determine min and max x values and select x inside this range
    if xmin is None:
        xmin = xs.min()
    if xmax is None:
        xmax = xs.max()

    keep = (xs >= xmin) & (xs <= xmax)
    xs = xs[keep]
    ys = ys[keep]

    # generate loess class object
    if use_loess:
        ls = loess(xs, ys)
    else:
        ls = lowess(xs, ys)

    # fit loess class to data
    ls.fit()

    # if stepsize is given determine xs to fit the data on
    if nsteps is not None:
        stepsize = (xmax - xmin) / nsteps
        #if stepsize is not None:
        xs_pred = np.asarray(np.arange(xmin, xmax+stepsize, stepsize))
        xs_pred = np.linspace(xmin, xmax, nsteps)
        xs_pred = xs_pred[(xs_pred < xs.max()) & (xs_pred > xs.min())]

    # predict on data
    if use_loess:
        if loess_bootstrap:
            pred =  ls.predict(xs_pred, stderror=False)
        else:
            pred =  ls.predict(xs_pred, stderror=stderr)
    else:
        pred =  ls.predict(xs_pred, stderror=stderr, K=K)

    # get predicted values
    ys_hat = pred.values

    if stderr:
        # calculate confidence intervals and standard error if that was not calculated before
        if loess_bootstrap:
            bl = bootstrap_loess(ls)
            bl.calc_loess_stderror_by_bootstrap(newdata=xs_pred, K=K)
            conf = bl.confidence()
        else:
            stderr = pred.stderr
            conf = pred.confidence()

        # retrieve upper and lower confidence intervals
        lower = conf.lower
        upper = conf.upper
    else:
        lower = np.nan
        upper = np.nan

    df = pd.DataFrame({
        'x': xs_pred,
        #'y': ys,
        'y_pred': ys_hat,
        'std': stderr,
        'conf_lower': lower,
        'conf_upper': upper
    })

    #return (xs,ys,ys_hat,stderr)
    return df