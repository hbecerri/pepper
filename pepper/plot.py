# Coffea still has not implemented plotting systematic uncertainties
# As a really cheap workaround file simply copies the plot1d and plotratio
# functions from Coffea and adds a few lines for systematics

import numbers
import numpy as np
from coffea.hist.hist_tools import overflow_behavior, SparseAxis, DenseAxis
from coffea.hist.plot import (
    poisson_interval, clopper_pearson_interval, normal_interval)
import mplhep as hep
import matplotlib.pyplot as plt
import warnings


def plot1d(hist, ax=None, clear=True, overlay=None, stack=False,
           overflow='none', line_opts=None, fill_opts=None, error_opts=None,
           legend_opts={}, overlay_overflow='none', density=False,
           binwnorm=None, order=None, sys=None):
    if ax is None:
        ax = plt.gca()
    else:
        if not isinstance(ax, plt.Axes):
            raise ValueError("ax must be a matplotlib Axes object")
        if clear:
            ax.clear()
    if hist.dim() > 2:
        raise ValueError("plot1d() can only support up to two dimensions "
                         "(one for axis, one to stack or overlay)")
    if overlay is None and hist.sparse_dim() == 1 and hist.dense_dim() == 1:
        overlay = hist.sparse_axes()[0].name
    elif overlay is None and hist.dim() > 1:
        raise ValueError("plot1d() can only support one dimension without an "
                         "overlay axis chosen")
    if density and binwnorm is not None:
        raise ValueError("Cannot use density and binwnorm at the same time!")
    if binwnorm is not None:
        if not isinstance(binwnorm, numbers.Number):
            raise ValueError("Bin width normalization not a number, but a "
                             "%r" % binwnorm.__class__)
    if line_opts is None and fill_opts is None and error_opts is None:
        if stack:
            fill_opts = {}
        else:
            line_opts = {}
            error_opts = {}

    axis = hist.axes()[0]
    if overlay is not None:
        overlay = hist.axis(overlay)
        if axis == overlay:
            axis = hist.axes()[1]
    if isinstance(axis, SparseAxis):
        raise NotImplementedError("Plot a sparse axis (e.g. bar chart)")
    elif isinstance(axis, DenseAxis):
        ax.set_xlabel(axis.label)
        ax.set_ylabel(hist.label)
        edges = axis.edges(overflow=overflow)
        if order is None:
            identifiers = (hist.identifiers(overlay, overflow=overlay_overflow)
                           if overlay is not None else [None])
        else:
            identifiers = order
        plot_info = {
            'identifier': identifiers,
            'label': list(map(str, identifiers)),
            'sumw': [],
            'sumw2': []
        }
        for i, identifier in enumerate(identifiers):
            if identifier is None:
                sumw, sumw2 = hist.values(sumw2=True, overflow=overflow)[()]
            elif isinstance(overlay, SparseAxis):
                sumw, sumw2 = hist.integrate(overlay, identifier).values(
                    sumw2=True, overflow=overflow)[()]
            else:
                sumw, sumw2 = hist.values(sumw2=True, overflow='allnan')[()]
                the_slice = (i if overflow_behavior(
                    overlay_overflow).start is None else i + 1,
                    overflow_behavior(overflow))
                if hist._idense(overlay) == 1:
                    the_slice = (the_slice[1], the_slice[0])
                sumw = sumw[the_slice]
                sumw2 = sumw2[the_slice]
            plot_info['sumw'].append(sumw)
            plot_info['sumw2'].append(sumw2)

        def w2err(sumw, sumw2):
            err = []
            for a, b in zip(sumw, sumw2):
                err.append(np.abs(poisson_interval(a, b) - a))
            return err

        kwargs = None
        if line_opts is not None and error_opts is None:
            _error = None
        else:
            _error = w2err(plot_info['sumw'], plot_info['sumw2'])
        if fill_opts is not None:
            histtype = 'fill'
            kwargs = fill_opts
        elif error_opts is not None and line_opts is None:
            histtype = 'errorbar'
            kwargs = error_opts
        else:
            histtype = 'step'
            kwargs = line_opts
        if kwargs is None:
            kwargs = {}

        hep.histplot(plot_info['sumw'], edges, label=plot_info['label'],
                     yerr=_error, histtype=histtype, ax=ax,
                     density=density, binwnorm=binwnorm, stack=stack,
                     **kwargs)

        if stack and error_opts is not None:
            stack_sumw = np.sum(plot_info['sumw'], axis=0)
            stack_sumw2 = np.sum(plot_info['sumw2'], axis=0)
            err = poisson_interval(stack_sumw, stack_sumw2)
            if sys is not None:
                err = (stack_sumw + (np.hypot(err - stack_sumw, sys)
                       * np.array([[-1], [1]])))
            isnan = np.logical_or.reduce(np.isnan(err), axis=0)
            err[:, isnan] = 0
            opts = {'step': 'post', 'label': 'Sum unc.', 'hatch': '///',
                    'facecolor': 'none', 'edgecolor': (0, 0, 0, .5),
                    'linewidth': 0}
            opts.update(error_opts)
            ax.fill_between(x=edges, y1=np.r_[err[0, :], err[0, -1]],
                            y2=np.r_[err[1, :], err[1, -1]], **opts)

        if legend_opts is not None:
            _label = overlay.label if overlay is not None else ""
            ax.legend(title=_label, **legend_opts)
        else:
            ax.legend(title=_label)
        ax.autoscale(axis='x', tight=True)
        ax.set_ylim(0, None)

    return ax


def plotratio(num, denom, ax=None, clear=True, overflow='none',
              error_opts=None, denom_fill_opts=None, guide_opts=None,
              unc='clopper-pearson', label=None, sys=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        if not isinstance(ax, plt.Axes):
            raise ValueError("ax must be a matplotlib Axes object")
        if clear:
            ax.clear()
    if not num.compatible(denom):
        raise ValueError("numerator and denominator histograms have "
                         "incompatible axis definitions")
    if num.dim() > 1:
        raise ValueError("plotratio() can only support one-dimensional "
                         "histograms")
    if error_opts is None and denom_fill_opts is None and guide_opts is None:
        error_opts = {}
        denom_fill_opts = {}

    axis = num.axes()[0]
    if isinstance(axis, SparseAxis):
        raise NotImplementedError("Ratio for sparse axes (labeled axis with "
                                  "errorbars)")
    elif isinstance(axis, DenseAxis):
        ax.set_xlabel(axis.label)
        ax.set_ylabel(num.label)
        edges = axis.edges(overflow=overflow)
        centers = axis.centers(overflow=overflow)

        sumw_num, sumw2_num = num.values(sumw2=True, overflow=overflow)[()]
        sumw_denom, sumw2_denom = denom.values(sumw2=True,
                                               overflow=overflow)[()]

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            rsumw = sumw_num / sumw_denom
        if unc == 'clopper-pearson':
            rsumw_err = np.abs(
                clopper_pearson_interval(sumw_num, sumw_denom) - rsumw)
        elif unc == 'poisson-ratio':
            # poisson ratio n/m is equivalent to binomial n/(n+m)
            rsumw_err = np.abs(clopper_pearson_interval(
                sumw_num, sumw_num + sumw_denom) - rsumw)
        elif unc == 'num':
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                rsumw_err = np.abs(poisson_interval(
                    rsumw, sumw2_num / sumw_denom**2) - rsumw)
        elif unc == "normal":
            rsumw_err = np.abs(normal_interval(
                sumw_num, sumw_denom, sumw2_num, sumw2_denom))
        else:
            raise ValueError("Unrecognized uncertainty option: %r" % unc)

        if error_opts is not None:
            opts = {'label': label, 'linestyle': 'none'}
            opts.update(error_opts)
            emarker = opts.pop('emarker', '')
            errbar = ax.errorbar(x=centers, y=rsumw, yerr=rsumw_err, **opts)
            plt.setp(errbar[1], 'marker', emarker)
        if denom_fill_opts is not None:
            unity = np.ones_like(sumw_denom)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                denom_unc = poisson_interval(
                    unity, sumw2_denom / sumw_denom**2)
                if sys is not None:
                    denom_unc = 1 + (np.hypot(denom_unc - 1, sys / sumw_denom)
                                     * np.array([[-1], [1]]))
            isnan = np.logical_or.reduce(np.isnan(denom_unc), axis=0)
            denom_unc[:, isnan] = 0
            opts = {'step': 'post', 'facecolor': (0, 0, 0, 0.3),
                    'linewidth': 0}
            opts.update(denom_fill_opts)
            ax.fill_between(
                edges, np.r_[denom_unc[0], denom_unc[0, -1]],
                np.r_[denom_unc[1], denom_unc[1, -1]], **opts)
        if guide_opts is not None:
            opts = {'linestyle': '--', 'color': (0, 0, 0, 0.5), 'linewidth': 1}
            opts.update(guide_opts)
            ax.axhline(1., **opts)

    if clear:
        ax.autoscale(axis='x', tight=True)
        ax.set_ylim(0, None)

    return ax
