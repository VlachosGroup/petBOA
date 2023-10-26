import os
import matplotlib
from matplotlib.ticker import AutoMinorLocator
import matplotlib.pyplot as plt
import itertools
import numpy as np

# Set matplotlib default values
font = {'size': 20}
matplotlib.rc('font', **font)
matplotlib.rcParams['axes.linewidth'] = 1.5
matplotlib.rcParams['xtick.major.size'] = 8
matplotlib.rcParams['xtick.major.width'] = 2
matplotlib.rcParams['ytick.major.size'] = 8
matplotlib.rcParams['ytick.major.width'] = 2


def plot_profile(t_vec,
                 c_profile,
                 legend_labels,
                 xlabel=None,
                 ylabel=None,
                 title=None,
                 save_path=None,
                 ):
    """
    plot the ode profiles
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    for i in range(len(c_profile.T)):
        ax.plot(t_vec, c_profile[:, i], label=legend_labels[i])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    fig_name = 'profile'
    if title is not None:
        ax.set_title(title)
        fig_name += ('_' + title.replace(' ', ''))
    if save_path is None:
        save_path = os.getcwd()
    fig.savefig(os.path.join(save_path, fig_name + '.png'), bbox_inches="tight")


def plot_overlap(t_vec1,
                 c_profile1,
                 t_vec2,
                 c_profile2,
                 legend_labels,
                 xlabel='t (s)',
                 ylabel='C (mol/L)',
                 title=None,
                 fig_name=None,
                 save_path=None):
    """
    Plot the ode profiles,
    Check whether the first profile matches with the second
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    for i in range(len(c_profile1.T)):
        ax.plot(t_vec1, c_profile1[:, i], label=legend_labels[i])
    # scatter for the second profile
    for i in range(len(c_profile2.T)):
        ax.scatter(t_vec2, c_profile2[:, i], s=35, alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    if title is not None:
        ax.set_title(title)
    if fig_name is None:
        fig_name = 'profiles_overlap'
        if title is not None:
            fig_name = ('_' + title.replace(' ', ''))
        else:
            fig_name = "overlap_plot"
    if save_path is None:
        save_path = os.getcwd()
    fig.savefig(os.path.join(save_path, fig_name + '.png'), bbox_inches="tight")


def plot_residual(t_vec1,
                  c_profile1,
                  t_vec2,
                  c_profile2,
                  legend_labels,
                  xlabel='t (s)',
                  ylabel='C (mol/L)',
                  title=None,
                  save_path=None):
    """
    Plot the ode profiles,
    Check whether the first profile matches with the second
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    for i in range(len(c_profile1.T)):
        ax.plot(t_vec1, c_profile1[:, i] - c_profile2[:, i], label=legend_labels[i])
    ax.set_xlabel(xlabel=xlabel)
    ax.set_ylabel(ylabel=ylabel)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    fig_name = 'error_residual'
    if title is not None:
        ax.set_title(title)
        fig_name += ('_' + title.replace(' ', ''))
    if save_path is None:
        save_path = os.getcwd()
    fig.savefig(os.path.join(save_path, fig_name + '.png'), bbox_inches="tight")


def plot_parity(X_data,
                Y_data,
                Y_opt,
                legend_labels,
                estimator_name,
                plot_name='Parity-Plot-MKM'):
    font = {'size': 15}
    matplotlib.rc('font', **font)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    marker = itertools.cycle(['o', '^', '+', 's', 'p', 'd', 'v'])
    markers = [next(marker) for i in range(len(legend_labels))]
    for i in range(len(legend_labels)):
        ax.scatter(y=np.array(Y_opt).T[i],
                    x=np.array(Y_data).T[i],
                    label=legend_labels[i],
                    marker=markers[i])
        ax.plot(np.linspace(0, 1),
                np.linspace(0, 1),
                color='black')
    ax.set_xlabel(xlabel="Mole Fractions - Exp.")
    ax.set_ylabel(ylabel="Mole Fractions - Model")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    plt.tight_layout()
    plt.savefig(estimator_name + '/' + plot_name)
    
