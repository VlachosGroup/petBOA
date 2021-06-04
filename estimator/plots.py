import os
import matplotlib
import matplotlib.pyplot as plt

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

    fig_name = 'profiles_overlap'
    if title is not None:
        ax.set_title(title)
        fig_name += ('_' + title.replace(' ', ''))
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
