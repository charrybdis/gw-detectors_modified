import numpy as np
import matplotlib.pyplot as plt
import glob

def plot_hist(counts, bins=10, figsize=(8,6), **kwargs):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111) 
    ax.hist(counts, bins=bins, **kwargs)
    ax.grid(True)
    return fig, ax


def coordinate_comparison(azims, poles, strain_azims, strain_poles, figsize=(8,8)):
    cmap = plt.get_cmap('rainbow')
    colors = cmap(np.linspace(0,1,len(azims)))
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True, figsize=figsize)
    fig.suptitle('Comparison of strain coordinates and true coordinates')
    fig.supxlabel('Azimuth (rad)')
    fig.supylabel('Poles (rad)')
    ax1.scatter(strain_azims, strain_poles, c=colors)
    ax2.scatter(azims, poles, c=colors)
    return fig, ax1, ax2


def coordinate_comparison_skymap(azims, poles, strain_azims, strain_poles, figsize=(8,6)):
    color_map = plt.cm.viridis(np.linspace(0, 1, len(azims)))

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="mollweide")
    for i in range(len(azims)):
        ax.scatter(strain_azims[i], np.array(strain_poles[i]) - np.pi/2, c=[color_map[i]], s=50, 
                   vmin=0.8, vmax=1.0)
        ax.scatter(azims[i], np.array(poles[i]) - np.pi/2, c=[color_map[i]], s=50, 
                   vmin=0.8, vmax=1.0)
        ax.plot([strain_azims[i], azims[i]], [np.array(strain_poles[i]) - np.pi/2, np.array(poles[i]) - np.pi/2],
                c='gray')
        ax.plot()
    ax.grid(True)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Best match for true signal coordinates")
    return fig, ax


def one_coord_map(coord_values, coord, other_coord, value_to_plot, figsize=(8,6)):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111) 
    for value in coord_values: 
        indexes = np.where(np.array(coord) == value)[0]
        other_coord_i = [other_coord[i] for i in indexes]
        value_to_plot_i = [value_to_plot[i] for i in indexes]
        ax.scatter(other_coord_i, value_to_plot_i, label=f'{value:.2f}')
        ax.plot(other_coord_i, value_to_plot_i, alpha=0.6)
    for val in [0, np.pi/4, np.pi/2, 3*np.pi/4, 2*np.pi]: 
        ax.plot(other_coord, np.full(len(other_coord), val), '--', c='k', alpha=0.2)
    return fig, ax


def skymap_scatter(azims, poles, value_to_plot):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="mollweide")
    sc = ax.scatter(azims[0], np.array(poles[0]) - np.pi/2, c=value_to_plot[0], s=50, vmin=0.8, vmax=1.0, cmap='Reds')
    for i in range(len(azims))[1:]:
        ax.scatter(azims[i], np.array(poles[i]) - np.pi/2, c=value_to_plot[i], s=50, vmin=0.8, vmax=1.0, cmap='Reds')
    plt.colorbar(sc)
    ax.grid(True)
    return fig, ax
