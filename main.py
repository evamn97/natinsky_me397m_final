import pandas as pd
import numpy as np
import numpy.ma as ma
import PIL.Image as Image
from pathos.helpers import cpu_count
from pathos.pools import ProcessPool as Pool
import matplotlib.pyplot as plt
import os
from argparse import ArgumentParser


def normalize(arr):
    """ Normalizes an input array to between 0 and 1.

    :param arr: numpy array input
    :return: normalized array
    """
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))


def single_grad(row):
    """ Finds the gradient of a single scan row.

    :param row: input row, pd.Series
    :return: row gradient, np.Ndarray
    """
    import numpy as np
    import pandas as pd
    # row = pd.Series(row.squeeze())  # only required if input is numpy array
    s_grad = np.abs(np.nan_to_num(row.rolling(15, min_periods=1, center=True).std()))
    return s_grad


def num_grad(df, n):
    """ Parallelized row gradient calculations.

    :param df: input pandas dataframe of xyz data
    :param n: number of rows/profiles to split the data for row-wise calculations
    :return: dataframe of 'z' gradient values
    """
    df_split = np.array_split(df['z'], n)
    pool = Pool(cpu_count())
    grad = pd.DataFrame(np.asarray(pool.map(single_grad, df_split)).flatten(), columns=['z'])

    return grad


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('-d', '--root-dir', help='directory with png and xyz data folders', default='data')

    return parser.parse_args()


def img_analysis(root_dir, name):
    """ Main image analysis function with figure plotting and json writing.

    :param root_dir: directory containing "png" and "xyz" folders
    :param name: extension split filename (e.g.,'TGX11Calibgrid_210701_153452')
    :return: None
    """

    input_image_file = os.path.join(root_dir, "png", (name + '.png'))      # get image file
    with Image.open(input_image_file) as img:
        img.load()

    # ----------------------------------- Numerical Analysis -----------------------------------
    input_xyz_file = os.path.join(root_dir, "xyz", (name + '.xyz'))     # get xyz file
    xyz = pd.read_csv(input_xyz_file, names=['x', 'y', 'z'], delimiter='\t', index_col=False)
    # xyz = xyz * 1000000     # scale by 1e6 for convenience since we're dealing with nanometers
    profiles = len(set(xyz['y'].values))    # find the number of profiles based on y values
    z_arr = xyz['z'].values.reshape((profiles, -1))     # get a 2D array of z-height values (like a topography map)

    z_gradient = num_grad(xyz, profiles)    # calculate gradient
    xyz['grad'] = z_gradient                # add gradient column to df

    z_gradient_arr = z_gradient.values.reshape((profiles, -1))  # get a 2D gradient array to match z_arr

    # find some useful values to use for limit
    median = z_gradient.median().values[0]
    average = np.average(z_gradient.values)
    domain = (z_gradient.max() - z_gradient.min()).values[0]

    limit = median

    # json file writing
    json_save_name = os.path.join(root_dir, "json", (name + ".json"))
    xyz.to_json(json_save_name)
    hg_json_save_name = os.path.join(root_dir, "json", (name + "-HG" + ".json"))
    xyz_HG = xyz[xyz['grad'] > limit]
    xyz_HG.to_json(hg_json_save_name)

    # mask out the low-gradient areas for plotting
    X, Y = np.meshgrid(np.arange(0, profiles), np.arange(0, profiles))
    X_mask = ma.masked_where(z_gradient_arr < limit, X)
    Y_mask = ma.masked_where(z_gradient_arr < limit, Y)

    # ---------------------------------------- Plotting ----------------------------------------

    # plot input image (png)
    fig, ax = plt.subplots(1, 3, figsize=(22, 8))
    ax[0].set_xticklabels([])
    ax[0].set_yticklabels([])
    ax[0].set_title('Input Image', fontsize='xx-large')
    ax[0].imshow(img)
    ax[0].grid(True)

    # plot gradient as image
    ax[1].set_xticklabels([])
    ax[1].set_yticklabels([])
    ax[1].set_title('Row-wise Height Gradient', fontsize='xx-large')
    ax[1].imshow(z_gradient_arr, cmap='viridis')
    ax[1].grid(True)

    # plot high gradient areas based on limit
    ax[2].set_xticklabels([])
    ax[2].set_yticklabels([])
    ax[2].set_title('High Gradient Areas', fontsize='xx-large')
    ax[2].imshow(img)
    ax[2].grid(True)
    ax[2].scatter(X_mask, Y_mask, marker='s', color='blue', alpha=0.2)

    fig.tight_layout()

    # save figures
    save_fig_name = os.path.join(root_dir, "figures", (name + ".png"))
    plt.savefig(save_fig_name)
    # plt.show()
    plt.close()     # to avoid runtime errors


if __name__ == '__main__':

    # get args
    args = parse_args()
    data_dir = args.root_dir

    file_list = os.listdir(os.path.join(data_dir, "xyz"))
    if not os.path.isdir(os.path.join(data_dir, "figures")):    # create a directory for figures if none exists
        os.mkdir(os.path.join(data_dir, "figures"))
    if not os.path.isdir(os.path.join(data_dir, "json")):       # create a directory for json output
        os.mkdir(os.path.join(data_dir, "json"))

    # run analysis on each scan
    for file in file_list:
        scan_name = os.path.splitext(file)[0]
        img_analysis(data_dir, scan_name)
