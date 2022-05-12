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


def scan_grad(df, n):
    """ Parallelized row gradient calculations.

    :param df: input pandas dataframe of xyz data
    :param n: number of rows/profiles to split the data for row-wise calculations
    :return: dataframe of 'z' gradient values
    """
    df_split = np.array_split(df.iloc[:, 2], n)
    pool = Pool(cpu_count())
    grad = pd.DataFrame(np.asarray(pool.map(single_grad, df_split)).flatten(), columns=['z'])

    return grad


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('-d', '--root-dir', help='directory with png and/or xyz files', default='data/png')

    return parser.parse_args()


def img_analysis(root_dir, name):
    """ Main image analysis function with figure plotting and json writing.

    :param root_dir: directory containing png images or xyz files
    :param name: extension split filename (e.g.,'TGX11Calibgrid_210701_153452')
    :param mode: determines analysis mode. must be 'png' or 'xyz'.
    :return: None
    """
    mode = os.path.splitext(name)[-1]

    if mode.lower() == '.jpg' or mode.lower() == '.jpeg':   # processes jpegs the same way as png
        mode = '.png'

    if mode == '.png':
        # ------------------------------------- Image Analysis -------------------------------------

        input_file = os.path.join(root_dir, name)      # get image file
        with Image.open(input_file) as img:
            img = img.convert('RGB')  # convert to 3-channel (in case it's RGBA)
            img.load()

        z_arr = np.average(np.asarray(img), axis=2)
        profiles, x_samples = z_arr.shape

        X, Y = np.meshgrid(np.arange(0, x_samples), np.arange(0, profiles))
        xyz_df = pd.DataFrame(np.stack((X.flatten(), Y.flatten(), z_arr.flatten()), axis=1), columns=['x pixel', 'y pixel', 'intensity'])

        xyz_df['gradient'] = scan_grad(xyz_df, profiles)    # calculate gradient and add to df
        z_gradient_arr = xyz_df['gradient'].values.reshape((profiles, -1))  # get a 2D array of gradient values for plotting

    elif mode == '.xyz':
        # ----------------------------------- Numerical Analysis -----------------------------------
        input_file = os.path.join(root_dir, name)     # get xyz file
        xyz_df = pd.read_csv(input_file, names=['x', 'y', 'z height'], delimiter='\t', index_col=False)
        # xyz_df *= 1000000     # scale by 1e6 for convenience since we're dealing with nanometers
        profiles = len(set(xyz_df['y'].values))    # find the number of profiles based on y values
        z_arr = xyz_df['z height'].values.reshape((profiles, -1))     # get a 2D array of z-height values (like a topography map)
        x_samples = z_arr.shape[1]

        xyz_df['gradient'] = scan_grad(xyz_df, profiles)                # add gradient column to df

        z_gradient_arr = xyz_df['gradient'].values.reshape((profiles, -1))  # get a 2D gradient array

    else:
        assert mode in ['.png', '.xyz']   # if mode isn't png or xyz, break out of function
        return None

    # find some useful values to use for limit
    median = np.median(xyz_df['gradient'].values)
    average = np.average(xyz_df['gradient'].values)

    limit = median

    # json file writing
    json_save_path = os.path.join(os.path.dirname(root_dir), "json")
    json_save_name = os.path.join(json_save_path, (name + ".json"))
    xyz_df.to_json(json_save_name)
    hg_json_save_name = os.path.join(json_save_path, (name + "-HG" + ".json"))
    xyz_HG = xyz_df[xyz_df['gradient'] > limit]
    xyz_HG.to_json(hg_json_save_name)

    # mask out the low-gradient areas for plotting
    X, Y = np.meshgrid(np.arange(0, x_samples), np.arange(0, profiles))
    X_mask = ma.masked_where(z_gradient_arr < limit, X)
    Y_mask = ma.masked_where(z_gradient_arr < limit, Y)

    # ---------------------------------------- Plotting ----------------------------------------

    # plot input image
    fig, ax = plt.subplots(1, 3, figsize=(22, 8))
    ax[0].set_xticklabels([])
    ax[0].set_yticklabels([])
    ax[0].set_title('Input Image', fontsize='xx-large')
    if mode == '.png':
        ax[0].imshow(img)
    else:
        ax[0].imshow(z_arr)
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
    if mode == '.png':
        ax[2].imshow(img)
    else:
        ax[2].imshow(z_arr)
    ax[2].grid(True)
    ax[2].scatter(X_mask, Y_mask, marker='s', color='blue', alpha=0.2)

    fig.tight_layout()

    # save figures
    save_fig_path = os.path.join(os.path.dirname(root_dir), "figures")
    save_fig_name = os.path.join(save_fig_path, (name + ".png"))
    plt.savefig(save_fig_name)
    # plt.show()
    plt.close()     # to avoid runtime errors


if __name__ == '__main__':

    # get args
    args = parse_args()
    data_dir = os.path.normpath(args.root_dir)

    file_list = os.listdir(data_dir)
    if not os.path.isdir(os.path.join(os.path.dirname(data_dir), "figures")):    # create a directory for figures if none exists
        os.mkdir(os.path.join(os.path.dirname(data_dir), "figures"))
    if not os.path.isdir(os.path.join(os.path.dirname(data_dir), "json")):       # create a directory for json output
        os.mkdir(os.path.join(os.path.dirname(data_dir), "json"))

    # run analysis on each scan
    for file in file_list:
        if any(ext in file for ext in ['.png', '.xyz']):    # check if file is an acceptable format before analyzing
            img_analysis(data_dir, file)
