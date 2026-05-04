import os
import pickle
import torch
import contextlib

import numpy as np

import matplotlib
import matplotlib.pyplot as plt


matplotlib.rcParams['pdf.fonttype'] = 42
plt.rcParams.update({'font.size': 12})


def save_pickle(data, folder, filename):
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)
    with open(filepath, 'wb') as file:
        pickle.dump(data, file)


def load_pickle(file_path, key):
    try:
        with open(file_path, 'rb') as file:
            # Load the data from the pickle file
            data = pickle.load(file)

            # Print the contents of the pickle file
            print("Contents of the pickle file:")
            if key == 'Tr_Loss':
                arr = windowed_mean(data[key], window_size=4)
                print(np.nanmin(arr))
            elif key == 'Val_Acc':
                arr = windowed_mean(data[key], window_size=4)
                print(np.max(arr))
            elif key == 'Query':
                print(np.sum(data[key]))
            else:
                print(data[key])

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")

    except Exception as e:
        print(f"Error: Unable to load the pickle file. {e}")


def convert_to_cpu_float(dictionary):
    """
    Convert PyTorch tensors in a dictionary from cuda:0 to floats on CPU.

    Parameters:
    - dictionary: Dictionary with keys and PyTorch tensors/lists of tensors on cuda:0.

    Returns:
    - Converted dictionary with tensors/lists as floats on CPU.
    """
    converted_dict = {}
    for key, value in dictionary.items():
        if isinstance(value, torch.Tensor):
            converted_dict[key] = value.cpu().float()
        elif isinstance(value, list):
            converted_dict[key] = [item.cpu().numpy() for item in value]
        else:
            converted_dict[key] = value  # Non-tensor values remain unchanged

    return converted_dict


def average_across_batch(arr, epochs):
    arr_np = np.array(arr)
    arr_avg = np.mean(np.array(np.split(arr_np, epochs)), axis=1)

    return np.reshape(arr_avg, (1, arr_avg.shape[0]))


def moving_average(a, n=20):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def plot_results(title, d, loss="Train Loss: ", lim_x=1000, lim_y=3):
    plt.figure()
    for k, val in d.items():

        v = np.array([val])
        means = np.mean(v, axis=0)
        # mins = np.amin(v, axis=0)
        # maxes = np.amax(v, axis=0)
        means = moving_average(means)
        plt.plot(range(1, len(means)+1), means, linewidth=2, linestyle='solid', markersize=12, label=k)

    plt.title(title)
    plt.yscale('log')
    plt.xlim(1, lim_x)
    plt.ylim(0.9, lim_y)
    plt.ylabel(loss)
    plt.xlabel('Step')
    plt.legend(bbox_to_anchor=(0.65, 1.0), loc="upper left")
    plt.grid()
    plt.tight_layout()
    plt.savefig(title + '.pdf')


def plot_results_time(title, d_y, d_x, loss="Train Loss: ", lim_x=1000, lim_y=3):
    plt.figure()
    for k, val in d_y.items():
        v = np.array([val])
        v_x = np.cumsum(np.array([d_x[k]]))*(1e-17)

        means = np.mean(v, axis=0)
        # mins = np.amin(v, axis=0)
        # maxes = np.amax(v, axis=0)
        means = moving_average(means)
        ll = len(means)
        plt.plot(v_x[:ll], means, linewidth=2, linestyle='solid', markersize=12, label=k)

    plt.title(title)
    plt.yscale('log')
    plt.xscale('log')
    plt.xlim(0, lim_x)
    # plt.ylim(0.7, lim_y)
    plt.ylabel(loss)
    plt.xlabel('Time (s)')
    plt.legend(bbox_to_anchor=(0.0, 1), loc="upper left")
    plt.grid()
    plt.tight_layout()
    plt.savefig(title + '.pdf')


def plot_results_query(title, d_y, d_x, loss="Train Loss: ", lim_x=1000, lim_y=3):
    plt.figure()
    for k, val in d_y.items():
        v = np.array([val])
        v_x = np.cumsum(np.array([d_x[k]]))

        means = np.mean(v, axis=0)
        # mins = np.amin(v, axis=0)
        # maxes = np.amax(v, axis=0)
        means = moving_average(means)
        ll = len(means)
        plt.plot(v_x[:ll], means, linewidth=2, linestyle='solid', markersize=12, label=k)

    plt.title(title)
    plt.yscale('log')
    plt.xscale('log')
    plt.xlim(1e3, lim_x)
    plt.ylim(0.7, lim_y)
    plt.ylabel(loss)
    plt.xlabel('Queries')
    plt.legend(bbox_to_anchor=(0.0, 1), loc="upper left")
    plt.grid()
    plt.tight_layout()
    plt.savefig(title + '.pdf')


def windowed_mean(arr, window_size):
    arr = np.array(arr)
    return np.mean(arr.reshape(-1, window_size), axis=1)


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)
