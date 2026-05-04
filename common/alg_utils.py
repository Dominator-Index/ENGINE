import math
import torch


@torch.no_grad()
def generate_basis(m, rank, ref_mat, orthogonal=False):
    Z = torch.randn(m, rank, dtype=ref_mat.dtype, device=ref_mat.device)

    if orthogonal is True:
        U, S, Vt = torch.linalg.svd(Z.T @ Z)
        S = 1. / torch.sqrt(S)
        return (Z @ U @ torch.diag(S) @ Vt) * math.sqrt(rank)
    else:
        return Z


@torch.no_grad()
def sgd_with_wdecay(param, grad, lr, weight_decay):
    param.mul_(1 - lr * weight_decay)
    param.add_(grad, alpha=-lr)


@torch.no_grad()
def ema_update(param, grad, lr, weight_decay, beta):
    updated_param = param.mul(1 - lr * weight_decay)
    updated_param.add_(grad, alpha=-lr)

    param.mul_(beta).add_(updated_param, alpha=1.0 - beta)


def exponential_cooling_schedule(initial_temp, alpha, iteration):
    """
    Exponential cooling schedule function for simulated annealing.

    Parameters:
    initial_temp (float): The initial temperature.
    alpha (float): The cooling rate (0 < alpha < 1).
    iteration (int): The current iteration number.

    Returns:
    float: The temperature at the given iteration.
    """
    return initial_temp * (alpha ** iteration)


def logarithmic_cooling_schedule(initial_temp, iteration):
    """
    Logarithmic cooling schedule function for simulated annealing.

    Parameters:
    initial_temp (float): The initial temperature.
    iteration (int): The current iteration number.

    Returns:
    float: The temperature at the given iteration.
    """
    return initial_temp / (1 + math.log(1 + iteration))


def linear_cooling_schedule(initial_temp, final_temp, max_iterations, iteration):
    """
    Linear cooling schedule function for simulated annealing.

    Parameters:
    initial_temp (float): The initial temperature.
    final_temp (float): The final temperature.
    max_iterations (int): The maximum number of iterations.
    iteration (int): The current iteration number.

    Returns:
    float: The temperature at the given iteration.
    """
    return initial_temp - (initial_temp - final_temp) * (iteration / max_iterations)
