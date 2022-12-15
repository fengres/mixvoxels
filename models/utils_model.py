import torch
import torch.nn

def positional_encoding(positions, freqs):
    freq_bands = (2**torch.arange(freqs).float()).to(positions.device)  # (F,)
    pts = (positions[..., None] * freq_bands).reshape(
        positions.shape[:-1] + (freqs * positions.shape[-1], ))  # (..., DF)
    pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)
    return pts

def static_raw2alpha(sigma, dist):
    # sigma, dist  [N_rays, N_samples]
    alpha = 1. - torch.exp(-sigma*dist)

    T = torch.cumprod(torch.cat([torch.ones(alpha.shape[0], 1).to(alpha.device), 1. - alpha + 1e-10], -1), -1)

    weights = alpha * T[:, :-1]  # [N_rays, N_samples]
    return alpha, weights, T[:,-1:]

def sigma2alpha(sigma, dist):
    # sigma, dist  [N_rays, N_samples, n_frame]
    n_frames = sigma.shape[2]
    dist = dist.unsqueeze(dim=-1).expand(-1, -1, n_frames)
    alpha = 1. - torch.exp(-sigma*dist)
    return alpha

def raw2alpha(sigma, dist):
    # sigma, dist  [N_rays, N_samples, n_frame]
    n_frames = sigma.shape[2]
    dist = dist.unsqueeze(dim=-1).expand(-1, -1, n_frames)
    alpha = 1. - torch.exp(-sigma*dist)

    T = torch.cumprod(torch.cat([torch.ones(alpha.shape[0], 1, n_frames).to(alpha.device), 1. - alpha + 1e-10], dim=1), dim=1)

    weights = alpha * T[:, :-1, :]  # [N_rays, N_samples, n_frame]
    return alpha, weights, T[:,-1:,:]
