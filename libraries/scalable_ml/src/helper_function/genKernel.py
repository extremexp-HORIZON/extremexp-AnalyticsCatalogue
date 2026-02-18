import torch
import numpy as np

def generateRandKernel(ksize):
    """Generate a random uniform kernel"""
    return torch.rand((ksize, ksize))

def gabor_kernel(ksize, sigma, theta, lam, psi, gamma=0.5):
    """
    ksize: kernel size (int)
    sigma: Gaussian envelope std
    theta: orientation (radians)
    lam: wavelength of sinusoidal factor
    psi: phase offset
    gamma: spatial aspect ratio
    """
    xmax = ksize // 2
    ymax = ksize // 2
    xmin = -xmax
    ymin = -ymax
    
    y, x = np.meshgrid(range(ymin, ymax+1), range(xmin, xmax+1))
    
    # rotation
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)
    
    gb = np.exp(-0.5 * (x_theta**2 + (gamma**2) * y_theta**2) / sigma**2) * \
         np.cos(2 * np.pi * x_theta / lam + psi)
    return gb.astype(np.float32)


def gabor_bank(out_channels, in_channels, ksize):
    weights = torch.zeros(out_channels, in_channels, ksize, ksize)
    
    # Choose some parameters
    sigmas = np.linspace(ksize/4, ksize/2, num=3)
    lams = np.linspace(ksize/2, ksize, num=3)
    psis = [0, np.pi/2]
    
    idx = 0
    for sigma in sigmas:
        for lam in lams:
            for psi in psis:
                if idx >= out_channels:
                    break
                theta = np.pi * idx / out_channels  # different orientation per filter
                kernel = gabor_kernel(
                    ksize=ksize,
                    sigma=sigma,
                    theta=theta,
                    lam=lam,
                    psi=psi,
                    gamma=0.5,
                )
                # Same kernel for each input channel (like grayscale replicated on RGB)
                for c in range(in_channels):
                    weights[idx, c] = torch.from_numpy(kernel)
                idx += 1
            if idx >= out_channels:
                break
        if idx >= out_channels:
            break

    # Normalize each filter to zero mean, unit norm (optional but common)
    w_flat = weights.view(out_channels, -1)
    w_flat -= w_flat.mean(dim=1, keepdim=True)
    w_flat /= (w_flat.norm(dim=1, keepdim=True) + 1e-8)
    weights = w_flat.view_as(weights)
    return weights


def smooth_random_kernels(out_channels, in_channels, ksize, smooth_ksize=5, sigma=1.0):
    # Step 1: random normal init
    w = torch.randn(out_channels, in_channels, ksize, ksize)
    
    # Build Gaussian blur kernel (2D, shared over channels)
    ax = torch.arange(-smooth_ksize // 2 + 1., smooth_ksize // 2 + 1.)
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, smooth_ksize, smooth_ksize)
    
    # Apply blur per (out_channel, in_channel)
    w_flat = w.view(-1, 1, ksize, ksize)  # treat each 2D kernel as "image"
    pad = smooth_ksize // 2
    w_smooth = F.conv2d(
        F.pad(w_flat, (pad, pad, pad, pad), mode="reflect"),
        kernel
    )
    w_smooth = w_smooth.view(out_channels, in_channels, ksize, ksize)
    
    # Normalize
    w_flat = w_smooth.view(out_channels, -1)
    w_flat -= w_flat.mean(dim=1, keepdim=True)
    w_flat /= (w_flat.norm(dim=1, keepdim=True) + 1e-8)
    return w_flat.view_as(w_smooth)

