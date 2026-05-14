# Sparse-View CT Scan Reconstruction

## Problem

Computed Tomography (CT) reconstruction from sparse angular projections is a severely ill-posed inverse problem. A detector measures X-ray attenuation along a limited set of angles, producing a sinogram `b = A·x + η`, where `A` is the forward projection operator, `x` is the unknown N×N image, and `η` is noise. With few projections (sparse-view), standard Filtered Backprojection (FBP) produces severe streak artifacts that obscure diagnostic detail.

## Solution

`reconstruct.m` implements a multi-scale iterative algorithm:

1. **Initialization** — Hann-windowed FBP provides a starting estimate.
2. **Multi-resolution** — For large images, reconstruction proceeds coarse-to-fine with upsampling between levels.
3. **TV Regularization** — Total Variation regularization with adaptive edge weighting preserves boundaries while suppressing noise.
4. **CGLS-IRLS** — Each outer iteration solves a weighted least-squares subproblem via Conjugate Gradient Least Squares with Iteratively Reweighted Least Squares weights enforcing gradient sparsity.
5. **Nesterov Acceleration** — Momentum steps accelerate convergence.
6. **Adaptive λ** — Regularization strength adapts dynamically to the residual-to-noise ratio.

## Usage

```matlab
addpath('operators', 'utils', 'evaluation');
prob = load_problem('data/practice/tier1_practice.mat');
[x, info] = reconstruct(prob);
```

See `run_tier.m` for a full evaluation driver.
