# Introductory JAX tutorials 
## using simple, practical problems common in astrophysics

- This repository contains tutorials introducing JAX using relatively simple, practical inference problems. 
- This was originally developed for an invited tutorial at the 2025 Simulation-Based Inference for Galaxy Evolution conference in Bristol, UK. 
- The first tutorial presented in Bristol involved inferring the parameters of a 2D Gaussian from noisy 2D images. 
- As time permits, I'll flesh out that tutorial some more and add other simple examples (e.g., spectrophotometric fitting, dynamical systems, etc.). 
- These tutorials are intended to be pedagogical and exploratory -- they are not necessarily research-grade but may be broadly useful. 
- Part of the motivation is (for me!) to learn how to use state-of-the-art JAX-based packages for various tasks spanning Bayesian inference, machine learning, physical modeling, etc.
- Comments and contributions welcome! Email Viraj Pandya at vgp2108@columbia.edu to be added as a repo collaborator


# Google Colab implementation
Early Bristol SBI Galaxy Evolution conference tutorial is available on Colab at:  
https://colab.research.google.com/drive/1zoXet-ez1EckrrmTG9YVXARJXvmR9fsX#scrollTo=7f48e649-1023-44a2-9536-3fd9aeb7e763


# Installing JAX
Best to follow https://docs.jax.dev/en/latest/installation.html 

In the near future I'll add instructions for installing w/ support for Macbook GPU and/or Nvidia GPU support on clusters

Here's what I found helpful for my M1 Macbook pro (no GPU yet -- I run on cluster/colab for that):
- download [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install) or [mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html)
- CONDA_SUBDIR=osx-arm64 conda create -n bristol python=3.11
- conda activate bristol
- pip install jax jupyter matplotlib numpyro optax equinox flax pandas astropy chainconsumer
- python -m ipykernel install --user --name bristol
- jupyter lab &

The last line will open jupyter lab in your browser where you can then open tutorial_jax_sbi_galevo.ipynb






