=======================================
Controllable Generation for Climate Modeling
=======================================

.. image:: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg
	:target: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg
	:alt: License

Official implementation of the experiments in the paper `"**Controllable Generation for Climate Modeling
**" <https://s3.us-east-1.amazonaws.com/climate-change-ai/papers/neurips2022/61/paper.pdf>`_ presented at the Tackling Climate Change with Machine Learning, NeurIPS Workshop 2022. 
 
`This repository <https://github.com/MoulikChoraria/Climate_Generation>`_ is implemented in `PyTorch <https://pytorch.org/>`_.



Browsing the code
========================
Parts of the code in Pytorch Lightning are adapted from the repository `here <https://github.com/nocotan/pytorch-lightning-gans>`. The code structure is the following:

*    ``setup_dataset.py``: The `file <https://github.com/MoulikChoraria/Climate_Generation/blob/main/setup_dataset.py>`_ allows the user to download the `CHIRPS <https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/netcdf/p05/>`_ dataset at a 0.05 resolution.

*    ``data_utils.py``: The `file <https://github.com/MoulikChoraria/Climate_Generation/blob/main/data_utils.py>`_ contains the utilities for reading and pre-processing the dataset, prepping the data loader and formatting the data suitably for conditional generation.

*    ``models``: The `folder <https://github.com/MoulikChoraria/Climate_Generation/blob/main/models>`_ houses code for different deep network implementations for training the generator and discriminator respectively, including polynomial models, unet-type models and resnet based models.

*    ``train_module py``: The `file <https://github.com/MoulikChoraria/Climate_Generation/blob/main/train_module.py>`_ abstracts the training module for training conditional GANs in `PyTorch Lightning <https://www.pytorchlightning.ai/>`_ , and currently houses implementations for standard, wasserstein and auxiliary cGANS.

*    ``main.py``: The `file <https://github.com/MoulikChoraria/Climate_Generation/blob/main/main.py>`_ houses the main script for training the conditional GANs for climate generation as required.

Summary
==========================

Recent years have seen increased interest in modeling future climate trends, especially from the point of view of accurately predicting, understanding and mitigating
downstream impacts. For instance, current state-of-the-art process-based agriculture models rely on high-resolution climate data during the growing season for
accurate estimation of crop yields. However, high-resolution climate data for future climates is unavailable and needs to be simulated, and that too for multiple possible climate scenarios, which becomes prohibitively expensive via traditional methods. Meanwhile, deep generative models leveraging the expressivity of neural networks have shown immense promise in modeling distributions in high dimensions. Here, we cast the problem of simulation of climate scenarios in a generative modeling
framework. Specifically, we leverage GANs (Generative Adversarial Networks) for simulating synthetic climate scenarios. We condition the model by quantifying the degree of “extremeness" of the observed sample, which allows us to sample from different parts of the distribution. We demonstrate the efficacy of the proposed method on the CHIRPS precipitation dataset. 


Citing
======
If you use this code, please cite [1]_:

*BibTeX*:: 

  @inproceedings{ChorariaSZWV2022,
  author    = {Choraria, Moulik and Szwarcman, Daniela and Zadrozny, Bianca and Watson, Campbell D and Varshney, Lav R.},
  title     = {Controllable Generation for Climate Modeling},
  year      = {2022},
  booktitle = {Annual Conference on Neural Information Processing Systems},
  booksubtitle = {Tackling Climate Change with Machine Learning Workshop, NeurIPS 2022},
  url = {https://s3.us-east-1.amazonaws.com/climate-change-ai/papers/neurips2022/61/paper.pdf}}
  
References
==========

.. [1] Moulik Choraria, Daniela Szwarcman, Bianca Zadrozny, Campbell D. Watson and Lav R. Varshney. **Controllable Generation for Climate Modeling**, Tackling Climate Change with Machine Learning, NeurIPS Workshop 2022.
 
