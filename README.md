# xAI_Defense-GAN
This repository contains the partial implementation of xAI Defense-GAN. It doesn't contain the code of Defense-GAN that is used to reconstruct adversarial images.

# Getting started
- Install all required models, listed in requirements.txt
    pip install -r requirements.txt
- Run any evaluation in the experiments directory. This are terminal scripts which have some optional parameter (The models to be tested, the adversarial attacks, and for the fix mask the mask type). For example:
    python experiments/grad_defensegan.py --models model_e model_g --dss mnist cw
    python experiments/fix-mask_defensegan.py --models model_a model_b model_d --mask center

# Datasets
To make easier the reproduction of the experiments presented on the thesis. The adversarial datasets and their reconstruction using Defense-GAN are in the datasets folder. In case the user wants to add more attacks, the script join_recs.py can be used to transform the reconstruction outputted by defense-GAN to the format needed here. 

# Models
The models directory contains the trained models used for the experiments, presented in the thesis. There is also the code which can be used to train and save all modells again.

# Experiments
The experiments directory contains all different experiments used to obrtain the results presented in the thesis.

