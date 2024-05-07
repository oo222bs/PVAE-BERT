# Paired Variational Autoencoders - BERT

[Language Model-Based Paired Variational Autoencoders for Robotic Language Learning](https://ieeexplore.ieee.org/document/9878160)

Last updated: 7 May 2024.

This code has been partially adapted from Copyright (c) 2018, [Tatsuro Yamada](https://github.com/ogata-lab/PRAE/)

Copyright (c) 2022, Ozan Özdemir <<ozan.oezdemir@uni-hamburg.de>>

## Requirements
- Python 3
- Pytorch
- NumPy
- Tensorboard

## Implementation
PVAE & PVAE-BERT - Pytorch Implementation

## Training Example
```
$ cd src
$ python main_pvae.py
```
- main_pvae.py: trains the PVAE model
- pvae.py: defines the PVAE and PVAE-BERT architecture
- prae.py: defines the PRAE architecture.
- channel_separated_cae: defines the channel separated CAE
- standard_cae: defines the standard CAE
- config.py: training and network configurations
- data_util.py: for reading the data
- generation.py: translates instructions to actions
- recognition.py: translates actions to descriptions
- extraction.py: extracts shared representations
- reproduction.py: reproduces the actions
- lang2lang.py: reproduces the descriptions

## Trained PVAE-BERT Model
Available [here](https://drive.google.com/file/d/1eUz6cOm_VkT1vM0V7uRMWq6ApD_wr4B0/view)

## Citation

**PVAE-BERT**
```bibtex
@ARTICLE{OKWLW22,
  author={Özdemir, Ozan and Kerzel, Matthias and Weber, Cornelius and Hee Lee, Jae and Wermter, Stefan},
  journal={IEEE Transactions on Cognitive and Developmental Systems}, 
  title={Language-Model-Based Paired Variational Autoencoders for Robotic Language Learning}, 
  year={2023},
  volume={15},
  number={4},
  pages={1812-1824},
  doi={10.1109/TCDS.2022.3204452}}
```
