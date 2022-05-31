# Paired Variational Autoencoders - BERT

[Language Model-Based Paired Variational Autoencoders for Robotic Language Learning](https://arxiv.org/abs/2201.06317)

Last updated: 31st May 2022.

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

## Citation

**PVAE-BERT**
```bibtex
@Article{OKWLW22,
  author       = "Özdemir, Ozan and Kerzel, Matthias and Weber, Cornelius and Lee, Jae Hee and Wermter, Stefan",
  title        = "Language Model-Based Paired Variational Autoencoders for Robotic Language Learning",
  journal      = "arXiv:2201.06317",
  month        = "Jan",
  year         = "2022",
  url          = "https://arxiv.org/pdf/2201.06317.pdf"
}
```