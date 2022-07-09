# LSTA-3N

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

The current repository faces one of the main challanges in computer vision: action recognition applied to egocentric videos.
The project proposed focuses only on verb-recognition and makes use of two well-known solution in this field: TA3N and LSTA.
This combination has led to a (naive) brand new architecture, called LSTA3N, and it has been experimented on Epic Kitchens dataset.

--------------------------------------------------------------

## Table of content
- [Description](#description)
- [Setup](#setup)
  - [Requirements](#requirements)
  - [Enviroment](#enviroment)
  - [Dataset and Features](#dataset-and-features)
- [Getting started](#getting-started)
    - [Configuration of parameters](#configuration-of-parameters)
    - [Train & Test](#train--test)
- [Contacts](#contacts)

--------------------------------------------------------------

## Description
As starting point, we based our architecture on the current state of art for Egocetric Action Recognition: TA3N. 
This model is able to focus on regions of interest in order to better discriminate particular actions.
Despite of its good performances, each frame is considered independently and the attention map is generated frame by frame, causing a loss of information.
Hence, we relied on LSTA, a recurrent unit extended from LSTM, which comes with built-in attention module and tensor-structured memory.

<p align="center"><img src="model/Architecture.jpg" alt="LSTA3N_Architecture" width="700"/></p>

The architecture requires spatial feature tensors as input which are fed through LSTA. 
LSTA returns attentively-weighted feature vectors, which are then sent to TA3N.
Putting together LSTA with TA3N implies that attention is carried out on both the spatial and the temporal dimension.

The ideas behind the project are inspired by the following papers and the code from each GitHub repos.

| Paper | Title | Implementation source |
| ----- | ----- | --------------------- |
| [1907.12743](https://arxiv.org/abs/1907.12743) | Temporal Attentive Alignment for Large-Scale Video Domain Adaptation | [EPIC-KITCHENS-100_UDA_TA3N](https://github.com/jonmun/EPIC-KITCHENS-100_UDA_TA3N) |
| [1811.10698](https://arxiv.org/abs/1811.10698) | LSTA: Long Short-Term Attention for Egocentric Action Recognition | [LSTA](https://github.com/swathikirans/LSTA) |

--------------------------------------------------------------

## Setup

### Requirements
1. The project has been tested on Linux (Ubuntu 18.04), MacOS and Windows (10)
2. Python 3.6+ version is needed
3. Good NVIDIA GPU (4GB+) is strongly suggested but it's not mandatory

### Enviroment
Once cloned the repo, some python libraries are required to setup your (virtual) enviroment properly.


They can be installed via pip:
```bash
    pip install -r requirements.txt
```

or via conda:
```bash
    conda env create -f environment.yml
```

### Dataset and Features
EPIC KITCHENS dataset can be found [here](https://epic-kitchens.github.io/2022). 

However, the dataset is made of RGB and Flow frames while the scripts in this project exploit pre-extracted features; hence, it's essential to extract them from images before running them. Features' shape should be `[Batch, N°Channels, N°Frame, Width, Height]`. We worked with features extracted from RGB frames, so `batch_size = 2048` and `n_channel = 3` .

Three different domains have been selected to perform domain shift: `P01, P22 and P08`, respectively known in this project as `D1, D2 and D3`.

--------------------------------------------------------------

## Getting started

* `dataset/` : loads and handles Epic Kitchens dataset
* `model/`
    * `LSTA/` : LSTA folder with built-in attention module and convolutional cell's structure
    * `TRN_module.py` : frame-aggregation module
    * `module.py` : configures all parameters and creates the main object that will be run in `main.py`
* `scripts/` : scripts to both train and test our architecture
* `utils/`
    * `loss.py` : different types of loss function
    * `options.py` : configuration list of parameters to build VideoModel

### Configuration of parameters
`options.py` needs a briefly configuration in order to build properly VideoModel. Most of parameters have description to help understanding their logic.
Those which need more attention are:

* *dropout_v* : randomly zeroes some of the elements of the input tensor with probability p, preventing the co-adaptation of neurons
* *dann* : decreses progressively learning rate at each epoch; alternatively, it can be reduced by *lr_decay* every *lr_steps* epoch
* *place_adv* : losses implemented in TA3N module; these are  video relation-based adversarial loss, video-based adversarial loss, frame-based adversarial loss
* *use_attn* : attention module implemented
* *beta* : weights of loss described in *place_adv*
* *gamma* : weight for the attentive entropy loss

### Train & Test
Before running `main.py` you need to choose NVIDIA GPU or CPU as processing unit. Just check in the project where the extension `.cuda()` is and eventually change to `.cpu()`. Therefore, in command line:

```bash
    python main.py
```
and it will perform both train and test.

For only-train-run:
```sh
    sh scripts/train.sh
```

For only-test-run:
```sh
    sh scripts/test.sh
```

--------------------------------------------------------------

## Contacts

| Author | Mail | GitHub | 
| ------ | ---- | ------ |
| **Lorenzo Bergadano** | s304415@studenti.polito.it | [lolloberga](https://github.com/lolloberga) |
| **Matteo Matteotti** | s294552@studenti.polito.it | [mttmtt31](https://github.com/mttmtt31) |
| **Paolo Rizzo** | paolo.rizzo@studenti.polito.it | [polrizzo](https://github.com/polrizzo) |


