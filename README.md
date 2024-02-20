# README

## Introduction
Our code based on Fast Image Retrieval (FIRe), an open source image retrieval project release by Center of Image and Signal Processing Lab (CISiP Lab), Universiti Malaya.

### License
This project is released under [BSD 3-Clause License](https://github.com/CISiPLab/fast-image-retrieval/blob/main/LICENSE).

### Installation
Please head up to [Get Started Docs](https://fast-image-retrieval.readthedocs.io/en/latest/get_started.html) for guides on setup conda environment and installation.

### Datasets
|Dataset|Name in framework|
|---|---|
|AwA2|awa2|
|CUB|cub|
|SUN|sun|


### Structure
Download the data separately and store it in the following structure.

```
|-- code/
    |-- bert/
    |-- submit/
    |-- requirements.txt
    |-- README.md
|-- data
    |-- AwA2/
    |-- CUB_200_2011/
    |-- SUN/
    |-- xlsa17/
```


### Train

Copy the get_attr_xx.py to `/data/` and run to generate text.
```
python get_attr_awa2.py
python get_attr_cub.py
python get_attr_sun.py
```
`cd code/submit/` and train 

```
python train.py --config configs/templates/our.yaml --ds awa2 --nbit 64
python train.py --config configs/templates/our.yaml --ds cub --nbit 64
python train.py --config configs/templates/our.yaml --ds sun --nbit 64
```