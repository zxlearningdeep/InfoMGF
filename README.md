# Beyond Redundancy: Information-aware Unsupervised Multiplex Graph Structure Learning

This repository contains the source code and datasets for the NeurIPS'24 paper "Beyond Redundancy: Information-aware Unsupervised Multiplex Graph Structure Learning".

Paper Link: https://openreview.net/pdf?id=xaqPAkJnAS

The overall framework:

![InfoMGF Fig](https://github.com/zxlearningdeep/InfoMGF/blob/main/framework.png)

# Available Data

All the datasets can be downloaded from [datasets link](https://drive.google.com/file/d/1WU8j5YbwNr-cD-UQEfqX8nRDW0maDbWR/view?usp=sharing).

Place the 'data' folder from the downloaded files into the 'InfoMGF' directory.

# Requirements

This code requires the following:

* Python==3.9.16
* PyTorch==1.13.1
* DGL==0.9.1
* Numpy==1.24.2
* Scipy==1.10.1
* Scikit-learn==1.2.1
* Munkres==1.1.4
* kmeans-pytorch==0.3 

# Training

`python main.py -dataset acm` 

Here, "acm" can be replaced by "dblp", "yelp","mag".


# BibTeX

```
@inproceedings{shen2024beyond,
  title={Beyond Redundancy: Information-aware Unsupervised Multiplex Graph Structure Learning},
  author={Shen, Zhixiang and Wang, Shuo and Kang, Zhao},
  booktitle={Advances in neural information processing systems},
  year={2024}
}

```
