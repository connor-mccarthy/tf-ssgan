<div align="center">
  <h1>GAN-BERT Implementation Tensorflow 2.5 </h1>

<p align="center">

<a href="https://github.com/connor-mccarthy/ganbert/workflows/build/badge.svg">
    <img src="https://github.com/connor-mccarthy/ganbert/workflows/build/badge.svg" alt="Python Workflow" />
</a>
<a href="https://img.shields.io/badge/python-3.8.10-blue.svg">
    <img src="https://img.shields.io/badge/python-3.8.10-blue.svg" alt="Python 3.8.10" />
</a>
<a href="https://img.shields.io/badge/code%20style-black-000000.svg">
    <img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black" >
</a>
</div>

This is a reproduction of a model from the 2020 [research paper](https://www.aclweb.org/anthology/2020.acl-main.191.pdf) on GAN-BERT, a generative adversarial approach to semi-supervised learning for natural language with neural networks.

![](./ganbert.jpeg)  
Source: [GAN-BERT paper](https://www.aclweb.org/anthology/2020.acl-main.191.pdf)

## Reproducing
With Python 3.8.10:

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/main.py
```

## Citation

Image and GAN-BERT from:

```bibtex
@inproceedings{croce-etal-2020-gan,
    title = "{GAN}-{BERT}: Generative Adversarial Learning for Robust Text Classification with a Bunch of Labeled Examples",
    author = "Croce, Danilo  and
      Castellucci, Giuseppe  and
      Basili, Roberto",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.191",
    pages = "2114--2119"
}
```
