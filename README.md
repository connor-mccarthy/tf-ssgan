<div align="center">
  <h1>SSGAN Implementation Tensorflow</h1>

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

A simple API for a complex idea in current deep learning research: semi-supervised classification using generative adversarial networks (SSGANs).

This particular flavor of SSGANs is motivated by and modeled after the 2020 [research paper](https://www.aclweb.org/anthology/2020.acl-main.191.pdf) on GANBERT. See [ganbert/](`./ganbert/) for an implementation of the GANBERT model descibred in the paper using the `tf-ssgan` library.

## Getting Started

### Installation

```sh
pip install git+https://github.com/connor-mccarthy/tf-ssgan.git
```

### Code

This implementation uses the simple Keras [`Model`](https://www.tensorflow.org/api_docs/python/tf/keras/Model) API. This makes it easy to implement an SSGAN for diverse classification problems.

```python
from tf_ssgan import SSGAN

# see ./ganbert/model_components.py for generator/discriminator details
generator = make_generator(...)
discriminator = make_discriminator(...)

ssgan = SSGAN(
    generator=generator,
    discriminator=discriminator,
    name="my_ssgan",
)

ssgan.compile(
    g_optimizer=tf.keras.optimizers.Adam(1e-4),
    d_optimizer=tf.keras.optimizers.Adam(1e-4),
)

ssgan.fit(
  train_ds,
  validation_data=val_ds,
  epochs=1000,
)
```

## Reproducing GANBERT

With Python 3.8.10:

```python
python -m venv .venv
source .venv/bin/activate
pip install -r ganbert/ganbert_requirements.txt
python ganbert
```

## Citation

GANBERT paper:

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
