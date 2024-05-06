# MoverScorePlus 🚀

MoverScorePlus is an enhanced implementation inspired by the original MoverScore, as introduced by the AIPHES research group for evaluating text similarity using advanced natural language processing techniques. This project seeks to build upon the foundational work done in the [MoverScore GitHub repository](https://github.com/AIPHES/emnlp19-moverscore) by incorporating additional features, optimizations, and user-friendly documentation for broader accessibility and use in natural language processing applications.

## Description ✍️

This project contains Python scripts leveraging pre-trained models from the Hugging Face Transformers library to calculate text similarity scores. Designed to assist researchers and developers in natural language processing, MoverScorePlus offers a powerful tool for evaluating the semantic similarity between two pieces of text with improvements in ease of use and performance.

## Installation 🛠️

To install the required dependencies, run the following command:

```bash
pip install -r requirements.txt
```

### Requirements ⚙️

- torch==2.2.0
- transformers==4.37.2
- pyemd==1.0.0
- numpy==1.24.4

## Usage 📖

Follow these examples to use the scripts:

1. Clone the reposity into your working folder

```bash
git clone https://github.com/BSc-Thesis-Tamas-Tibor/moverscore.git
```

2. Use the package as follows:

```python
# Example usage of the moverscore.py script
from MoverScorePlus import MoverScore, MoverScoreConfig
config = MoverScoreConfig(stop_words=["<Your stopwords>"])

# Create the configuration of the score
moverscore = MoverScore(config=config)

# Create lists of the documents, please note that
# each string represents one whole document, therefore
# the number of elements in the two lists must match
ref = ["<Your list of reference texts>"]
hyp = ["<Your list of hypothesis texts>"]

# Call the moverscore
moverscore(ref, hyp)
```

## Test the score 📔

We provide an example notebook to test the score in [google colab](https://colab.research.google.com/drive/1KvG1L2kmd2Ptt_mvDeAzihvREaqJRWAJ?usp=sharing).

## License 📄

This project is licensed under the terms of the MIT license. For more information, see the LICENSE file.

## Contributing 👥

Contributions to this project are welcome. Please follow the standard pull request process for contributions.

---

## Acknowledgments 🤝 

This project was inspired by and is based upon the foundational work done by the AIPHES research group in their [emnlp19-moverscore](https://github.com/AIPHES/emnlp19-moverscore) repository. We extend our gratitude to the original authors for their significant contributions to the field of natural language processing.

If you find it useful please cite the following paper:

```
@inproceedings{zhao2019moverscore,
  title = {MoverScore: Text Generation Evaluating with Contextualized Embeddings and Earth Mover Distance},
  month = {August},
  year = {2019},
  author = {Wei Zhao, Maxime Peyrard, Fei Liu, Yang Gao, Christian M. Meyer, Steffen Eger},
  address = {Hong Kong, China},
  publisher = {Association for Computational Linguistics},
  booktitle = {Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing},
  }
```