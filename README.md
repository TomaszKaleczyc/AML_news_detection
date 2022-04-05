# AML news detection

![cover](data/cover.JPG)


## Resources

* Dataset: #TODO
* Working environment pre-requisites: Ubuntu18.04 LTS / Python 3.8.10 / virtualenv
* Use the `Makefile` commands to:
  * create the project virtual environment
  * print the source terminal command to activate environment in terminal
  * run tensorboard to view training progress & results

## Project structure

```
├── data                                # Data used in the project
├── environment                         # Definition and contents of the project virtualenv
├── output                              # Default location for model training results
│   └── lightning_logs                  # Generated automatically by pytorch-lightning during training
└── src                                 # Source files of the project
    ├── dataset                         # Classes used in building and managing the project dataset
    ├── model                           # Classes and functions used in building and running the models
    ├── settings                        # Setting variables used across the repo 
    └── utilities                       # Utility functions

```

## Problem statement

This project is intended to create a model for identification and classification of news articles related to money laundering.

The classification is aimed at determining whether a given article belongs to one of the below classes:
* non-AML related
* AML related - allegations / accusations / charges context
* AML related - conviction / sentencing context

For the purpose of this analysis a small dataset will be curated from a number of links found online.

## Approach

The problem outlined is a Natural Language Processing (NLP) class problem - text classification. For some time now, state of the art results for such problems are achieved using [Deep Learning Transformer architectures](https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)).

For this reason we will use a Transformer based architecture to solve the problem. The backbone for our model will be the popular [Bidirectional Encoder Representations from Transformers](https://en.wikipedia.org/wiki/BERT_(language_model)) - or BERT for short:

<img src="data/using-bert.png" alt="drawing" width="750"/>

The reason for using this model is it has an open source implementation (available via [huggingface.co](https://huggingface.co/)) that was pre-trained on a very large corpora that included BooksCorpus (ca. 800M words) and English Wikipedia (ca. 2500M words). This will allow us to use transfer learning and fine-tune the model effectively using a much smaller dataset.

We will not be using a BERT classifier directly though. Our posed problem is classification of articles, which can be of variable length - and as we will see in the data analysis this often means exceeding the limit of 512 tokens (words) imposed by BERT.

There are a number of ways to overcome this limitation, all involving splitting the input text into smaller parts, 'digestible' for BERT. These approaches involve among others:
* Head / Tail analysis - where the text is truncated to the maximum length (using the first / last words respectively)
* Pooling article part feature vectors using symmetrical functions, such as sum or mean

The above approaches result in removing the context information from the article and rely on effective partial classification of the article.

This is why in this repository we will use a custom built architecture of the Recurrence over BERT (RoBERT) class as described in the [HIERARCHICAL TRANSFORMERS FOR LONG DOCUMENT CLASSIFICATION paper ](https://arxiv.org/pdf/1910.10781.pdf) by Raghavendra Pappagari et al.

The basic idea behind RoBERT (and the twin Transformer over BERT architecture) is that article parts (with the order preserved and some overlap between parts introduced) can be considered sequences themselves. As such, they can be analysed using a sequence analysis model, like an LSTM or Transformer.

For the purpose of this analysis, due to time and capacity constraints we will use a RoBERT architecture. The architecture will be made out of the following parts:
1. `feature_extractor` - the pre-trained BERT architecture that transforms article parts into feature vectors (embeddings)
1. `aggregating_network` - a custom LSTM, processing the feature vectors sequentially
1. `predictor` - a custom fully-connected classifier that transforms the final LSTM layer into the output prediction logits

The analysis will be described in separate notebooks describing:
1. `PROOF_OF_CONCEPT` - training the custom architecture on a larger dataset as a proof of concept for the model implementation
1. `AML_ANALYSIS` - trying out the architecture on the custom built dataset for the problem statement