# AML news detection

![cover](data/cover.JPG)

Identification and classification of news articles related to money laundering using a [Bidirectional Encoder Representations from Transformers](https://en.wikipedia.org/wiki/BERT_(language_model)) (or BERT for short - an example of a [Deep Learning Transformer](https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)) architecture).

## Resources

* Dataset: #TODO
* Working environment pre-requisites: Ubuntu18.04 LTS / Python 3.6.9 / virtualenv
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