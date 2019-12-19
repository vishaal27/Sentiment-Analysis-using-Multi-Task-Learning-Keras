# Sentiment-Analysis-using-Multi-Task-Learning-Keras

## Introduction
A course project for the Deep Learning (CSE641) Spring'19 course at IIIT Delhi. The aim was to perform transfer learning on the task of sentiment classification
using two datasets - Imdb and Insults in social commentary. 
Three major approaches were followed: pure self-attention based models (transformers), transfer learning using chain-thaw method and multi-task learning. This repository contains the code for the multi-task learning architecture.

## Navigating through the code
- `process_sentiment.py` creates word embeddings to pass into the input embedding layer.
- `test.py` creates and trains the model (with LSTMs) with IMDB as the primary task and Insults as the auxiliary task.
- `test_pure_dense.py` creates and trains the model (with only FC layers) with IMDB as the primary task and Insults as the auxiliary task
- `test_imdb.py` creates and trains the model on the primary task only (IMDB)
- `test_sentiment.py` creates and trains the model on the auxiliary task only (IMDB)
- `checkpoints_<X>` contains the saved checkpoints for the corresponding models.

The entire codebase with analysis and pre-trained models can be found at: https://drive.google.com/drive/folders/1zSLZiTWNfrtXnfGQ6rzs4QmnIeIhZJni?usp=sharing.


