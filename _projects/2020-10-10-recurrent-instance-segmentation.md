---
layout: project_base
title: Recurrent Instance segmentation
tags: segmentation recurrent-neural-network LSTM
permalink: /projects/recurrent-segmentation
thumbnail: /assets/images/recurrent-instance-segmentation.png
---


Implemented *Recurrent Instance Segmentation* by Bernardino Romera-Paredes et al. as described [here](https://arxiv.org/abs/1511.08250 "arXiv.org") and [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038 "arXiv.org") by Jonathan Long et al. using Keras Framework with TensorFlow backend. The project provided me the opportunity to implement and gauge the following techniques closely:
* Fully Convolutional Networks as opposed to traditional Segmentation networks with CNN ending with Dense layer(s). FCN proved to be independent of the size of input providing ability to use the model in varying picture without adhering to padding. Experimenting and visualizing data manifold with gradient was informative in understanding the role played by activation functions in training and inferenc of a deep model. We experimented on FCN-8 /16 /32 with Relu, Leaky-Relu, Sigmoid, Tanh and dealt with vanishing gradient problem in case of Sigmoid and tanh by experimenting with L1 loss, L2 loss, Batch Normalization, Dropout.
* Attention using Recurrent Neural Network (RNN) - next in line for our experiments. 
* Recurrent Neural Network and Attention-based models
* ConvLSTM
* Keras Functional API vs Sequential API (partial towards the former :P)
* Writing your own trainable and non-trainable layers in Keras


<!--more-->

![Recurrent Instance Segmentation]({{ '/assets/images/recurrent-instance-segmentation.png' | relative_url}})
{: style="width: 400px; max-width: 100%;"}