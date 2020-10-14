---
layout: project_base
title: Recurrent Instance segmentation
tags: segmentation recurrent-neural-network LSTM
permalink: /projects/recurrent-segmentation
thumbnail: https://github.com/SushantDaga/sushantdaga.github.io/blob/posts/assets/images/recurrent-instance-segmentation.png
<!-- publish: false -->
---


Implemented *Recurrent Instance Segmentation* by Bernardino Romera-Paredes et al. as described [here](https://arxiv.org/abs/1511.08250 "arXiv.org") using Keras Framework with TensorFlow backend. The project provided me the opportunity to implement and gauge the following techniques closely:
* [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038 "arXiv.org") by Jonathan Long et al as opposed to traditional Segmentation networks with CNN ending with Dense layer(s)
* Recurrent Neural Network and Attention-based models
* ConvLSTM
* Keras Functional API vs Sequential API (partial towards the former :P)
* Writing your own trainable and non-trainable layers in Keras


<!--more-->

![Recurrent Instance Segmentation]({{ '/assets/images/recurrent-instance-segmentation.png' | relative_url}})
{: style="width: 400px; max-width: 100%;"}