---
layout: project_base
title: Its all about MSE loss
tags: loss-function machine-learning deep-learning maths-deep-learning
permalink: /writing/mse-loss
---
<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>


Mean square error (MSE) is probably the most widely used loss function in Deep Learning (DL). So much so that it is often blindly put up when doing some fun experiments in DL. But why this? From where does it come from and why does it take the form it does? What's happening when I train my model with MSE loss?


Mean squared loss/error is mathematically defined as $$L_{MSE} = \frac{1}{N} \sum\limits_{i=0}^{N-1}(y_i-x_i)^2




