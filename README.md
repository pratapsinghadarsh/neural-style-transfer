
# Neural Style Transfer
Neural Style Transfer (NST) is a technique that allows us to combine the content of an image with the style of another image.
NST can be used for a variety of applications including art creation, photo editing, and video processing.

The algorithm transfers style from one input image (the style image) onto another input image (the content image) using CNN nets (usually VGG-16/19) and gives a composite, stylized image out which keeps the content from the content image but takes the style from the style image.

How do we generate an Image in Neural Style Transfer?
In normal problems(i.e. Classification, Regression etc), we make a model(set of operations and operands(weights,biases and data) and train the model(weights and biases) to fit the distribution of the data), but in Neural Style Transfer, we take pixel values of the image as our weights and biases and train the image(i.e. change pixel values during training) and generate the Image. 

You can access all the details about this project such as style and content, gram matrix, losses and type of losses in the pdf provided.





## Overview

- NST: Training an Image

- Style and Content Images

- Feature extraction from style and content images

- Gram Matrix

- Loss Function for NST (Style Loss and Content Loss)

- Generate Image




## Installation

This code is written on google colab

```bash
 1. All the codes and images used are given 
 2. Download everything in your machine using git clone
 3. Upload code file and images on google colab and you are good to go
```
    
# Libraries used in this Project

## 1.  Numpy:

NumPy can be used to perform a wide variety of mathematical operations on arrays and it is almost 50 times faster than python lists. In our project we used NumPy to covert tensor into image.

## 2. Tensorflow:
TensorFlow is a free and open-source software library for machine learning and artificial intelligence.
TensorFlow can be used to develop models for various tasks, including natural language processing, image recognition, handwriting recognition, and different computational-based simulations such as partial differential equations. Tensorflow library provides us mny features such as converting an image into array.
it also provides us pretrained cnn models such as VGG.

### VGG:
VGG stands for Visual Geometry Group; it is a standard deep Convolutional Neural Network (CNN) architecture with multiple layers.
We used VGG19 in this project. VGG-19 is a convolutional neural network that is 19 layers deep.The pretrained network can classify images into 1000 object categories, such as keyboard, mouse, pencil, and many animals.

## 3. OS:
The OS module in Python provides functions for interacting with the operating system. OS comes under Python's standard utility modules. This module provides a portable way of using operating system-dependent functionality

## 4. CV2:
OpenCV-Python covers a broad spectrum of computer vision and image processing tasks. It provides functions for image and video manipulation, feature detection, object recognition, and Machine Learning.  It can process images and videos to identify objects, faces, or even the handwriting of a human. When it is integrated with various libraries, such as Numpy which is a highly optimized library for numerical operations, then the number of weapons increases in your Arsenal i.e whatever operations one can do in Numpy can be combined with OpenCV.

## 5. PIL:
Python Imaging Library (PIL) is a free and open-source additional library for the Python programming language that adds support for opening, manipulating, and saving many different image file formats
## Usage

You can choose any content image and any style image.
Each time you choose a new photo run all the code cells to avoid any error.

Final photo after the style transfer will automatically get saved in your drive. Intermediate photos will also get saved depending upon no. of epochs you are using. If you don't have gpu available use smaller no. of epochs.


## Examples
Transfering style gives beautiful artistic results:

![1 (1)](https://github.com/pratapsinghadarsh/neural-style-transfer/assets/139372823/f06ac4a6-be50-478f-bd18-e3ff6e53cae4)
![frida_modern_art](https://github.com/pratapsinghadarsh/neural-style-transfer/assets/139372823/f736b226-9b27-400a-bbae-6551ec710234)

You can access the details of this project in the pdf provided.


## Authors

- [Adarsh Pratap Singh](https://www.github.com/pratapsinghadarsh), IIT Roorkee -  If you have any query or suggestions do let me know at adarsh_ps@ch.iitr.ac.in



