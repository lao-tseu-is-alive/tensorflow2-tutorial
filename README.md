# tensorflow2-tutorial
Warming up for using Deep Learning, with some TensorFlow 2.0 Tutorials

### 01 classify hand digits numbers 
let's start with the classical "hello world !" of machine learning : 
writing a handwritten digits' classification. 
This is possible thanks to the free [MNIST dataset from M. Yann Lecun](http://yann.lecun.com/exdb/mnist/)

you can run this tutorial with:

    python3 01_mnist_classifier.py

### 02 Classify images of clothing
this one is about using Fashion MNIST, we also save the fitted model.

    python3 02_keras_mnist_clothing_classifier.py

Then we try to reuse it on a set of unseen images.  
This will gives us very poor unusable predictions. 
Allowing us to show an important conclusion, 
if the train dataset is strongly preprocessed, as it is the case here.
It will be near to unusable with new images. 
More information is available on this in this [excellent article 
from PyImageSearch from Dr Adrian Rosebrock](https://www.pyimagesearch.com/2019/02/11/fashion-mnist-with-keras-and-deep-learning/).
to quote him :

"...*While the Fashion MNIST dataset is slightly more challenging than the MNIST digit recognition dataset, unfortunately, it cannot be used directly in real-world fashion classification tasks, unless you preprocess your images in the exact same manner as Fashion MNIST (segmentation, thresholding, grayscale conversion, resizing, etc.)*...."


