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

### 05 Classify images of cats and dogs using transfer learning
this one is about trying to classify images of cats and dogs by using transfer learning 
from a pre-trained network :  The MobileNet V2 model developed at Google !

we also save the fitted model to try it on a set of unseen images.
First run the training (and the let do the fine-tuning of the model) :

    python3 05_cnn_transfer_learning_mobilenet-v2_classifier.py

Then we try to reuse it on a set of unseen images.  

    python3 05_cnn_transfer_learning_mobilenet-v2_classifier_reuse_with_new_images.py 
    # REUSING A Mobile Net V2  model modified with transfer learning on cats and dogs 
    # Cats and dogs categories are obviously : 
    ['cats', 'dogs']
    # Tensorflow version : 2.4.0
    # Loading model already fitted.
    ### will try to load model from path : trained_models/tf2_model_cnn_transfer_learning_mobilenet-v2_with_data_augmentation_classifier
    ### Now trying model.predict with a brand new set of images !
    # found 27 images in path : unseen_test_samples/cats_dogs 
    # ✅ ✅ prediction for cats01.jpeg is CORRECT  0 = cats       
    # ✅ ✅ prediction for cats02.jpeg is CORRECT  0 = cats       
    # ✅ ✅ prediction for cats03.jpeg is CORRECT  0 = cats       
    # ✅ ✅ prediction for cats04.jpeg is CORRECT  0 = cats       
    # ✅ ✅ prediction for cats05.jpeg is CORRECT  0 = cats       
    # ✅ ✅ prediction for cats06.jpeg is CORRECT  0 = cats       
    # ✅ ✅ prediction for cats07.jpeg is CORRECT  0 = cats       
    # ❌ ❌  prediction for cats08.jpeg is  WRONG   1 = dogs      
    # ✅ ✅ prediction for cats09.jpeg is CORRECT  0 = cats       
    # ✅ ✅ prediction for cats10.jpeg is CORRECT  0 = cats       
    # ✅ ✅ prediction for cats11.jpeg is CORRECT  0 = cats       
    # ✅ ✅ prediction for cats12.jpeg is CORRECT  0 = cats       
    # ✅ ✅ prediction for cats13.jpeg is CORRECT  0 = cats       
    # ✅ ✅ prediction for dogs01.jpeg is CORRECT  1 = dogs       
    # ✅ ✅ prediction for dogs02.jpeg is CORRECT  1 = dogs       
    # ✅ ✅ prediction for dogs03.jpeg is CORRECT  1 = dogs       
    # ✅ ✅ prediction for dogs04.jpeg is CORRECT  1 = dogs       
    # ✅ ✅ prediction for dogs05.jpeg is CORRECT  1 = dogs       
    # ✅ ✅ prediction for dogs06.jpeg is CORRECT  1 = dogs       
    # ✅ ✅ prediction for dogs07.jpeg is CORRECT  1 = dogs       
    # ✅ ✅ prediction for dogs08.jpeg is CORRECT  1 = dogs       
    # ✅ ✅ prediction for dogs09.jpeg is CORRECT  1 = dogs       
    # ✅ ✅ prediction for dogs10.jpeg is CORRECT  1 = dogs       
    # ✅ ✅ prediction for dogs11.jpeg is CORRECT  1 = dogs       
    # ✅ ✅ prediction for dogs12.jpeg is CORRECT  1 = dogs       
    # ✅ ✅ prediction for dogs13.jpeg is CORRECT  1 = dogs       
    # ✅ ✅ prediction for dogs14.jpeg is CORRECT  1 = dogs       
    ================================================================================
    26 CORRECT PREDICTIONS, 1 WRONG PREDICTIONS
    96.30% percent success !
