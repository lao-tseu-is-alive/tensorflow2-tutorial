from my_tf_lib import images_classification as ic

CONST_MODEL_PATH = 'trained_models/tf2_model_cifar10_2xConv2D_MaxPooling2D'
CONST_NEW_IMAGES = 'unseen_test_samples/cifar10'
CONST_IMAGE_SIZE = (32, 32)  # CIFAR-10 FASHION image sizes
CONST_IMAGE_RESCALE = 1. / 255.0  # Normalisation factor for colors values

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

if __name__ == '__main__':
    print('# REUSING A SIMPLE CNN model trained with CIFAR-10')
    print(" # CIFAR-10 categories : \n{}".format(class_names))
    ic.make_images_predictions_from_model(
        path_to_images=CONST_NEW_IMAGES,
        path_to_model=CONST_MODEL_PATH,
        class_list=class_names,
        image_size=CONST_IMAGE_SIZE,
        normalization_factor=CONST_IMAGE_RESCALE
    )
