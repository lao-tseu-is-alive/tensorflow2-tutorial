from my_tf_lib import images_classification as ic

CONST_MODEL_PATH = 'trained_models/tf2_model_cnn_flowers_with_data_augmentation_classifier'
CONST_NEW_IMAGES = 'unseen_test_samples/flowers'
CONST_IMAGE_SIZE = (180, 180)
# Normalisation is not needed here , because it is integrated in the first layer of model :
# layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
CONST_IMAGE_RESCALE = 1.0

CONST_CLASS_NAMES = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

if __name__ == '__main__':
    print('# REUSING A SIMPLE CNN model trained with CIFAR-10')
    print(" # Flowers categories : \n{}".format(CONST_CLASS_NAMES))
    ic.make_images_predictions_from_model(
        path_to_images=CONST_NEW_IMAGES,
        path_to_model=CONST_MODEL_PATH,
        class_list=CONST_CLASS_NAMES,
        image_size=CONST_IMAGE_SIZE,
        normalization_factor=CONST_IMAGE_RESCALE
    )
