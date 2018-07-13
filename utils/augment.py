import random
import numpy as np
import tensorflow as tf


class Augment(object):

    def __init__(self):
        self.sess = tf.Session()

    def augment(self, images, width, height, operations):
        augmented = []
        for i in range(len(images)):
            aug = self.randomCrop(images[i], width, height)
            if 'brightness' in operations:
                aug = self.random_brightness(image=aug,
                                             max_delta=0.2)
            if 'contrast' in operations:
                aug = self.random_contrast(image=aug,
                                           lower=0.2,
                                           upper=1.8)
            if 'hue' in operations:
                aug = self.random_hue(image=aug,
                                      max_delta=0.2)
            if 'flip_h' in operations:
                aug = self.random_flip_h(image=aug)

            augmented.append(aug)

        print('> Augmented images with {}'.format(operations), sep=' ')
        return self.sess.run(augmented)

    def randomCrop(self, img, width, height):
        x = random.randint(0, img.shape[1] - width)
        y = random.randint(0, img.shape[0] - height)
        return img[y:y+height, x:x+width]

    def randomCropAll(self, imgs, width, height):
        imgs_l = []
        for i in range(imgs.shape[0]):
            imgs_l.append(self.randomCrop(imgs[i], width, height))
        return np.array(imgs_l)

    def random_hue(self, image, max_delta=0.5):
        return tf.image.random_hue(image=image,
                                   max_delta=max_delta)

    def random_contrast(self, image, lower, upper):
        return tf.image.random_contrast(image=image,
                                        lower=lower,
                                        upper=upper)

    def random_brightness(self, image, max_delta=0.5):
        return tf.image.random_brightness(image=image,
                                          max_delta=max_delta)

    def random_flip_h(self, image):
        return tf.image.random_flip_left_right(image=image)
