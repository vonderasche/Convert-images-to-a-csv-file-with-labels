# convert images and labels to csv format

import numpy as np
import os
import csv
import cv2
import matplotlib.pyplot as plt


class CreateDataset(object):
    """Converts images into csv format"""

    def __init__(self, path, color_channels, img_dim):
        """ """
        self.path = path
        self.color_channels = color_channels
        self.img_dim = img_dim
        self.loaded_images = []
        self.loaded_labels = []

    def load_data(self):
        """Load images and path names from a provided path."""

        for root, dirs, files in os.walk(self.path):
            for file in files:
                image_path = (os.path.join(root, file))
                image_path = image_path.split(os.sep)
                self.loaded_labels.append(image_path[-2])
                img = cv2.imread(os.path.join(root, file))
                img = np.asarray(img)
                self.loaded_images.append(img)

        return [self.loaded_images, self.loaded_labels]

    def color_img(self, loaded_images):
        """Set the color to either RGB or BW. """
        if self.color_channels == 1:
            images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                      for img in loaded_images]
        elif self.color_channels == 3:
            images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                      for img in loaded_images]
        return images

    def crop_img(self, loaded_images):
        """Creates a square image using the images shortest side as the new height and width."""

        images = []
        for img in loaded_images:
            if self.color_channels == 1:
                height, width = img.shape
                if (height < width):
                    x_start = int((width - height) / 2)
                    x_end = height + x_start
                    img = img[0:height, x_start:x_end]
                    images.append(img)

                elif (width < height):
                    y_start = int((height - width) / 2)
                    y_end = width + y_start
                    img = img[y_start:y_end, 0:width]
                    images.append(img)

                else:
                    images.append(img)
            if self.color_channels == 3:
                height, width, color_channels = img.shape
                if (height < width):
                    x_start = int((width - height) / 2)
                    x_end = height + x_start
                    img = img[0:height, x_start:x_end]
                    images.append(img)

                elif (width < height):
                    y_start = int((height - width) / 2)
                    y_end = width + y_start
                    img = img[y_start:y_end, 0:width]
                    images.append(img)

                else:
                    images.append(img)
        return images

    def resize_img(self, loaded_images):
        """Resizes the image based on the user specified size."""

        images = [cv2.resize(img, (self.img_dim, self.img_dim))
                  for img in loaded_images]
        return images

    def shape_img(self, loaded_images):
        """Flattens the 2d images into a 1d array."""

        images = [img.flatten() for img in loaded_images]
        return images

    def save_data(self, loaded_images, loaded_labels):
        """Saves the flattened images and the labels to two sperate csv files."""

        np.savetxt('images.csv', loaded_images, delimiter=',', fmt='%i')
        np.savetxt('labels.csv', loaded_labels, delimiter=',',  fmt="%s")

    def print_image_stats(self, images, labels):
        """ Prints stats for the images and labels converted to csv files"""

        num_of_classes = set(labels)
        number_of_classes = len(num_of_classes)
        total_number_of_images = (len(labels))
        print('There are ', total_number_of_images,
              ' images aross ', number_of_classes, ' classes.')
        print('\n')
        print('Classes are as follows:', num_of_classes)

    def test_an_image(self, loaded_images, label_loaded):
        """ Saves a sample set of images as a grid. """

        if self.color_channels == 1:

            fig, ax = plt.subplots(nrows=5, ncols=5,
                                   sharex=True, sharey=True)
            ax = ax.flatten()
            for i in range(25):
                img = loaded_images[i].reshape(self.img_dim, self.img_dim)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                plt.xticks([])
                plt.yticks([])
                ax[i].imshow(img)
            fig.savefig('sample_black.png')
            plt.close(fig)
            plt.show()

        elif self.color_channels == 3:
            fig, ax = plt.subplots(nrows=5, ncols=5,
                                   sharex=True, sharey=True)
            ax = ax.flatten()
            for i in range(25):
                img = loaded_images[i].reshape(self.img_dim, self.img_dim, 3)
                plt.xticks([])
                plt.yticks([])
                ax[i].imshow(img)
            fig.savefig('sample_color.png')
            plt.close(fig)
            plt.show()


def main():
    path = 'E:\PROGRAMMING\DATASETS\RAW_IMAGES'

    dataset = CreateDataset(path, 3, 150)

    x_data, y_data = dataset.load_data()

    x_data = dataset.color_img(x_data)

    x_data = dataset.crop_img(x_data)

    x_data = dataset.resize_img(x_data)

    x_data = dataset.shape_img(x_data)

    dataset.save_data(x_data, y_data)

    dataset.print_image_stats(x_data, y_data)

    dataset.test_an_image(x_data, y_data)


main()
