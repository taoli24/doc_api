import numpy as np


class GrayScale:
    def __int__(self):
        pass

    def __call__(self, image):
        """
        Get and PIL image or numpy n-dim array as image and convert it to grayscale image

        :param image: input image data
        :return: Grayscale image of input type
        """
        if str(type(image)).__contains__('PIL'):
            image = image.convert('L')
        elif str(type(image)).__contains__('numpy'):
            image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
        else:
            raise ValueError('Input type is not valid.')
        return image
