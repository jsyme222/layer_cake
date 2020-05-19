#! usr/bin/python3.8
import numpy as np
import os
import re
from PIL import Image


def normalize_rgba(value):
    value = int(round(value))
    val = 0
    if 0 < value < 255:
        val = value
    if 255 < value:
        val = 255

    return val


def shaded(c, p):
    """
    Returns the rgba value for the color (c) shifted to a given percent (p)
    Alpha is 0-255
    :param c: str(r, g, b, a)
    :param p: int(<percent-to-shift>)
    :return: str(r, g, b, a)
    """
    cleaned = [normalize_rgba(x) for x in c[:-1]]
    r, g, b = cleaned
    R = r * (100 + p) / 100
    G = g * (100 + p) / 100
    B = b * (100 + p) / 100
    R, G, B = [normalize_rgba(x) for x in [R, G, B]]
    rgba = (R, G, B, 255)
    return rgba


class Decorator:
    """
    The decorator class applies gradients to png images by copying the image and coloring
    each pixel accordingly. Can map multi-color gradients based upon percentage or map
    a single color gradient to any greyscale image.
    """

    def __init__(self, image, style=None):
        """
        Creates an Image object from the supplied image path if Image object is not supplied.
        Style can be str() of css rules creating a gradient, or a tuple() containing the rgba
        values of a single color.
        :param image: str() | <Image>
        :param style: str() | tuple(r,g ,b, a)
        """
        try:
            self.image = Image.open(image)
        except ValueError:
            self.image = image
        self.final_image = ""
        self.style = style

    def create_pixel_map(self):
        pixel_map = self.image.load()

        img = Image.new(self.image.mode, self.image.size)
        pixels_new = img.load()
        for i in range(img.size[0]):
            for j in range(img.size[1]):
                if 0 in pixel_map[i, j]:
                    pixels_new[i, j] = (0, 0, 0, 255)
                else:
                    pixels_new[i, j] = pixel_map[i, j]

        img_2 = Image.new(img.mode, img.size)
        img_final = img_2.load()
        img.close()

        return img_2, img_final, pixels_new

    def create_gradient_from_css(self):
        try:
            # Clean the css data, *hopefully* any non-css is caught here
            # to verify style type
            css = self.style
            rules = css.split(';')
            final_css = rules[-2]
            final_gradient = []
            first_clean = final_css.split('rgba')[1:]
        except (AttributeError, ValueError) as e:
            print("Not a gradient style;\nContinuing;")
            return False

        for value in first_clean:
            value = value.split('\n')[0]
            x = re.split('(\\))', value)
            rgb = " ".join(x[:2])
            perc = int((x[2].split("%")[0].lstrip()).split('.')[0])
            rgb = rgb.split(',')
            new_rgb = []
            for num in rgb:
                num = re.sub('[^0-9]', '', num)
                new_rgb.append(num)
            final_gradient.append(
                (
                    (
                        int(new_rgb[0]),
                        int(new_rgb[1]),
                        int(new_rgb[2]),
                        255),
                    perc if perc > 0 else 1
                )
            )
        return final_gradient

    def apply_gradient(self):
        img_2, img_final, pixels_new = self.create_pixel_map()
        gradient_values = self.create_gradient_from_css()
        if gradient_values:
            for i in range(img_2.size[0]):
                for j in range(img_2.size[1]):
                    if pixels_new[i, j] != (0, 0, 0, 255):
                        pix = min(pixels_new[i, j])
                        index = 0
                        while index <= len(gradient_values):
                            try:
                                next_point = gradient_values[index + 1][1]
                            except IndexError:
                                next_point = 255
                            print(gradient_values[index], index)
                            if gradient_values[index][1] < pix < next_point:
                                img_final[i, j] = gradient_values[index][0]
                            index += 1

                    else:
                        if pixels_new[i, j] == (0, 0, 0, 255):
                            img_final[i, j] = (0, 0, 0, 0)
                        else:
                            img_final[i, j] = pixels_new[i, j]

        img_2.show()
        self.final_image = img_final
        return img_2

    def create_color_layer(self, color=None):

        if not color and self.style:
            color = self.style

        img_2, img_final, pixels_new = self.create_pixel_map()

        for i in range(img_2.size[0]):
            for j in range(img_2.size[1]):
                if pixels_new[i, j] != (0, 0, 0, 255):
                    pix = min(pixels_new[i, j])
                    if pix < 30:
                        pix = -pix
                    if 35 < pix:
                        print("found")
                        pix = int(100 * (pix / 255))
                        print(shaded(color, pix))
                        img_final[i, j] = shaded(color, pix)
                    else:
                        img_final[i, j] = shaded(pixels_new[i, j], 40)
                else:
                    if pixels_new[i, j] == (0, 0, 0, 255):
                        img_final[i, j] = (0, 0, 0, 0)
                    else:
                        img_final[i, j] = pixels_new[i, j]

        img_2.show()
        self.final_image = img_2
        return img_2

    def decorate(self):
        if not self.create_gradient_from_css():
            self.create_color_layer(self.style)
        else:
            self.apply_gradient()

    def close(self):
        self.final_image.close()

    def save(self, path="", title=""):
        self.image.close()
        if not path:
            path = os.getcwd()
        if not os.path.isdir(path):
            os.mkdir(path)
        img_path = os.path.join(path + (title or "baked-layer") + ".png")
        self.final_image.save(img_path, 'PNG')

        return img_path


class Cake:
    """The cake class takes in the layers of a png and
    manipulates the image accordingly"""

    def __init__(self, layers):
        """Layers will be a dictionary containing the
        layer position (key:int => 0 being base) and the layer image
        (value:PIL.PngImagePlugin.PngImageFile)
        :param layers:
        """
        self.layers = layers
        self.baked_image = None
        self.thumbnail = None

    def create_thumbnail(self, image):
        image.load()
        image_data = np.asarray(image)
        image_data_bw = image_data.take(3, axis=2)
        non_empty_columns = np.where(image_data_bw.max(axis=0) > 0)[0]
        non_empty_rows = np.where(image_data_bw.max(axis=1) > 0)[0]
        crop_box = (
            min(non_empty_rows),
            max(non_empty_rows),
            min(non_empty_columns),
            max(non_empty_columns)
        )
        image_data_new = image_data[crop_box[0]:crop_box[1] + 1, crop_box[2]:crop_box[3] + 1, :]

        new_image = Image.fromarray(image_data_new)
        new_image.crop(crop_box)
        new_image.show()
        return new_image

    def bake(self):

        layers = self.layers
        while (layer_length := len(layers.keys())) > 1:
            # Combine each layer starting from the highest ordered
            current_layer = layer_length - 1
            base_layer = layers[current_layer - 1]
            new_layer = layers[current_layer]
            base_layer.paste(new_layer, (0, 0), new_layer)
            layers[current_layer - 1] = base_layer
            layers.pop(current_layer)
        final_image = layers[0]
        final_image.format = "PNG"
        self.baked_image = final_image
        self.thumbnail = self.create_thumbnail(final_image)

        return True

    def decorate(self, layer, style):
        d = Decorator(layer)

    def show(self):
        if self.baked_image:
            self.baked_image.show()
            return True
        else:
            print("No image has been baked yet")
            return False

    def save(self, path="", title=""):
        """
        Saves the baked image in the directory specified by path with the title given
        :param path:
        :param title:
        :return:
        """
        if not path:
            path = os.getcwd()
        if not os.path.isdir(path):
            os.mkdir(path)
        img_path = os.path.join(path + (title or "baked-image") + ".png")
        thumbnail_path = os.path.join(path, ((title or "baked-image") + "_thumbnail") + ".png")
        self.baked_image.save(img_path, 'PNG')
        self.thumbnail.save(thumbnail_path, 'PNG')

        return img_path, thumbnail_path


CSS = '''background: -webkit-linear-gradient(top,
    rgba(28, 0, 4, 1) 0%,
    rgba(83, 0, 8, 1) 22.1923828125%,
    rgba(128, 0, 31, 1) 37.255859375%,
    rgba(187, 38, 61, 1) 58.6181640625%,
    rgba(224, 67, 91, 1) 94.7998046875%
  );
  background: -moz-linear-gradient(top,
    rgba(28, 0, 4, 1) 0%,
    rgba(83, 0, 8, 1) 22.1923828125%,
    rgba(128, 0, 31, 1) 37.255859375%,
    rgba(187, 38, 61, 1) 58.6181640625%,
    rgba(224, 67, 91, 1) 94.7998046875%
  );
  background: -o-linear-gradient(top,
    rgba(28, 0, 4, 1) 0%,
    rgba(83, 0, 8, 1) 22.1923828125%,
    rgba(128, 0, 31, 1) 37.255859375%,
    rgba(187, 38, 61, 1) 58.6181640625%,
    rgba(224, 67, 91, 1) 94.7998046875%
  );
  background: -ms-linear-gradient(top,
    rgba(28, 0, 4, 1) 0%,
    rgba(83, 0, 8, 1) 22.1923828125%,
    rgba(128, 0, 31, 1) 37.255859375%,
    rgba(187, 38, 61, 1) 58.6181640625%,
    rgba(224, 67, 91, 1) 94.7998046875%
  );
  linear-gradient(to bottom,
    rgba(28, 0, 4, 1) 0%,
    rgba(83, 0, 8, 1) 22.1923828125%,
    rgba(128, 0, 31, 1) 37.255859375%,
    rgba(187, 38, 61, 1) 58.6181640625%,
    rgba(224, 67, 91, 1) 94.7998046875%
  );'''

if __name__ == '__main__':
    layer1 = 'images/demo/layer1.png'
    green = (64, 66, 99, 255)
    d = Decorator(layer1, CSS)
    d.decorate()
