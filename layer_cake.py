#! usr/bin/python3.8
import numpy as np
import os
import re
from PIL import Image


class Decorator:
    """
    The decorator class was intended to apply gradients but up until this
    point it simply recolors each pixel pf the png according to the supplied
    color to the create_color_layer method
    """

    def __init__(self, image, style=None):
        """
        Creates an Image object from the supplied image path
        :param image: => str() | Image path to image
        :param gradient: => str() css code created by psd gradient uploader
        """
        try:
            self.image = Image.open(image)
        except ValueError:
            self.image = image
        self.final_image = ""
        self.style = style
        self.breakpoint_count = 0
        self.value = 1

    def create_gradient_from_css(self):
        try:
            # Clean the css data, *hopefully* any non-css is caught here
            # to verify style type
            css = self.gradient
            rules = css.split(';')
            final_css = rules[-2]
            final_gradient = []
            first_clean = final_css.split('rgba')[1:]
        except (ValueError, IndexError) as e:
            print("Not a gradient style;\nContinuing;")
            return False

        for value in first_clean:
            value = value.split('\n')[0]
            x = re.split('(\))', value)
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
        self.breakpoint_count = len(final_gradient)
        return final_gradient

    def apply_gradient(self):
        pixels = self.image.load()
        image_new = Image.new(self.image.mode, self.image.size)  # New copied image to return
        pixels_new = image_new.load()
        gradient_values = self.create_gradient_from_css()
        if gradient_values:
            for x in range(len(gradient_values) - 1):
                self.value = gradient_values[x][1]
                rgb, perc = gradient_values[x]
                key = 3 * perc
                for i in range(image_new.size[0]):
                    for j in range(image_new.size[1]):
                        for pix in pixels[i, j]:
                            print(key, self.value)
                            if pix in range(key, self.value):
                                pixels[i, j] = rgb
                            else:
                                pixels_new[i, j] = pixels[i, j]
                            try:
                                self.value = gradient_values[x + 2][1]
                            except IndexError:
                                self.value = 255

        image_new.show()
        image_new.close()

    def create_color_layer(self, color=None):

        def shaded(c, p):
            """
            Returns the rgba value for the color (c) shifted to a given percent (p)
            Alpha is 0-255
            :param c: str() -> (r, g, b, a)
            :param p:
            :return:
            """
            rgba = list(c)
            cleaned = map(lambda x: x if 0 < x else 1, rgba)
            r, g, b, a = cleaned
            R = r * (100 + p) / 100
            G = g * (100 + p) / 100
            B = b * (100 + p) / 100
            R, G, B = map(lambda x: round(x) if x <= 255 else 255, [R, G, B])
            rgba = (R, G, B, a)
            return rgba

        pixelMap = self.image.load()

        img = Image.new(self.image.mode, self.image.size)
        pixelsNew = img.load()
        for i in range(img.size[0]):
            for j in range(img.size[1]):
                if 0 in pixelMap[i, j]:
                    pixelsNew[i, j] = (0, 0, 0, 255)
                else:
                    pixelsNew[i, j] = pixelMap[i, j]

        img_2 = Image.new(img.mode, img.size)
        img_final = img_2.load()

        for i in range(img_2.size[0]):
            for j in range(img_2.size[1]):
                if pixelsNew[i, j] != (0, 0, 0, 255):
                    if 50 < min(pixelsNew[i, j]) <= 60:
                        img_final[i, j] = shaded(color, -50)
                    if 60 < min(pixelsNew[i, j]) <= 70:
                        img_final[i, j] = shaded(color, -30)
                    if 70 < min(pixelsNew[i, j]) <= 80:
                        img_final[i, j] = shaded(color, -20)
                    elif 80 < min(pixelsNew[i, j]) <= 100:
                        img_final[i, j] = color
                    elif 100 < min(pixelsNew[i, j]) <= 140:
                        img_final[i, j] = shaded(color, 10)
                    elif 140 < min(pixelsNew[i, j]) <= 180:
                        img_final[i, j] = shaded(color, 20)
                    elif 180 < min(pixelsNew[i, j]) <= 220:
                        img_final[i, j] = shaded(color, 30)
                    elif 220 < min(pixelsNew[i, j]) <= 255:
                        img_final[i, j] = shaded(color, 40)
                    elif min(pixelsNew[i, j]) > 220:
                        img_final[i, j] = color
                    else:
                        img_final[i, j] = pixelsNew[i, j]
                else:
                    if pixelsNew[i, j] == (0, 0, 0, 255):
                        img_final[i, j] = (0, 0, 0, 0)
                    else:
                        img_final[i, j] = pixelsNew[i, j]

        img.close()
        img_2.show()
        self.final_image = img_2
        return img_2

    def decorate(self):
        if not self.create_gradient_from_css():
            pass
        else:
            self.apply_gradient()
        self.create_color_layer()

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

    def bake(self):

        def create_thumbnail(image):
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
        self.thumbnail = create_thumbnail(final_image)

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
