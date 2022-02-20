from abc import ABC, abstractmethod
import cv2 as cv
import numpy as np

from ..utils import disp


class BaseImage(ABC):
    
    color_names = ["blue", "green", "red"]
    
    def __init__(self, filepath):
        
        self.img = cv.imread(filepath)
        self.remove_text_area()
        
        if self.img.shape != self.expected_shape:
            raise ValueError(f"Expected image shape {self.expected_shape}, got {self.img.shape}")
        
        self.grey = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
        self.color_channels = {name: channel for name, channel in zip(self.color_names, cv.split(self.img))}
        self.color_means, self.color_sds = cv.meanStdDev(self.img)
        self.color_means = self.color_means.flatten().round(3)
        self.color_sds = self.color_sds.flatten().round(3)
        assert self.color_means.shape == (3,)
        
        self.blur = round(cv.Laplacian(self.grey, cv.CV_64F).var(), 3)
        self.edges = cv.Canny(self.img, 100, 200)
        
    @property
    @abstractmethod
    def expected_shape(self):
        pass
    
    @property
    @abstractmethod
    def background_color(self):
        pass
    
    @property
    @abstractmethod
    def text_area(self):
        pass
    
    @text_area.setter
    @abstractmethod
    def text_area(self):
        pass
    
    def remove_text_area(self):
        self.text_area = np.full_like(self.text_area, self.background_color, dtype="uint8")
    
    def get_pure_colors(self):
        thres = {k: cv.threshold(v, 250, 255, cv.THRESH_BINARY)[1] for k, v in self.color_channels.items()}
        pure_blue = cv.bitwise_and(thres["blue"], cv.bitwise_not(cv.bitwise_or(thres["red"], thres["green"])))
        pure_red = cv.bitwise_and(thres["red"], cv.bitwise_not(cv.bitwise_or(thres["blue"], thres["green"])))
        pure_green = cv.bitwise_and(thres["green"], cv.bitwise_not(cv.bitwise_or(thres["red"], thres["blue"])))
        return pure_blue, pure_green, pure_red
    
    def remove_color_channel(self, rm):
        assert rm in self.color_names
        img = cv.merge([self.color_channels[k] if k != rm else np.zeros_like(self.color_channels[k]) for k in self.color_names])
    
    def show(self):
        disp(self.img)


class AnglesImage(BaseImage):
    
    def __init__(self, filepath):
        super().__init__(filepath)
    
    @property
    def background_color(self):
        return 255
    
    @property
    def expected_shape(self):
        return (484, 1210, 3)
    
    @property
    def text_area(self):
        return self.img[2:20, 192:850, :]
    
    @text_area.setter
    def text_area(self, value):
        self.img[2:20, 192:850, :] = value


class AtlasImage(BaseImage):
    def __init__(self, filepath):
        super().__init__(filepath)
        
    @property
    def background_color(self):
        return 255
    
    @property
    def expected_shape(self):
        return (968, 1210, 3)
    
    @property
    def text_area(self):
        return self.img[2:20, 275:780, :]
    
    @text_area.setter
    def text_area(self, value):
        self.img[2:20, 275:780, :] = value


class ClaspImage(BaseImage):
    pass


class GradientImage(BaseImage):
    
    def __init__(self, filepath):
        super().__init__(filepath)
    
    @property
    def background_color(self):
        return 255
    
    @property
    def expected_shape(self):
        return (484, 1210, 3)
    
    @property
    def text_area(self):
        return self.img[2:20, 192:860, :]
    
    @text_area.setter
    def text_area(self, value):
        self.img[2:20, 192:860, :] = value


class LaplaceImage(BaseImage):
    
    def __init__(self, filepath):
        super().__init__(filepath)
    
    @property
    def background_color(self):
        return 255
    
    @property
    def expected_shape(self):
        return (484, 1210, 3)
    
    @property
    def text_area(self):
        return self.img[2:20, 192:780, :]
    
    @text_area.setter
    def text_area(self, value):
        self.img[2:20, 192:780, :] = value


class SurfSurfImage(BaseImage):
    
    def __init__(self, filepath):
        super().__init__(filepath)
        
    @property
    def background_color(self):
        return 255
    
    @property
    def expected_shape(self):
        return (484, 1210, 3)
    
    @property
    def text_area(self):
        return self.img[2:20, 350:760, :]
    
    @text_area.setter
    def text_area(self, value):
        self.img[2:20, 350:760, :] = value


class VerifyImage(BaseImage):
    pass
