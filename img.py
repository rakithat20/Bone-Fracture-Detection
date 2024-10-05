import sys;
import cv2 as cv

class Img :

    def __init__(self,path,color):
        self.path = path;
        self.color = color;
        if(color == 'gray'):
            self.src = cv.imread(path,cv.IMREAD_GRAYSCALE)
        else:
            self.src = cv.imread(path)

    def getImg(self):
        return self.src





