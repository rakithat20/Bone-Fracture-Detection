import cv2
from img import Img

image = Img(r"C:\Users\mathe\Desktop\works\Project\Bone-Fracture-Detection\broken-leg-xray.jpg",'gray')
img = image.getImg()

cv2.imshow("Display window",img)
alpha = -1.6
beta = 0

adjusted = cv2.convertScaleAbs(img,alpha=alpha,beta=beta)
cv2.imshow("Display window",adjusted)
k = cv2.waitKey(0)
