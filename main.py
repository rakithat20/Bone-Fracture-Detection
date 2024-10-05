import cv2
import numpy as np  # Import numpy for pi
from img import Img

# Load the image in grayscale
image = Img(r"C:\Users\mathe\Desktop\works\Project\Bone-Fracture-Detection\broken-leg-xray.jpg", 'gray')
img = image.getImg()

# Show the original image
cv2.imshow("Original Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Adjust brightness/contrast
alpha = -1.6
beta = 0
adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
cv2.imshow("Brightness/Contrast Adjusted", adjusted)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Apply Gaussian Blur to reduce noise
blurred = cv2.GaussianBlur(adjusted, (5, 5), 0)
cv2.imshow("Blurred Image", blurred)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Equalize histogram for better contrast
equalized = cv2.equalizeHist(blurred)
cv2.imshow("Equalized Image", equalized)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Edge detection using Canny
edges = cv2.Canny(equalized, threshold1=50, threshold2=150)
cv2.imshow("Edge Detection", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Apply morphological operations to close gaps in edges
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
morphed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
cv2.imshow("Morphological Operations", morphed)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Use Hough Line Transform to detect lines (potential fractures)
lines = cv2.HoughLinesP(morphed, rho=1, theta=np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)  # Use np.pi

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

cv2.imshow("Detected Lines", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Highlight the fracture by drawing rectangles around detected lines
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Draw rectangles to mark possible fracture locations
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow("Fracture Highlight", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
