import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the images
image = cv2.imread(r"C:\Users\ASUS\Pictures\hand2.jpg", cv2.IMREAD_GRAYSCALE)

# Step 1: Preprocess the image
blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

# Create a copy of the original image to draw the results
result_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

# Initialize variables
discontinuity_threshold = 400  # Initial threshold for detecting gaps
min_threshold = 200  # Minimum allowed threshold value

while True:
    # Step 2: Edge Detection
    edges = cv2.Canny(blurred_image, 50, 150)

    # Step 3: Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    numbBoxes = 0  # Reset the box count for each iteration

    # Step 4: Analyze Gaps in Contours and Draw Bounding Boxes
    for contour in contours:
        # Approximate the contour with a polygon
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)

        # Check for large gaps between points in the approximated contour
        for i in range(len(approx) - 1):
            pt1 = approx[i][0]
            pt2 = approx[i + 1][0]
            distance = np.linalg.norm(pt1 - pt2)

            if distance > discontinuity_threshold:
                # Step 5: Draw a bounding box around the detected discontinuity
                x_min = min(pt1[0], pt2[0])
                y_min = min(pt1[1], pt2[1])
                x_max = max(pt1[0], pt2[0])
                y_max = max(pt1[1], pt2[1])

                # Expand the bounding box slightly for better visibility
                padding = 5
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(image.shape[1], x_max + padding)
                y_max = min(image.shape[0], y_max + padding)

                # Draw the bounding box on the result image
                cv2.rectangle(result_image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                numbBoxes += 1

    # Check the number of detected boxes
    if numbBoxes >= 2:
        print(discontinuity_threshold)
        print("/nFractures detected!")
        break  # Exit the loop if sufficient boxes are found

    # Reduce the threshold for the next iteration
    discontinuity_threshold -= 10
    if discontinuity_threshold < min_threshold:
        print("No fractures detected.")
        break  # Exit the loop if the threshold becomes too low

# Display the original image and result with bounding boxes
plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')

plt.subplot(1, 2, 2)
plt.title("Detected Discontinuities with Bounding Boxes")
plt.imshow(result_image)
plt.show()
