import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, Label, Button, filedialog, Frame
from PIL import Image, ImageTk

def preprocess_and_detect(image_path):
    # Step 1: Load and preprocess the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Unable to open image at {image_path}")
        return None, None

    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    result_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Initialize variables
    discontinuity_threshold = 400
    min_threshold = 200
    numbBoxes = 0

    while True:
        # Step 2: Edge Detection
        edges = cv2.Canny(blurred_image, 50, 150)

        # Step 3: Find contours in the edge-detected image
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Step 4: Analyze Gaps in Contours and Draw Bounding Boxes
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)

            for i in range(len(approx) - 1):
                pt1 = approx[i][0]
                pt2 = approx[i + 1][0]
                distance = np.linalg.norm(pt1 - pt2)

                if distance > discontinuity_threshold:
                    x_min = min(pt1[0], pt2[0])
                    y_min = min(pt1[1], pt2[1])
                    x_max = max(pt1[0], pt2[0])
                    y_max = max(pt1[1], pt2[1])

                    padding = 5
                    x_min = max(0, x_min - padding)
                    y_min = max(0, y_min - padding)
                    x_max = min(image.shape[1], x_max + padding)
                    y_max = min(image.shape[0], y_max + padding)

                    cv2.rectangle(result_image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                    numbBoxes += 1

        if numbBoxes >= 2:
            print(f"Fractures detected with threshold: {discontinuity_threshold}")
            break

        discontinuity_threshold -= 10
        if discontinuity_threshold < min_threshold:
            print("No fractures detected.")
            break

    return image, result_image

def upload_image():
    # File dialog to choose an image
    file_path = filedialog.askopenfilename()
    if file_path:
        original, result = preprocess_and_detect(file_path)
        if original is not None and result is not None:
            display_images(original, result)

def display_images(original, result):
    # Convert OpenCV images to PIL format for Tkinter
    original_image = Image.fromarray(original)
    result_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

    # Resize images to fit in the window
    original_image.thumbnail((400, 400))  # Resize to fit better
    result_image.thumbnail((400, 400))  # Resize to fit better

    # Convert images to Tkinter format
    original_image_tk = ImageTk.PhotoImage(original_image)
    result_image_tk = ImageTk.PhotoImage(result_image)

    # Display the images in the labels
    original_label.config(image=original_image_tk)
    original_label.image = original_image_tk
    result_label.config(image=result_image_tk)
    result_label.image = result_image_tk

# Set up the Tkinter window with an increased scale and friendly layout
root = Tk()
root.title("Bone Fracture Detection")
root.geometry("1000x600")  # Set the window size (width x height)
root.resizable(False, False)  # Make the window fixed in size

# Create a frame for layout
frame = Frame(root, padx=20, pady=20)
frame.pack(expand=True)

# Title label
title_label = Label(frame, text="Bone Fracture Detection System", font=("Arial", 24, "bold"))
title_label.grid(row=0, column=0, columnspan=2, pady=20)

# Create labels to hold the images
original_label = Label(frame, text="Original Image", font=("Arial", 16))
original_label.grid(row=1, column=0, padx=20, pady=20)

result_label = Label(frame, text="Processed Image with Bounding Boxes", font=("Arial", 16))
result_label.grid(row=1, column=1, padx=20, pady=20)

# Add a button to upload an image
upload_button = Button(frame, text="Upload X-ray Image", font=("Arial", 14), command=upload_image)
upload_button.grid(row=2, column=0, columnspan=2, pady=20)

# Add instructions label
instructions_label = Label(frame, text="Select an X-ray image to detect bone fractures.", font=("Arial", 12))
instructions_label.grid(row=3, column=0, columnspan=2)

# Start the Tkinter event loop
root.mainloop()
