import cv2

# Load the image
image = cv2.imread("extracted_images/image_2_1.jpeg")

# Define the top-left and bottom-right coordinates of the rectangle
start_point = [159, 60]  # (x, y)
end_point =  [1700, 159]   # (x, y)

# Define the rectangle color (BGR format) and thickness
color = (0, 255, 0)  # Green
thickness = 2  # Line thickness
for i in range(27):

    # Draw the rectangle on the image
    cv2.rectangle(image, start_point, end_point, color, thickness)
    start_point[1] += 93
    end_point[1] += 93

# Display the image
cv2.imshow("Rectangle", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the image (optional)
cv2.imwrite("output.jpg", image)
