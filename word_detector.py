# import cv2
# import matplotlib.pyplot as plt
#
# # Load the image
# image = cv2.imread("cropped_images/cropped_0.jpg")
#
# # Define rectangle coordinates
# start_point = [1,1]
# end_point = [65, 58]
# d = int(1280/50)
# for i in range(1):
#     if i% 2==0:
#         color = (255, 0, 0)
#     else:
#         color = (255, 0, 0)
#     thickness = 2
#     cv2.rectangle(image, start_point, end_point, color, thickness)
#     start_point[1] = start_point[1]+43
#     end_point[1] = end_point[1]+43
#
# # Convert BGR to RGB for correct color display in matplotlib
# image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
# # Display using matplotlib
# plt.imshow(image_rgb)
# plt.axis("off")  # Hide axes
# plt.show()
#


import cv2

# Load the image
image = cv2.imread("handwriting_image_2.jpg")

# Define new dimensions (width, height)
new_size = (1003, 1280)  # Resize to 300x300 pixels

# Resize the image
resized_image = cv2.resize(image, new_size)

# Save the resized image
cv2.imwrite("resized_image.jpg", resized_image)

print("Image resized and saved as 'resized_image.jpg'")
