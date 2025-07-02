
import cv2

import matplotlib.pyplot as plt


# Load the image

image = cv2.imread('original_images/example.jpg')


# Convert to RGB for displaying with matplotlib

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# Get image dimensions

height, width, _ = image.shape


# Add bi-directional arrows for width visualization

arrow_start_left = (20, height - 50)  # Left starting point

arrow_end_right = (width - 20, height - 50)  # Right end point


# Draw arrows in both directions

cv2.arrowedLine(image_rgb, arrow_start_left, arrow_end_right, (255, 0, 0), 3, tipLength=0.05)  # Rightward arrow

cv2.arrowedLine(image_rgb, arrow_end_right, arrow_start_left, (255, 0, 0), 3, tipLength=0.05)  # Leftward arrow


# Annotate the width value

width_label_position = (width // 2 - 100, height - 80)  # Center position slightly above the arrows

cv2.putText(image_rgb, f'Width: {width}px', width_label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)


# Save the annotated image

cv2.imwrite('output_images/annotated_width.jpg', image_rgb)


# Display the annotated image

plt.imshow(cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB))

plt.title("Annotated Image with Bi-Directional Width Arrows")

plt.axis('off')

plt.show()

