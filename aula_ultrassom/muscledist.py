import cv2
import math
import argparse
import pandas as pd

# Initialize the list of points and calibration distance
points = []
known_distance = 1.0 # The known distance (in cm) between the first two points

# Initialize a DataFrame to store the measurements
df = pd.DataFrame(columns=['Point 1', 'Point 2', 'Pixel Distance', 'Real Distance (cm)'])

def click_and_calculate_distance(event, x, y, flags, param):
    global points

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

        # If we have two points, calculate the pixel distance for calibration
        if len(points) == 2:
            x1, y1 = points[0]
            x2, y2 = points[1]
            pixel_distance = math.sqrt( ((x1-x2)**2)+((y1-y2)**2) )
            print(f"The pixel distance for calibration is: {pixel_distance} pixels")

        # If we have four points, calculate the real-world distance using calibration
        elif len(points) == 4:
            x1, y1 = points[2]
            x2, y2 = points[3]
            pixel_distance = math.sqrt( ((x1-x2)**2)+((y1-y2)**2) )

            # Calculate scale based on known distance and calibration pixel distance
            scale = known_distance / math.sqrt( ((points[0][0]-points[1][0])**2)+((points[0][1]-points[1][1])**2) )
            real_distance = pixel_distance * scale

            print(f"The real distance between the points is: {real_distance} cm")

            # Add measurement to DataFrame
            df.loc[len(df)] = [points[2], points[3], pixel_distance, real_distance]

            # Reset points
            points = []

# Argument parser
parser = argparse.ArgumentParser(description='Measure distance between two points in an image.')
parser.add_argument('image_path', type=str, help='Path to the image file.')

args = parser.parse_args()

# Load the image
image = cv2.imread(args.image_path)

# Display the image and bind the function to the window
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_calculate_distance)

# Keep the window open until the 'q' key is pressed
while True:
    cv2.imshow("image", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close all open windows
cv2.destroyAllWindows()

# Save the DataFrame to a CSV file
df.to_csv('measurements.csv', index=False)


# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding
_, thresholded = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Display the thresholded image
cv2.imshow("Thresholded Image", thresholded)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Apply thresholding
_, thresholded = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Apply Sobel filter in x direction
sobelx = cv2.Sobel(thresholded, cv2.CV_64F, 1, 0, ksize=5)

# Apply Sobel filter in y direction
sobely = cv2.Sobel(thresholded, cv2.CV_64F, 0, 1, ksize=5)

# Combine Sobel x and y outputs for complete edge detection
edges = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)

# Display the image with edges
cv2.imshow("Edges Image", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
