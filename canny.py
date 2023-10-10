import numpy as np 
from math import * # We need to apply log and squareroot
import cv2 # To convert the image in 2D and then to save image
import matplotlib.pyplot as plt # To plot and show the results after applying the functions
import os

scale = 255 #Value to scale up the image
sigma = [1,0.5,2]

#This function is to calculate the filter size
def calculate_filter_size (sigma = 0.0 , T = 0.0):
  sHalf = round(sqrt(-log(T) * 2 * sigma**2))
  N = 2 * sHalf + 1
  print("sHalf:", sHalf, "N:", N)
  Y, X = np.meshgrid(np.arange(-sHalf, sHalf+1), np.arange(-sHalf, sHalf+1))
  return [Y,X]

#This function is to calculate the gradient w.r.t x and y respectively
def calculate_gradient (x, y, sigma = 0.0):
  fx = (-x/(sigma**2)) * (np.exp (-(x**2 + y**2)/(2*sigma**2)))
  fy = (-y/(sigma**2)) *(np.exp (-(x**2 + y**2)/(2*sigma**2)))
  return (np.round(fx*scale) , np.round(fy*scale))

# This function is to change the image from RGB to gray_scale
def convert_to_grayscale(input_image_path):
    image = cv2.imread(input_image_path)
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return grayscale_image

#This function is for convolution
def convolution(image, kernel):
    # Get the dimensions of the image and kernel
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
# Output size = [(Input size - Filter size + 2 * Padding) / Stride] + 1
# As we using 0 padding and taking the jump of 1 and 1 on both sides (stride)
    # Calculate the output dimensions
    output_height = image_height - kernel_height + 1
    output_width = image_width - kernel_width + 1

    # Initialize array with zeros
    output = np.zeros((output_height, output_width))

    # Perform the convolution
    for i in range(output_height):
        for j in range(output_width):
            output[i, j] = np.sum(image[i:i+kernel_height, j:j+kernel_width] * kernel)

    return output

#This function is for applying Non-Maximun Supperession
def non_maximum_suppression(magnitude, direction):
    suppressed_edges = np.zeros_like(magnitude, dtype=np.uint8)
    neighbor1 = 0
    neighbor2 = 0
    for r in range(1, magnitude.shape[0] - 1):
        for c in range(1, magnitude.shape[1] - 1):
            mag = magnitude[r, c]
            angle = direction[r, c]

            if 0 <= angle < 22.5 or 157.5 <= angle <= 202.5 or 337.5 <= angle <= 360:
                neighbor1 = magnitude[r, c + 1]
                neighbor2 = magnitude[r, c - 1]
            elif 22.5 <= angle < 67.5 or 202.5 <= angle <= 247.5:
                neighbor1 = magnitude[r - 1, c + 1]
                neighbor2 = magnitude[r + 1, c - 1]
            elif 67.5 <= angle < 112.5 or 247.5  <= angle <= 292.5:
                neighbor1 = magnitude[r - 1, c]
                neighbor2 = magnitude[r + 1, c]
            elif 112.5  <= angle < 157.5 or 292.5  <= angle <= 337.5:
                neighbor1 = magnitude[r - 1, c - 1]
                neighbor2 = magnitude[r + 1, c + 1]

            if mag >= neighbor1 and mag >= neighbor2:
                suppressed_edges[r, c] = mag

    return suppressed_edges

#This function is to apply the thresholding
def hysteresis_thresholding(magnitude, high_threshold, low_threshold):
    height, width = magnitude.shape
    edge_map = np.zeros((height, width), dtype=bool)
    visited = np.zeros((height, width), dtype=bool)

    neighbor_offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)] # Define 8-offsets for neighbors

    stack = [] # Create a stack to store pixels

    for r in range(height):
        for c in range(width):
            if not visited[r, c] and magnitude[r, c] >= high_threshold:
                stack.append((r, c))

                while stack:
                    current_r, current_c = stack.pop()

                    if visited[current_r, current_c]:
                        continue

                    visited[current_r, current_c] = True
                    edge_map[current_r, current_c] = True

                    for dr, dc in neighbor_offsets:
                        nr, nc = current_r + dr, current_c + dc

                        if 0 <= nr < height and 0 <= nc < width and not visited[nr, nc] and magnitude[nr, nc] >= low_threshold:
                            stack.append((nr, nc))

    return edge_map

def main(sigma=1, T=0.3, input_image_path='/Users/alimujtaba/Documents/ITU/Computer Vision/Assignment 1/Circle/circle.jpg'):
    image_name = input_image_path.split('/')[-1][:-4]

    #These lines store the values of filter and gradient
    filter = calculate_filter_size(sigma,T)
    result = calculate_gradient (filter[1],filter[0],sigma)
    
    #Image change into grayscale  
    grey_img = convert_to_grayscale(input_image_path)

    # Perform the convolution
    result_fx = convolution(grey_img, result[0])
    result_fy = convolution(grey_img, result[1])

    # The magnitude formula which is applied on derivatives w.r.t x and y respectively
    magnitude = np.sqrt(result_fx**2 + result_fy**2)
    print(f'Magnitude of the Image is: {magnitude}')

    # Display the original grayscale image by using the library matplotlib
    plt.subplot(1, 5, 1)
    plt.imshow(grey_img, cmap='gray')
    plt.title('Original Grayscale Image')

    # Display the result of the convolution w.r.t x by using the library matplotlib
    plt.subplot(1, 5, 2)
    plt.imshow(result_fx, cmap='gray')
    plt.title('Result of Convolution w.r.t x')

    # Display the result of the convolution w.r.t y by using the library matplotlib
    plt.subplot(1, 5, 3)
    plt.imshow(result_fy, cmap='gray')
    plt.title('Result of Convolution w.r.t y')

    # Display the result of magnitude by using the library matplotlib
    plt.subplot(1, 5, 4)
    plt.imshow(magnitude, cmap='gray')
    plt.title('Result after apply Magnitude to image')

    #Scale down the image and the show the difference by scaling up and scaling down the images
    plt.subplot(1, 5, 5)
    plt.imshow(magnitude/scale, cmap='gray')
    plt.title('Result after scale down the image')
    plt.show()

    #Save images in same directory in which file is running
    cv2.imwrite(f'Generated Images/{image_name}_fx_{sigma}.jpg',result_fx)
    cv2.imwrite(f'Generated Images/{image_name}_fy_{sigma}.jpg',result_fy)
    cv2.imwrite(f'Generated Images/{image_name}_magnitude_{sigma}.jpg',magnitude)
    cv2.imwrite(f'Generated Images/{image_name}_scale_down_magnitude_{sigma}.jpg',magnitude/scale)

    #This is the direction formula against each point to see it's direction
    direction = np.arctan(result_fy/result_fx)
    print(f'Direction Matrix of the image is: {direction}')
    plt.imshow(direction, cmap='gray')
    plt.title('Result of Directions')
    plt.show()

    # This Magnitude and Direction is computed from result_fx and result_fy.
    result_edges_1 = non_maximum_suppression(magnitude, direction)
    print (f'After applying NMS the result is: {result_edges_1}')
    plt.imshow(result_edges_1, cmap='gray')
    plt.title('Result of NMS')
    plt.show()

    cv2.imwrite(f'Generated Images/{image_name}_quantized_{sigma}.jpg',result_edges_1)



    # Set the high and low thresholds.
    high_threshold = 20
    low_threshold = 10

    edge_map = hysteresis_thresholding(magnitude, high_threshold, low_threshold)
    print (f'After applying the Hysterisis Threshold the matrix is: {edge_map}')

    if sigma == 1:
        high_threshold = 25
        low_threshold = 5
        edge_map1 = hysteresis_thresholding(magnitude, high_threshold, low_threshold)
        plt.imshow(edge_map1, cmap='gray')
        plt.title('Result of Hysterisis Threshold for sigma = 1')
        plt.show()
    
    plt.imshow(edge_map, cmap='gray')
    plt.title('Result of Hysterisis Threshold')
    plt.show()


main_path ='/Users/alimujtaba/Documents/ITU/Computer Vision/Canny_Edge_Detector/Input images'
for path in os.listdir(main_path):
    for s in sigma:
        main( sigma=s , input_image_path=f'{main_path}/{path}')