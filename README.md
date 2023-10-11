# Canny-Edge-Detector

This document provides instructions on how to run the Canny Edge Detection code and includes a working example with comments to run other examples.

1. Folder Structure:
   Ensure that your project folder has the following structure:
   - YourProjectFolder/
     - YourCodeFile.py
     - InputImages/
       - sample_image.jpg
     - GeneratedImages/

2. Running the Code:
   To run the code, follow these steps:
   - Open your command-line terminal or IDE.
   - Navigate to the directory where your Python script is located:
     ```
     cd /path/to/YourProjectFolder
     ```
   - Run your Python script:
     ```
     python YourCodeFile.py
     ```

3. Working Example:
   To run the code for a working example, follow these steps:

   - In your Python script (YourCodeFile.py), find the following section:
     ```python
     # Main function call for the sample image
     main(sigma=1, T=0.3, input_image_path=input_image_path)
     ```

   - Replace 'input_image_path' with the path to your sample image. For example:
     ```python
     input_image_path = 'InputImages/your_sample_image.jpg'
     main(sigma=1, T=0.3, input_image_path=input_image_path)
     ```

4. Viewing Results:
   After running the code, you can view the generated images in the 'GeneratedImages' folder. The code will save various images with different operations applied, including edge detection, non-maximum suppression, and hysteresis thresholding.

   The generated images include:
   - Original Grayscale Image
   - Result of Convolution w.r.t x
   - Result of Convolution w.r.t y
   - Result after applying Magnitude to the image
   - Result after scaling down the image
   - Direction Matrix
   - Result of Non-Maximum Suppression
   - Result of Hysteresis Thresholding

5. Running Other Examples:
   To run additional examples, you can modify the 'input_image_path' and other parameters in your Python script. For example, you can change the input image and sigma values as needed.

   - Example 2: Change the 'input_image_path' to another image and run the main function for that image.
   - Example 3: Modify the 'sigma' value and 'input_image_path' to run edge detection with different settings.
   - Add more examples by repeating the above steps.

Enjoy experimenting with different images and parameters to observe the Canny edge detection results for various scenarios.

