import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Function to count the number of pixels in the shape
def count_shape_pixels(image_file):
    # Convert the uploaded image to a format that OpenCV can read
    image = Image.open(image_file)  # Open the image with PIL
    image = np.array(image)  # Convert the PIL image to a NumPy array
    
    # Convert the image to grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Threshold the image to separate the shape from the background
    _, binary = cv2.threshold(image_gray, 1, 255, cv2.THRESH_BINARY)
    
    # Count nonzero pixels (which belong to the shape)
    pixel_count = np.count_nonzero(binary)
    
    return pixel_count, binary, image.shape[:2]  # Return pixel count, binary image, and image shape

# Function to calculate the missing dimension (width or height) based on the known one
def calculate_missing_dimension(image_shape, known_dimension, is_height_known=True, dpi=300):
    height_pixels, width_pixels = image_shape[:2]
    
    # Calculate the aspect ratio (width / height)
    aspect_ratio = width_pixels / height_pixels
    
    if is_height_known:
        # Calculate the width using the aspect ratio
        known_dimension_inch = known_dimension / 2.54  # Convert cm to inches
        known_dimension_pixels = known_dimension_inch * dpi
        missing_dimension_pixels = known_dimension_pixels * aspect_ratio
        missing_dimension_cm = missing_dimension_pixels / dpi * 2.54  # Convert back to cm
        return missing_dimension_cm
    else:
        # Calculate the height using the aspect ratio
        known_dimension_inch = known_dimension / 2.54  # Convert cm to inches
        known_dimension_pixels = known_dimension_inch * dpi
        missing_dimension_pixels = known_dimension_pixels / aspect_ratio
        missing_dimension_cm = missing_dimension_pixels / dpi * 2.54  # Convert back to cm
        return missing_dimension_cm

# Function to calculate the area of one pixel in square centimeters
def area_of_one_pixel(height_pixels, width_pixels, height_cm, width_cm):
    # Calculate the height and width of one pixel in centimeters
    height_per_pixel = height_cm / height_pixels
    width_per_pixel = width_cm / width_pixels
    
    # Calculate the area of one pixel in square centimeters
    area_pixel_cm2 = height_per_pixel * width_per_pixel
    return area_pixel_cm2

# Streamlit UI
def main():
    st.title("Image Pixel Counter and Area Calculator")
    st.write("Upload an image, and the program will count the number of pixels in the shape and calculate the area of the shape in square centimeters.")
    
    # Allow user to upload an image
    uploaded_image = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])
    
    if uploaded_image is not None:
        # Display the uploaded image
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        # User inputs for height or width
        height_input = st.number_input("Enter height in cm (leave empty if you want to input width)", value=0.0, min_value=0.0)
        width_input = st.number_input("Enter width in cm (leave empty if you want to input height)", value=0.0, min_value=0.0)

        # Ensure at least one dimension is provided
        if height_input == 0.0 and width_input == 0.0:
            st.error("Please provide either the height or width in cm.")
            return

        # Process the image and calculate pixel count
        st.write("Processing image...")
        pixel_count, binary_image, image_shape = count_shape_pixels(uploaded_image)
        
        height_cm = width_cm = None  # Initialize the height and width variables
        
        if height_input > 0.0:
            # Calculate the missing width if height is provided
            width_cm = calculate_missing_dimension(image_shape, height_input, is_height_known=True)
            st.write(f"Calculated width: {width_cm:.2f} cm")
            height_cm = height_input  # Use the provided height value
        else:
            # Calculate the missing height if width is provided
            height_cm = calculate_missing_dimension(image_shape, width_input, is_height_known=False)
            st.write(f"Calculated height: {height_cm:.2f} cm")
            width_cm = width_input  # Use the provided width value
        
        # Calculate the area of one pixel in square centimeters
        pixel_area_cm2 = area_of_one_pixel(image_shape[0], image_shape[1], height_cm, width_cm)
        
        # Calculate the total area of the shape
        shape_area_cm2 = pixel_count * pixel_area_cm2

        # Output the results
        st.write(f"Number of pixels in the shape: {pixel_count}")
        st.write(f"Area of 1 pixel: {pixel_area_cm2:.10f} cm²")
        st.write(f"Total area of the shape: {shape_area_cm2:.2f} cm²")

# Run the Streamlit app
if __name__ == "__main__":
    main()
