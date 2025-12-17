import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
import pandas as pd

# Function to load the dataset
def load_dataset(file_path):
    """
    Load the dataset from the given file path.

    Args:
        file_path (str): The path to the CSV file containing the dataset.

    Returns:
        pd.DataFrame: The loaded dataset as a Pandas DataFrame.
    """
    df = pd.read_csv(file_path)
    return df

# Function to extract features from the food image using ResNet-50
def extract_features(img):
    """
    Extract features from an image using a pre-trained ResNet-50 model.

    Args:
        img (PIL.Image.Image): The input image.

    Returns:
        torch.Tensor: The extracted features.
    """
    model = models.resnet50(pretrained=True)  # Load ResNet-50 with ImageNet weights
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(img)
    image = image.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        features = model(image)
    return features

# Main function to perform image-to-recipe generation
def image_to_recipe(img_name, df):
    """
    Perform image-to-recipe generation by matching the food item from the dataset and generating a recipe.

    Args:
        img_name (str): The name of the uploaded image file.
        df (pd.DataFrame): The dataset containing food items and ingredients.

    Returns:
        tuple: The matched food item name, ingredients, and instructions, or an error message.
    """
    try:
        # Remove file extension from image name
        img_name_no_ext = img_name.rsplit('.', 1)[0]
        
        # Check if the image name (without extension) exists in the dataset
        if img_name_no_ext in df['Image_Name'].values:
            matched_food_item = df[df['Image_Name'] == img_name_no_ext].iloc[0]
            food_item_name = matched_food_item['Title']
            ingredients = matched_food_item['Cleaned_Ingredients']
            instructions = matched_food_item['Instructions']
        else:
            return None, None, "Image name not found in the dataset"
        
        return food_item_name, ingredients, instructions
    except Exception as e:
        return None, None, f"Error generating recipe: {e}"

# Streamlit app
def main():
    """
    Streamlit app to upload an image and generate a recipe based on the uploaded image.
    """
    st.title('Image to Recipe Generator')

    # Load the dataset
    file_path = "Food Ingredients and Recipe Dataset with Image Name Mapping.csv"
    df = load_dataset(file_path)

    # File uploader
    uploaded_file = st.file_uploader("Upload a food image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption='Uploaded food image.', use_column_width=True)

        # Get the image name
        img_name = uploaded_file.name

        # Button to generate recipe
        if st.button('Generate Recipe'):
            # Generate recipe
            food_item_name, ingredients, instructions = image_to_recipe(img_name, df)

            # Display generated recipe
            if food_item_name:
                st.subheader('Matched Food Item:')
                st.write(food_item_name)
                
                st.subheader('Ingredients:')
                st.write(ingredients)

                st.subheader('Instructions:')
                st.write(instructions)
            else:
                st.subheader('Failed to generate recipe:')
                st.write(instructions)  # Display the error message

if __name__ == '__main__':
    main()
