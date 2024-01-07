import requests
from bs4 import BeautifulSoup
import os

# Function to download passport images from a website
def download_passport_images(url, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Send a GET request to the URL
    response = requests.get(url)
    
    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find image elements on the webpage
    image_tags = soup.find_all('img')
    
    # Counter for naming the downloaded images
    img_counter = 1
    
    # Download and save each image
    for img_tag in image_tags:
        img_url = img_tag.get('src')  # Extract image URL
        if img_url:
            img_data = requests.get(img_url).content
            with open(f'{output_folder}/passport_{img_counter}.jpg', 'wb') as f:
                f.write(img_data)
            img_counter += 1

# URL of the website containing passport images
website_url = 'https://example.com/passport_images'

# Output folder to store downloaded images
output_folder = 'datasets/passport_images'

# Call the function to download passport images
download_passport_images(website_url, output_folder)
