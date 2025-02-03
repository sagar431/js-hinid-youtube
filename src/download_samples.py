import os
import requests
from pathlib import Path

def download_image(url, filename):
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.content)
        return True
    return False

def main():
    # Create samples directory
    samples_dir = Path("samples")
    samples_dir.mkdir(exist_ok=True)
    
    # Sample images URLs
    images = {
        "cat1.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg",
        "cat2.jpg": "https://cdn.britannica.com/39/7139-050-A88818BB/Himalayan-chocolate-point.jpg",
        "dog1.jpg": "https://cdn.britannica.com/79/232779-050-6B0411D7/German-Shepherd-dog-Alsatian.jpg",
        "dog2.jpg": "https://cdn.britannica.com/16/234216-050-C66F8665/beagle-hound-dog.jpg"
    }
    
    for filename, url in images.items():
        filepath = samples_dir / filename
        if download_image(url, filepath):
            print(f"Successfully downloaded {filename}")
        else:
            print(f"Failed to download {filename}")

if __name__ == "__main__":
    main() 