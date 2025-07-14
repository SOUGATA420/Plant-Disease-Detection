import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PIL import Image
from src.pipeline import PlantDiseasePipeline

pipeline = PlantDiseasePipeline()

# Use raw string or forward slashes
image = Image.open(r"C:\Users\Sougata Maity\Downloads\download.jpg")

result = pipeline.run(image)
print(result)
