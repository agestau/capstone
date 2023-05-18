# app.py

import io
import random
#import torch
#import torchvision.transforms as transforms
from flask import Flask, render_template, send_file
from PIL import Image, ImageDraw, ImageFont
from datasets import load_dataset, Dataset
from transformers import AutoProcessor, BlipForConditionalGeneration, pipeline
#from torchvision.datasets import FashionMNIST
#from fashion_MNIST import loaded_cnn_model, class_names
app = Flask(__name__)


def load_data(dataset):
    ds = load_dataset(dataset)
    return ds

def load_model_class():
    model_class = BlipForConditionalGeneration.from_pretrained("agestau/fashion_classification_3")
    return model_class

def load_model_cap_base():
    model_cap_base = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return model_cap_base

def load_model_cap_ft():
    model_cap_ft = BlipForConditionalGeneration.from_pretrained("agestau/f_cap_allrecs")
    return model_cap_ft


def pipeline_class():
    pipe_class = pipeline("image-classification", "agestau/fashion_classification_3")
    return pipe_class

def pipeline_cap_base():
    pipe_cap_base = pipeline("image-to-text", "Salesforce/blip-image-captioning-base")
    return pipe_cap_base

def pipeline_cap_ft():
    processor=AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    pipe_cap_ft = pipeline("image-to-text", "agestau/f_cap_allrecs", tokenizer=processor, image_processor=processor)
    return pipe_cap_ft


def random_image(dataset):
    example_index = random.randint(0, len(dataset["train"]) - 1)
    example = dataset["train"][example_index]
    label, img, text = example["subCategory"], example["image"], example["text"]
    return label, img, text


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/random_image')
def get_random_image():
    
    dataset=load_data("agestau/preproc-fashion-products")

    label, img, text = random_image(dataset)
    
    # Load the models
    model_class = load_model_class()
    model_cap_ft = load_model_cap_ft()

    # Load the pipelines
    pipeline_class = pipeline_class()
    pipeline_cap_ft = pipeline_cap_ft()
    
    # Get the predicted label
    pred_label = pipeline_class(img)[0]['label']

    # Get the predicted caption
    pred_caption = pipeline_cap_ft(img)[0]['generated_text']

    # Add text to the image
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    draw.text((10, 10), f"Predicted label: {pred_label} | True label: {label}\n", font=font, fill=255)
    draw.text((10, 10), f"Predicted caption: {pred_caption}\n", font=font, fill=255)
    draw.text((10, 10), f"True caption: {text}", font=font, fill=255)

    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)

    return send_file(buffer, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
