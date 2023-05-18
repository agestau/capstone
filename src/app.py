# app.py

import io
import random
from flask import Flask, render_template, send_file
from PIL import Image, ImageDraw, ImageFont
from datasets import load_dataset, Dataset
from transformers import AutoProcessor, BlipForConditionalGeneration, pipeline

app = Flask(__name__)

dataset=load_dataset("agestau/preproc-fashion-products")


def load_model_class():
    model_class = BlipForConditionalGeneration.from_pretrained("agestau/fashion_classification_3")
    return model_class

def load_model_cap_base():
    model_cap_base = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return model_cap_base

def load_model_cap_ft():
    model_cap_ft = BlipForConditionalGeneration.from_pretrained("agestau/f_cap_allrecs")
    return model_cap_ft


def load_pipeline_class():
    pipe_class = pipeline("image-classification", "agestau/fashion_classification_3")
    return pipe_class

def load_pipeline_cap_base():
    pipe_cap_base = pipeline("image-to-text", "Salesforce/blip-image-captioning-base")
    return pipe_cap_base

def load_pipeline_cap_ft():
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
    return render_template('utils/index.html')


@app.route('/random_image')
def get_random_image():

    label, img, text = random_image(dataset)
    img_pil = img.resize((255, 255), resample=Image.LANCZOS) # Making the picture bigger

    # Load the models
    model_class = load_model_class()
    model_cap_ft = load_model_cap_ft()

    # Load the pipelines
    pipeline_class = load_pipeline_class()
    pipeline_cap_ft = load_pipeline_cap_ft()

    # Get the predicted label
    pred_label = pipeline_class(img)[0]['label']

    # Get the predicted caption
    pred_caption = pipeline_cap_ft(img)[0]['generated_text']

    # Calculate the size of the text box
    textbox_width = 400
    textbox_height = img_pil.height

    # Create a new image with space for the text box and the original image
    output_img = Image.new('RGB', (img_pil.width + textbox_width, img_pil.height), color=(255, 255, 255))

    # Paste the image onto the output image
    output_img.paste(img_pil, (textbox_width, 0))

    # Create a draw object for the output image
    draw_output = ImageDraw.Draw(output_img)
    font_size = 12  # Specify the font size

    # Specify the font and text color
    font = ImageFont.load_default()
    text_color = (0, 0, 0)  # Black color for text
    green_color = (0, 128, 0)  # Green color for true values

    # Calculate the y-coordinate for the first line of text
    text_y = (textbox_height - 8 * (font_size + 10)) // 2

    # Draw the white text box
    draw_output.rectangle([(0, 0), (textbox_width, textbox_height)], fill=(255, 255, 255))

    # Draw each line of text inside the text box
    draw_output.text((10, text_y), "True values:", font=font, fill=green_color, bold=True)
    draw_output.text((10, text_y + font_size + 10), f"Label: {label}", font=font, fill=green_color)
    draw_output.text((10, text_y + 2 * (font_size + 10)), f"Caption: {text}", font=font, fill=green_color)
    draw_output.text((10, text_y + 4 * (font_size + 10)), "", font=font, fill=text_color)
    draw_output.text((10, text_y + 5 * (font_size + 10)), "Predicted values:", font=font, fill=text_color, bold=True)
    draw_output.text((10, text_y + 6 * (font_size + 10)), f"Label: {pred_label}", font=font, fill=text_color)
    draw_output.text((10, text_y + 7 * (font_size + 10)), f"Caption: {pred_caption}", font=font, fill=text_color)

    buffer = io.BytesIO()
    output_img.save(buffer, format='PNG')
    buffer.seek(0)

    return send_file(buffer, mimetype='image/png')