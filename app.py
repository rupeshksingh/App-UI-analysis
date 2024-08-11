import streamlit as st
import os
import cv2
import numpy as np
import base64
from langchain.chains import TransformChain
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.runnables import chain
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Literal
# from api_key import OPENAI_API_KEY

# Setup OpenAI API key
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

def load_image(inputs: dict) -> dict:
    """Load image from file and encode it as base64."""
    image1_path = inputs["image1_path"]
    image2_path = inputs["image2_path"]

    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def pixelwise_color_difference(img1, img2):
        if img1.shape != img2.shape:
            raise ValueError("Images must have the same dimensions")
        diff = cv2.absdiff(img1, img2)
        return diff

    def generate_heatmap(diff):
        heatmap = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
        return heatmap

    image1_base64 = encode_image(image1_path)
    image2_base64 = encode_image(image2_path)

    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)

    diff = pixelwise_color_difference(img1, img2)
    heatmap = generate_heatmap(diff)

    _, diff_img_encoded = cv2.imencode('.png', diff)
    _, heatmap_encoded = cv2.imencode('.png', heatmap)

    diff_base64 = base64.b64encode(diff_img_encoded).decode('utf-8')
    heatmap_base64 = base64.b64encode(heatmap_encoded).decode('utf-8')

    return {
        "image1": image1_base64,
        "image2": image2_base64,
        "diff_image": diff_base64,
        "heatmap_image": heatmap_base64
    }

load_image_chain = TransformChain(
    input_variables=["image1_path", "image2_path"],
    output_variables=["image1", "image2", "diff_image", "heatmap_image"],
    transform=load_image
)

class ImageInformation(BaseModel):
    action: Literal["tap", "scroll", "type", "no change"] = Field(description="Step that will lead to a change in UI of App from image 1 to image 2. If the images are exactly the same then output no change.")
    item: str = Field(description="which item on the app UI of image 1 the action is taken to change it to image 2: if tap is the action then the element name, if type is the action then what is typed and if scroll then return empty string")

@chain
def image_model(inputs: dict) -> str | list[str] | dict:
    model = ChatOpenAI(temperature=0, model="gpt-4o", max_tokens=1024)
    msg = model.invoke(
        [HumanMessage(
            content=[
                {"type": "text", "text": inputs["prompt"]},
                {"type": "text", "text": parser.get_format_instructions()},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{inputs['image1']}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{inputs['image2']}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{inputs['diff_image']}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{inputs['heatmap_image']}"}},
            ]
        )]
    )
    return msg.content

parser = JsonOutputParser(pydantic_object=ImageInformation)

def get_image_informations(image1_path: str, image2_path: str) -> dict:
    vision_prompt = """
    Act as a UI/UX expert and a software testing engineer.
    Given the 2 App UI image, one before step and one after the step provide the following information:
    Also you are given 2 additional processed image of the first 2 UI image. So the third image is the pixel wise difference of the UI images and the 4th image is the heatmap of the difference image.
    Analyze the following first two UI states and identify the single action that transforms UI 1 into UI 2.
    The action can be one of the following: tap, scroll, or type.

    1. Tap: This action involves interacting with a clickable item. It can be as simple as Clicking a whishlist button resulting in its color change or shape change, or a complete change of page.
    2. Scroll: This action involves scrolling, resulting in misalignment of elements in UI 2 compared to UI 1. The misalignment can be in both horizontal or vertical direction.
    3. Type: This action involves typing into a text box or input area. This can be identified if the second UI image has sum text input and the first UI has the same input space as empty.

    Think step by step before answering and analyze both UI image use your knowledge of UI and see what both image describe and what are the differences Then answer.
    Take help from the 3rd and 4th image to answer. The 3rd image is the pixel wise difference of the UI images and the 4th image is the heatmap of the difference image.
    """
    vision_chain = load_image_chain | image_model | parser
    return vision_chain.invoke({'image1_path': f'{image1_path}',
                                'image2_path': f'{image2_path}',
                                'prompt': vision_prompt})

# Streamlit App Interface

st.title("App UI Change Detection")
st.write("Upload two images of the App UI before and after a user action, and get the detected action and item.")

uploaded_file1 = st.file_uploader("Choose the first image", type=["png", "jpg", "jpeg", "bmp", "tiff", "gif"])
uploaded_file2 = st.file_uploader("Choose the second image", type=["png", "jpg", "jpeg", "bmp", "tiff", "gif"])

if uploaded_file1 and uploaded_file2:
    # Save uploaded images
    with open("image1.png", "wb") as f:
        f.write(uploaded_file1.getbuffer())
    with open("image2.png", "wb") as f:
        f.write(uploaded_file2.getbuffer())

    # Process images and get the result
    result = get_image_informations("image1.png", "image2.png")

    # Display the uploaded images
    st.image(uploaded_file1, caption='First Image')
    st.image(uploaded_file2, caption='Second Image')

    # st.image(diff_image, caption='Difference Image')
    # st.image(heatmap_image, caption='Heatmap Image')

    # Display the action and item
    st.write("### Detected Action:")
    st.write(result.get("action"))

    st.write("### Detected Item:")
    st.write(result.get("item"))
