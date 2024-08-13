import streamlit as st
import cv2
import os
import base64
import hashlib
from langchain.chains import TransformChain
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.runnables import chain
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Literal
from skimage.metrics import structural_similarity as ssim
# from api_key import OPENAI_API_KEY
import sqlite3
import numpy as np

# Setup OpenAI API key
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
# os.environ["OPENAI_API_KEY"] = userdata.get('OPENAI_API_KEY')

# Initialize the database connection
def init_db(db_path="image_cache.db"):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cursor = conn.cursor()

    # Create tables for image 1 and image 2
    cursor.execute('''CREATE TABLE IF NOT EXISTS image_cache1
                      (id INTEGER PRIMARY KEY,
                       image_hash TEXT UNIQUE,
                       image_base64 TEXT)''')
                       
    cursor.execute('''CREATE TABLE IF NOT EXISTS image_cache2
                      (id INTEGER PRIMARY KEY,
                       image_hash TEXT UNIQUE,
                       image_base64 TEXT)''')

    conn.commit()
    return conn, cursor

# Store an image in the database
def store_image(cursor, image_path, table_name):
    with open(image_path, "rb") as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
        image_hash = hashlib.md5(image_base64.encode()).hexdigest()

        cursor.execute(f"INSERT OR IGNORE INTO {table_name} (image_hash, image_base64) VALUES (?, ?)",
                       (image_hash, image_base64))
        cursor.connection.commit()

# Retrieve images from the database
def retrieve_images(cursor, table_name):
    cursor.execute(f"SELECT image_hash, image_base64 FROM {table_name}")
    return cursor.fetchall()

class ImageResultCache:
    def __init__(self, db_conn):
        self.db_conn = db_conn
        self.cursor = db_conn.cursor()
        self.result_cache = {}

    def _hash_image(self, image_path):
        """Create a hash for the image content."""
        with open(image_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    def _are_images_similar(self, img1_base64, img2_base64, threshold=0.95):
        """Use SSIM to check if images are similar."""
        img1 = cv2.imdecode(np.frombuffer(base64.b64decode(img1_base64), np.uint8), cv2.IMREAD_COLOR)
        img2 = cv2.imdecode(np.frombuffer(base64.b64decode(img2_base64), np.uint8), cv2.IMREAD_COLOR)

        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        score, _ = ssim(gray1, gray2, full=True)
        return score >= threshold

    def get_result(self, image_path1, image_path2, process_function):
    # Generate hashes for the image paths
        hash1 = self._hash_image(image_path1)
        hash2 = self._hash_image(image_path2)
        combined_hash = f"{hash1}:{hash2}"

        # Check if the combined hash is already in the result cache
        if combined_hash in self.result_cache:
            st.write("Exact match found in cache")
            return self.result_cache[combined_hash]

        # If not found in the cache, store the images in the database
        store_image(self.cursor, image_path1, 'image_cache1')
        store_image(self.cursor, image_path2, 'image_cache2')

        # Retrieve all images from the database
        cached_images1 = retrieve_images(self.cursor, 'image_cache1')
        cached_images2 = retrieve_images(self.cursor, 'image_cache2')

        img1_base64 = base64.b64encode(open(image_path1, "rb").read()).decode('utf-8')
        img2_base64 = base64.b64encode(open(image_path2, "rb").read()).decode('utf-8')

        # Check for similar images in the database
        for cached_hash1, cached_img1_base64 in cached_images1:
            for cached_hash2, cached_img2_base64 in cached_images2:
                if self._are_images_similar(img1_base64, cached_img1_base64) and \
                self._are_images_similar(img2_base64, cached_img2_base64):
                    combined_hash = f"{cached_hash1}:{cached_hash2}"
                    if combined_hash in self.result_cache:
                        st.write("Similar result found in cache")
                        return self.result_cache[combined_hash]

        # Process the images if no similar result was found
        result = process_function(image_path1, image_path2)
        self.result_cache[combined_hash] = result
        st.write("Result processed and cached.")
        return result

    def get_cache_size(self):
        return len(self.result_cache)

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
    model = ChatOpenAI(temperature=0, model="gpt-4o", max_tokens=1024, api_key=OPENAI_API_KEY)
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

# Initialize the database and cache in Streamlit's session state
if 'db_conn' not in st.session_state:
    db_conn, _ = init_db()
    st.session_state.db_conn = db_conn

if 'cache' not in st.session_state:
    st.session_state.cache = ImageResultCache(st.session_state.db_conn)

def get_image_informations(image1_path: str, image2_path: str, cache: ImageResultCache) -> dict:
    vision_prompt = """
    Act as a UI/UX expert and a software testing engineer.
    Given the 2 App UI image, one before step and one after the step provide the following information:
    Also you are given 2 additional processed image of the first 2 UI image. So the third image is the pixel wise difference of the UI images and the 4th image is the heatmap of the difference image.
    Analyze the following first two UI states and identify the single action that transforms UI 1 into UI 2.
    The action can be one of the following: tap, scroll, or type.

    1. Tap: This action involves interacting with a clickable item. It can be as simple as Clicking a wishlist button resulting in its color change or shape change, or a complete change of page.
    2. Scroll: This action involves scrolling, resulting in misalignment of elements in UI 2 compared to UI 1. The misalignment can be in both horizontal or vertical direction.
    3. Type: This action involves typing into a text box or input area. This can be identified if the second UI image has some text input and the first UI has the same input space as empty.

    Think step by step before answering and analyze both UI images using your knowledge of UI and see what both images describe and what are the differences. Then answer.
    Take help from the 3rd and 4th image to answer. The 3rd image is the pixel-wise difference of the UI images and the 4th image is the heatmap of the difference image.
    Note : There may be case when a small pop up appears in the App UI image, but that does not mean any change, Also some times only the background image changes dynamically but no change in content of that page. Keep all this in mind.
    """

    return cache.get_result(
        image1_path,
        image2_path,
        lambda img1, img2: (load_image_chain | image_model | parser).invoke({
            'image1_path': img1,
            'image2_path': img2,
            'prompt': vision_prompt
        })
    )

# Streamlit App Interface
st.title("App UI Change Detection")
st.write("Upload two images of the App UI before and after a user action, and get the detected action and item.")

uploaded_file1 = st.file_uploader("Choose the first image", type=["png", "jpg", "jpeg", "bmp", "tiff", "gif"])
uploaded_file2 = st.file_uploader("Choose the second image", type=["png", "jpg", "jpeg", "bmp", "tiff", "gif"])

if uploaded_file1 and uploaded_file2:
    with open("image1.png", "wb") as f:
        f.write(uploaded_file1.getbuffer())
    with open("image2.png", "wb") as f:
        f.write(uploaded_file2.getbuffer())

    if st.button("Process Images"):
        result = get_image_informations("image1.png", "image2.png", st.session_state.cache)

        st.image(uploaded_file1, caption='First Image')
        st.image(uploaded_file2, caption='Second Image')

        st.write("### Detected Action:")
        st.write(result.get("action"))

        st.write("### Detected Item:")
        st.write(result.get("item"))

        st.write(f"### Cache Size: {st.session_state.cache.get_cache_size()} entries")
