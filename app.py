import streamlit as st
from PIL import Image
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import logging
from skimage.metrics import structural_similarity as ssim

logging.basicConfig(level=logging.INFO)

# Function to extract hierarchy details from an XML tree
def extract_hierarchy_details(tree):
    def traverse_node(node, level=0):
        hierarchy = []
        hierarchy.append({
            'level': level,
            'tag': node.tag,
            'num_attributes': len(node.attrib),
            'num_children': len(node),
        })
        for child in node:
            hierarchy.extend(traverse_node(child, level + 1))
        return hierarchy

    root = tree.getroot()
    return traverse_node(root)

# Function to compare two XML hierarchies
def compare_xml_hierarchy(tree1, tree2):
    hierarchy1 = extract_hierarchy_details(tree1)
    hierarchy2 = extract_hierarchy_details(tree2)

    if len(hierarchy1) != len(hierarchy2):
        logging.info(f"XML files have different numbers of nodes: {len(hierarchy1)} vs {len(hierarchy2)}")
    
    mismatch_found = False
    
    for idx, (node1, node2) in enumerate(zip(hierarchy1, hierarchy2)):
        if node1['level'] != node2['level']:
            logging.info(f"XML hierarchy mismatch at index {idx}: Different nesting levels detected.")
            mismatch_found = True
        if node1['tag'] != node2['tag']:
            logging.info(f"XML hierarchy mismatch at index {idx}: Different tags detected - '{node1['tag']}' vs '{node2['tag']}'.")
            mismatch_found = True
        if node1['num_attributes'] != node2['num_attributes']:
            logging.info(f"XML hierarchy mismatch at index {idx}: Different number of attributes for tag '{node1['tag']}' at level {node1['level']}.")
            mismatch_found = True
        if node1['num_children'] != node2['num_children']:
            logging.info(f"XML hierarchy mismatch at index {idx}: Different number of children for tag '{node1['tag']}' at level {node1['level']}.")
            mismatch_found = True

    if len(hierarchy1) > len(hierarchy2):
        for remaining_node in hierarchy1[len(hierarchy2):]:
            logging.info(f"Extra node in first XML: Tag '{remaining_node['tag']}' at level {remaining_node['level']}'.")
            mismatch_found = True
    elif len(hierarchy2) > len(hierarchy1):
        for remaining_node in hierarchy2[len(hierarchy1):]:
            logging.info(f"Extra node in second XML: Tag '{remaining_node['tag']}' at level {remaining_node['level']}'.")
            mismatch_found = True

    if not mismatch_found:
        logging.info("XML hierarchies match.")
        return True
    else:
        logging.info("XML hierarchies do not match.")
        return False

# Parsing XML for bounding boxes
def parse_xml_for_bounding_boxes(tree):
    root = tree.getroot()
    bounding_boxes = []

    for elem in root.iter():
        bounds = elem.attrib.get("bounds")
        if bounds:
            x1y1, x2y2 = bounds.split("][")
            x1, y1 = map(int, x1y1[1:].split(","))
            x2, y2 = map(int, x2y2[:-1].split(","))
            bounding_boxes.append((x1, y1, x2, y2))

    unique_bounding_boxes = list(set(bounding_boxes))
    return unique_bounding_boxes

# Cropping images using bounding boxes
def crop_image(image, bounding_boxes):
    cropped_images_with_coords = []
    for box in bounding_boxes:
        cropped_image = image.crop(box)
        cropped_images_with_coords.append((cropped_image, box))

    return cropped_images_with_coords

# Image resizing for SSIM
def resize_images(img1, img2, min_size=(7, 7)):
    size = (
        max(min(img1.width, img2.width), min_size[0]),
        max(min(img1.height, img2.height), min_size[1]),
    )
    img1_resized = img1.resize(size, Image.LANCZOS)
    img2_resized = img2.resize(size, Image.LANCZOS)

    return img1_resized, img2_resized

# Enlarging small images
def enlarge_if_small(img, min_size=(7, 7), enlarge_size=(10, 10)):
    if img.width < min_size[0] or img.height < min_size[1]:
        img = img.resize(enlarge_size, Image.NEAREST)
    return img

# Compare images using SSIM
def compare_images_ssim(img1, img2, threshold):
    img1 = enlarge_if_small(img1)
    img2 = enlarge_if_small(img2)
    img1_resized, img2_resized = resize_images(img1, img2)
    img1_array = np.array(img1_resized)
    img2_array = np.array(img2_resized)

    total_similarity = 0.0

    for channel in range(3):
        channel_similarity, _ = ssim(
            img1_array[:, :, channel], img2_array[:, :, channel], full=True
        )
        total_similarity += channel_similarity

    average_similarity = total_similarity / 3.0
    logging.info(f"SSIM similarity: {average_similarity}")
    return average_similarity >= threshold

# Compare images using histogram
def compare_images_histogram(img1, img2, threshold):
    img1_cv = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2BGR)
    img2_cv = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2BGR)

    hist_img1 = cv2.calcHist(
        [img1_cv], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256]
    )
    hist_img2 = cv2.calcHist(
        [img2_cv], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256]
    )

    hist_img1[255, 255, 255] = 0
    hist_img2[255, 255, 255] = 0

    cv2.normalize(hist_img1, hist_img1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist_img2, hist_img2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    metric_val = cv2.compareHist(hist_img1, hist_img2, cv2.HISTCMP_CORREL)
    logging.info(f"Histogram similarity: {metric_val}")
    return metric_val >= threshold

# Combined comparison of images using SSIM and histogram
def compare_images(img1, img2, ssim_threshold, hist_threshold):
    ssim_result = compare_images_ssim(img1, img2, ssim_threshold)
    hist_result = compare_images_histogram(img1, img2, hist_threshold)

    return ssim_result and hist_result

# Compare and analyze individual components
def compare_components(cropped_images1, cropped_images2, ssim_threshold, hist_threshold):
    num_components1 = len(cropped_images1)
    num_components2 = len(cropped_images2)
    diff = abs(num_components1 - num_components2)

    vertical_scroll_count = 0
    horizontal_scroll_count = 0
    same_count = 0
    different_count = 0

    if num_components1 <= num_components2:
        smaller_components = cropped_images1
        larger_components = cropped_images2
        smaller_size = num_components1
        larger_size = num_components2
    else:
        smaller_components = cropped_images2
        larger_components = cropped_images1
        smaller_size = num_components2
        larger_size = num_components1

    for i in range(smaller_size):
        for j in range(i, min(i + diff + 1, larger_size)):
            img1, coords1 = smaller_components[i]
            img2, coords2 = larger_components[j]

            if compare_images(img1, img2, ssim_threshold, hist_threshold):
                x_match = coords1[0] == coords2[0] and coords1[2] == coords2[2]
                y_match = coords1[1] == coords2[1] and coords1[3] == coords2[3]

                if x_match and not y_match:
                    logging.info("Matching x coordinates but different y coordinates.")
                    vertical_scroll_count += 1
                elif y_match and not x_match:
                    logging.info("Matching y coordinates but different x coordinates.")
                    horizontal_scroll_count += 1
                elif x_match and y_match:
                    logging.info("Matching both x and y coordinates.")
                    same_count += 1
                else:
                    logging.info("No matching coordinates.")
                    different_count += 1

    categories = {
        "Vertical Scroll": vertical_scroll_count,
        "Horizontal Scroll": horizontal_scroll_count,
        "Same Components": same_count,
        "Different": different_count,
    }
    dominant_category = max(categories, key=categories.get)
    return dominant_category, categories

# Main processing function
def main():
    st.title("XML and Image Comparison Tool")
    
    st.header("Upload Files")
    xml_file1 = st.file_uploader("Upload first XML file", type="xml")
    image_file1 = st.file_uploader("Upload first Image file", type=["png", "jpg", "jpeg"])
    xml_file2 = st.file_uploader("Upload second XML file", type="xml")
    image_file2 = st.file_uploader("Upload second Image file", type=["png", "jpg", "jpeg"])
    
    ssim_threshold = st.slider("SSIM Threshold", 0.0, 1.0, 0.5)
    hist_threshold = st.slider("Histogram Threshold", 0.0, 1.0, 0.5)
    
    if st.button("Compare"):
        if xml_file1 and xml_file2 and image_file1 and image_file2:
            try:
                # Step 1: Parse XML files
                tree1 = ET.parse(xml_file1)
                tree2 = ET.parse(xml_file2)

                # Step 2: Check XML hierarchy
                st.header("XML Hierarchy Comparison")
                xml_hierarchy_match = compare_xml_hierarchy(tree1, tree2)
                if xml_hierarchy_match:
                    st.success("XML hierarchies match.")
                else:
                    st.warning("XML hierarchies do not match.")
                    return

                # Step 3: Check overall image similarity
                st.header("Overall Image Similarity")
                image1 = Image.open(image_file1)
                image2 = Image.open(image_file2)

                if compare_images(image1, image2, ssim_threshold, hist_threshold):
                    st.success("The images are similar.")
                    return
                else:
                    st.warning("The images are not similar.")

                # Step 4: Extract bounding boxes and crop images
                bounding_boxes1 = parse_xml_for_bounding_boxes(tree1)
                bounding_boxes2 = parse_xml_for_bounding_boxes(tree2)

                cropped_images_with_coords1 = crop_image(image1, bounding_boxes1)
                cropped_images_with_coords2 = crop_image(image2, bounding_boxes2)

                # Step 5: Compare components
                st.header("Component-wise Image Comparison")
                dominant_category, categories = compare_components(cropped_images_with_coords1, cropped_images_with_coords2, ssim_threshold, hist_threshold)
                st.write(f"Dominant category: {dominant_category}")
                st.write(categories)

            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("Please upload all required files.")

if __name__ == "__main__":
    main()
