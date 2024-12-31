# %%
#pip install scikit-image

# %% [markdown]
# Import Library 

# %%
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.color import rgb2gray
from skimage.measure import shannon_entropy

import seaborn as sns

from scipy.stats import norm
from scipy import signal
from scipy import misc
from scipy.ndimage import grey_erosion, grey_dilation, grey_closing, grey_opening
from scipy.ndimage import binary_erosion, binary_dilation, binary_opening, binary_closing

# %% [markdown]
# Image path

# %%
train_image_path = r"C:\Users\david\Desktop\DIP Final\training_dataset\image\\"
train_mask_path = r"C:\Users\david\Desktop\DIP Final\training_dataset\mask\\"
train_output_path = r"C:\Users\david\Desktop\DIP Final\training_dataset\output\\"
num_train_image = 80

test_image_path = r"C:\Users\david\Desktop\DIP Final\testing_dataset\image\\"
test_mask_path = r"C:\Users\david\Desktop\DIP Final\testing_dataset\mask\\"
test_output_path = r"C:\Users\david\Desktop\DIP Final\testing_dataset\output\\" 
num_test_image = 20




# %% [markdown]
# Function for extracting features

# %%
def extract_color_features(image, mask):
    # Apply the mask to isolate water region
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    # Convert to HSV
    hsv = cv2.cvtColor(masked_image, cv2.COLOR_BGR2HSV)

    # Compute mean and standard deviation for each channel in RGB and HSV
    mean_rgb = cv2.mean(image, mask=mask)[:3]
    mean_hsv = cv2.mean(hsv, mask=mask)[:3]
    
    std_rgb = np.std(image[mask > 0], axis=0)
    std_hsv = np.std(hsv[mask > 0], axis=0)

    # Return as a dictionary
    return {
        "mean_rgb": mean_rgb,
        "std_rgb": std_rgb.tolist(),
        "mean_hsv": mean_hsv,
        "std_hsv": std_hsv.tolist()
    }

def extract_texture_features(image, mask):
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Ensure mask dimensions match the image
    if mask.shape != gray_image.shape:
        mask = cv2.resize(mask, (gray_image.shape[1], gray_image.shape[0]))

    # Apply mask to retain 2D structure
    masked_gray = cv2.bitwise_and(gray_image, gray_image, mask=mask)

    # Local Binary Patterns (LBP)
    lbp = local_binary_pattern(masked_gray, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), density=True)

    # Haralick features (using GLCM)
    glcm = graycomatrix(masked_gray, [1], [0], 256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]

    # Entropy
    entropy = shannon_entropy(masked_gray)

    # Return as a dictionary
    return {
        "lbp_histogram": lbp_hist.tolist(),
        "contrast": contrast,
        "energy": energy,
        "homogeneity": homogeneity,
        "correlation": correlation,
        "entropy": entropy
    }

def extract_features(image, mask):
    # Extract features
    color_features = extract_color_features(image, mask)
    texture_features = extract_texture_features(image, mask)

    # Combine features
    features = {**color_features, **texture_features}
    return features

def extract_features_with_labels(image_path, mask_path, num_images):
    water_features = []
    non_water_features = []

    for i in range(1, num_images + 1):
        # Load image and mask
        image_file = os.path.join(image_path, f"{i}.png")
        mask_file = os.path.join(mask_path, f"{i}.png")
        image = cv2.imread(image_file)
        mask = cv2.imread(mask_file, 0)  # Load mask as grayscale (binary)

        # Create complement of the mask
        non_water_mask = cv2.bitwise_not(mask)

        # Extract water features using the original mask
        water_feature = extract_features(image, mask)
        water_features.append({"label": "water", "features": water_feature})

        # Extract non-water features using the complement mask
        non_water_feature = extract_features(image, non_water_mask)
        non_water_features.append({"label": "non-water", "features": non_water_feature})

    return water_features, non_water_features

# %% [markdown]
# MAP model definition

# %%
def train_map_model(features):
    # Separate features by class
    water_features = [f["features"] for f in features if f["label"] == "water"]
    non_water_features = [f["features"] for f in features if f["label"] == "non-water"]

    # Estimate prior probabilities
    num_water = len(water_features)
    num_non_water = len(non_water_features)
    total = num_water + num_non_water
    priors = {
        "water": num_water / total,
        "non-water": num_non_water / total
    }

    # Estimate likelihoods (assuming Gaussian distribution for simplicity)
    likelihoods = {}
    for class_name, class_features in [("water", water_features), ("non-water", non_water_features)]:
        likelihoods[class_name] = {}
        for feature_name in class_features[0].keys():
            # Collect all values for this feature across the class
            feature_values = [f[feature_name] for f in class_features if isinstance(f[feature_name], (int, float))]
            if feature_values:
                # Fit a Gaussian distribution
                mean = np.mean(feature_values)
                std = np.std(feature_values)
                likelihoods[class_name][feature_name] = (mean, std)

    return priors, likelihoods

# Example: Predicting with a MAP model
def predict_map(features, priors, likelihoods):
    predictions = []
    for f in features:
        posteriors = {}
        for class_name in priors.keys():
            # Start with the prior
            posterior = np.log(priors[class_name])
            for feature_name, value in f["features"].items():
                if feature_name in likelihoods[class_name]:
                    mean, std = likelihoods[class_name][feature_name]
                    # Compute the likelihood (log probability for numerical stability)
                    likelihood = norm.logpdf(value, mean, std)
                    posterior += likelihood
            posteriors[class_name] = posterior
        # Choose the class with the highest posterior
        predicted_class = max(posteriors, key=posteriors.get)
        predictions.append(predicted_class)
    return predictions

# %% [markdown]
# Training MAP model

# %%
# Extract and combine feature 
train_water_features, train_non_water_features = extract_features_with_labels(train_image_path, train_mask_path, num_train_image)
train_features = train_water_features + train_non_water_features

priors, likelihoods = train_map_model(train_features)

# Predict on the training set
train_predictions = predict_map(train_features, priors, likelihoods)

# Evaluate training accuracy
train_labels = [f["label"] for f in train_features]
train_accuracy = sum([1 for p, l in zip(train_predictions, train_labels) if p == l]) / len(train_labels)
print(f"Training Accuracy: {train_accuracy:.2f}")

# %% [markdown]
# Testing MAP model using testing data

# %%
# Extract and combine feature 
test_water_features, test_non_water_features = extract_features_with_labels(test_image_path, test_mask_path, num_test_image)
test_features = test_water_features + test_non_water_features

# Predict on the training set
test_predictions = predict_map(test_features, priors, likelihoods)

# Evaluate training accuracy
test_labels = [f["label"] for f in test_features]
test_accuracy = sum([1 for p, l in zip(test_predictions, test_labels) if p == l]) / len(test_labels)
print(f"Testing Accuracy: {test_accuracy:.2f}")

# %% [markdown]
# Object segmentation using morphological method

# %%
def complement(img):
    complement = np.full_like(img, 0)
    for i in range(0, len(img)):
        for j in range(1, len(img[0])):
            if img[i][j] == 0:
                complement[i][j] = 1
    
    return complement

def display_GrayScale (image_2d_array, title):
    plt.imshow(image_2d_array, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

def display_Binary (image_2d_array, title):
    plt.imshow(image_2d_array, cmap='Greys')
    plt.title(title)
    plt.axis('off')
    plt.show()

def display_morphological_operation (images, title):
    plt.figure(figsize=(5 * len(images), 5)) 
    for i, img in enumerate(images):
        plt.subplot(1, len(images), i+1)
        plt.imshow(img, cmap='gray')  # Display as grayscale
        plt.axis('off')  # Turn off axes
        plt.title(title[i])  # Optional: Add titles
    plt.tight_layout()
    plt.show()

def grayscale_to_binary (image, std_scale):
    row_mean = np.mean(image, axis = 1)
    image_mean = np.mean(row_mean)
    std = np.std(image)

    binary_image = np.full_like(image, 0)
    for i in range(0, len(image)):
        for j in range(1, len(image[0])):
            if image[i][j] > image_mean + std_scale* std:
                binary_image[i][j] = 1
    return  binary_image

#3*3 Gassian filter
def gaussian_kernel (kernel_size, variance):
    if kernel_size % 2 == 1:
        offset = (kernel_size - 1) / 2
    else:
        offset = kernel_size / 2
    x, y = np.mgrid[-offset:offset, -offset:offset]
    kernel = np.exp(-(x**2+y**2) / (2 * variance ** 2))
    kernel = kernel / kernel.sum()
    return kernel

cir2 = np.array([
        [1, 1],
        [1, 1]
], dtype=np.uint8)

cir3 = np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
], dtype=np.uint8)

cir7 = np.array([
        [0, 0, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 0, 0]
], dtype=np.uint8)

cir9 = np.array([
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
], dtype=np.uint8)

def auto_threshold (image, s1, s2):
    row_mean = np.mean(image, axis = 1)
    image_mean = np.mean(row_mean)
    std = np.std(image)
    thresholds = [image_mean - s1 * std, image_mean - s2 * std, image_mean, image_mean + s2 * std + image_mean + s1 * std]
    return  thresholds

def connect_edgels_with_thresholding(gradient_image, thresholds):
    """
    Connect edgels in a morphological gradient image using multiple thresholding.

    Parameters:
        gradient_image (np.ndarray): 2D grayscale morphological gradient image.
        thresholds (list of int): List of threshold values for segmentation.

    Returns:
        np.ndarray: Binary image with connected edgels.
    """
    # Initialize an empty binary image for the final result
    connected_edges = np.zeros_like(gradient_image, dtype=np.uint8)

    # Iterate through threshold levels
    for i, thresh in enumerate(thresholds):
        # Threshold the gradient image
        _, binary = cv2.threshold(gradient_image, thresh, 255, cv2.THRESH_BINARY)
        
        binary = binary.astype(np.uint8)
        # Find connected components
        num_labels, labels = cv2.connectedComponents(binary)
        
        # Add connected components to the final result
        connected_edges = cv2.bitwise_or(connected_edges, binary)

    return connected_edges

def morphological_segmentation_v1(image, gaussian_kernel_size, gaussian_var, grad_stc_elem, std_scale, clo_stc_elem0, clo_stc_elem1, clo_stc_elem2, thres1, thres2):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #Gaussian smoothing
    gf1 = gaussian_kernel (gaussian_kernel_size, gaussian_var)
    smooth_img = signal.convolve2d(gray_image, gf1, boundary='symm', mode='same')

    #Morphological gradient
    eroded_image = grey_erosion(smooth_img, footprint=grad_stc_elem)
    dialated_image = grey_dilation(smooth_img, footprint=grad_stc_elem)
    morphological_gradient = dialated_image - eroded_image

    #Transform gradient into binary
    thresholds = auto_threshold(morphological_gradient, 0.15, 0.1)
    binary_img = connect_edgels_with_thresholding(morphological_gradient, thresholds)

    #Expand the edge using dilation
    closed_binary = binary_closing(binary_img, clo_stc_elem0)
    closed_binary = binary_closing(closed_binary, clo_stc_elem1)
    closed_binary = binary_closing(closed_binary, clo_stc_elem2)

    #Print result 
    #imgs = [smooth_img, morphological_gradient, binary_img, dilated_binary]
    #titles =["Smoothed", "Morphological Gradient", "Binary edge", "Closing edge"]
    #display_morphological_operation(imgs, titles)


    #Fill the boundries by
    # 1. Take complement and Connecting component (transformed back to gray scale)
    for i in range(0, len(closed_binary)):
        for j in range(1, len(closed_binary[0])):
            if closed_binary[i][j] == 0:
                closed_binary[i][j] = 255
            else:
                closed_binary[i][j] = 0

    # 2. Find connected component 
    closing_uint8 = (closed_binary * 255).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closing_uint8 , connectivity=8)

    # 3. Keeping the components that is larger than the threshold
    filled_img = np.zeros_like(closing_uint8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= thres1:
            filled_img[labels == i] = 255
    #display_GrayScale(filled_img, "Fill white component")

    #Fill small black component
    # 1.Take complement
    for i in range(0, len(filled_img)):
        for j in range(1, len(filled_img[0])):
            if filled_img[i][j] == 0:
                filled_img[i][j] = 255
            else:
                filled_img[i][j] = 0
    
    # 2. Keeping the components that is larger than the threshold
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(filled_img , connectivity=8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] <= thres2:
            filled_img[labels == i] = 0

    # 3.Take complement again to highlight water
    for i in range(0, len(filled_img)):
        for j in range(1, len(filled_img[0])):
            if filled_img[i][j] == 0:
                filled_img[i][j] = 255
            else:
                filled_img[i][j] = 0
    
    #display_GrayScale(filled_img, "Fill black component")

    return filled_img

def compute_IoU(img, golden):
    numerator = 0
    denominator = 0
    for i in range(0, len(img)):
        for j in range(1, len(img[0])):
            if img[i][j] != 0 and golden[i][j] != 0:
                numerator += 1
                denominator += 1
            elif img[i][j] != 0 or golden[i][j] != 0:
                denominator += 1
    return numerator / denominator

# %% [markdown]
# Function to classify segments

# %%
def classify_segments(mask, original_image, priors, likelihoods):
    """
    Classify connected segments in a binary image using a MAP model.

    Parameters:
        binary_image (np.ndarray): 2D binary array with connected segments.
        original_image (np.ndarray): Original image for feature extraction.
        map_model (tuple): Trained MAP model (priors, likelihoods).

    Returns:
        np.ndarray: Image with classified segments.
        dict: Dictionary of segment labels and their classifications.
    """
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask , connectivity=8)
    
    # Initialize results
    classified_segments = np.zeros_like(labels)
    segment_classifications = {}
    
    # Loop through each segment
    for segment_id in range(1, num_labels):  # Skip the background (label 0)
        # Create a mask for the current segment
        segment_mask = (labels == segment_id).astype(np.uint8)
        
        features = []
        # Extract features for the segment
        feature = extract_features(original_image, segment_mask)
        features.append({"features": feature})
        #print("Extracted Features:")

        # Classify the segment using the MAP model
        label = predict_map(features, priors, likelihoods)
        
        # Store the classification result
        segment_classifications[segment_id] = label

        # Assign a unique value to the classified segments (e.g., 1 for water, 2 for non-water)
        if label[0] == 'water':
            classified_segments[labels == segment_id] = 255
        else:
            classified_segments[labels == segment_id] = 0

    return classified_segments, segment_classifications


# %% [markdown]
# Predict train data

# %%
train_IoU = 0

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

for k in range(1, num_train_image + 1):
    image_file = os.path.join(train_image_path, f"{k}.png")
    image = cv2.imread(image_file)
    org_mask= morphological_segmentation_v1(
        image = image, 
        gaussian_kernel_size = 7, 
        gaussian_var = 2, 
        grad_stc_elem = cir7, 
        std_scale = 0.1,
        clo_stc_elem0 = cir3, 
        clo_stc_elem1 = cir3,
        clo_stc_elem2= cir3,
        thres1= 1000,
        thres2 = 1000
    )

    if org_mask.dtype != np.uint8:
        mask = (org_mask * 255).astype(np.uint8)
    else:
        mask = org_mask.copy() 
    
    classified_segments1, segment_classifications1 = classify_segments(mask, image, priors, likelihoods)
    #display_GrayScale(classified_segments1, "Updated Mask1")
    #print("Segment classifications1:", segment_classifications1)

    for i in range(0, len(mask)):
        for j in range(1, len(mask[0])):
            if mask[i][j] == 0:
                mask[i][j] = 255
            else:
                mask[i][j] = 0
    classified_segments2, segment_classifications2 = classify_segments(mask, image, priors, likelihoods)
    #display_GrayScale(classified_segments2, "Updated Mask2")
    #print("Segment classifications1:", segment_classifications1)

    for i in range(0, len(mask)):
        for j in range(1, len(mask[0])):
            if classified_segments1[i][j] == 255 or classified_segments1[i][j] == 255:
                mask[i][j] = 255
            else:
                mask[i][j] = 0


    if np.array_equal(np.unique(mask), [0]):
        mask = binary_closing(org_mask, cir9)
        mask = binary_closing(org_mask, cir9)


    image_out = os.path.join(train_output_path, f"{k}.png")
    out_mask = Image.fromarray(mask)
    out_mask.save(image_out)

    #Calculate final IoU
    golden_file = os.path.join(train_mask_path, f"{k}.png")
    golden = cv2.imread(golden_file)
    golden = cv2.cvtColor(golden, cv2.COLOR_BGR2GRAY)
    IoU = compute_IoU (mask, golden)
    print("Image %i IoU: %f" % (k, IoU))
    train_IoU += IoU

print("Final average IoU: %f" % (train_IoU / num_train_image))

# %% [markdown]
# Predict test data

# %%
test_IoU = 0
for k in range(1, num_test_image + 1):
    image_file = os.path.join(test_image_path, f"{k}.png")
    image = cv2.imread(image_file)
    org_mask= morphological_segmentation_v1(
        image = image, 
        gaussian_kernel_size = 7, 
        gaussian_var = 2, 
        grad_stc_elem = cir7, 
        std_scale = 0.1,
        clo_stc_elem0 = cir3, 
        clo_stc_elem1 = cir3,
        clo_stc_elem2= cir3,
        thres1= 1000,
        thres2 = 1000
    )

    if org_mask.dtype != np.uint8:
        mask = (org_mask * 255).astype(np.uint8)
    else:
        mask = org_mask.copy() 
    
    classified_segments1, segment_classifications1 = classify_segments(mask, image, priors, likelihoods)
    #display_GrayScale(classified_segments1, "Updated Mask1")
    #print("Segment classifications1:", segment_classifications1)

    for i in range(0, len(mask)):
        for j in range(1, len(mask[0])):
            if mask[i][j] == 0:
                mask[i][j] = 255
            else:
                mask[i][j] = 0
    classified_segments2, segment_classifications2 = classify_segments(mask, image, priors, likelihoods)
    #display_GrayScale(classified_segments2, "Updated Mask2")
    #print("Segment classifications1:", segment_classifications1)

    for i in range(0, len(mask)):
        for j in range(1, len(mask[0])):
            if classified_segments1[i][j] == 255 or classified_segments1[i][j] == 255:
                mask[i][j] = 255
            else:
                mask[i][j] = 0


    if np.array_equal(np.unique(mask), [0]):
        mask = binary_closing(org_mask, cir9)
        mask = binary_closing(org_mask, cir9)
        
    image_out = os.path.join(test_output_path, f"{k}.png")
    out_mask = Image.fromarray(mask)
    out_mask.save(image_out)
    
    #Calculate final IoU
    golden_file = os.path.join(test_mask_path, f"{k}.png")
    golden = cv2.imread(golden_file)
    golden = cv2.cvtColor(golden, cv2.COLOR_BGR2GRAY)
    IoU = compute_IoU (mask, golden)
    print("Image %i IoU: %f" % (k, IoU))
    test_IoU += IoU

print("Final average IoU: %f" % (test_IoU / num_test_image))

# %%



