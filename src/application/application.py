import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import torch.nn.functional as F # activation functions
import cv2
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
import scipy.fftpack
from scipy.ndimage import gaussian_filter
import clip
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk
import customtkinter as ctk
#from transformers import pipeline


# grad_CAM implmentation
# Grad-CAM requires access to the activations of a convolutional layer and the gradients of the target class to those activations
# the last convolutional layer is used as it captures the most detailed spatial information

# for the multiclass case of labelling

multi_class_labelling=  {
    'denoising-diffusion-gan': 0, 
    'diffusion_GAN': 1, 
    'gansformer': 2, 
    'proGAN': 3, 
    'projectedGAN': 4, 
    'real': 5, 
    'stable_diffusion': 6, 
    'style_gan_1': 7, 
    'style_gan_2': 8, 
    'style_gan_3': 9, 
    'taming_transformer': 10
}

fake_class_labelling = {
    'diffusion_GAN': 0,
    'denoising-diffusion-gan': 1,
    'stable_diffusion': 2,
    'style_gan_1': 3,
    'style_gan_2' : 4,
    'style_gan_3': 5,
    'gansformer' : 6,
    'proGAN' : 7,
    'projectedGAN' : 8,
    'taming_transformer' : 9,
}

# dict storing probability of each loaded architectures
probability_fake_class = {
    'diffusion_GAN': 0,
    'denoising-diffusion-gan': 0,
    'stable_diffusion': 0,
    'style_gan_1': 0,
    'style_gan_2' : 0,
    'style_gan_3': 0,
    'gansformer' : 0,
    'proGAN' : 0,
    'projectedGAN' : 0,
    'taming_transformer' : 0,
}

probability_fake_class_multi = {
    'denoising-diffusion-gan': 0, 
    'diffusion_GAN': 0, 
    'gansformer': 0, 
    'proGAN': 0, 
    'projectedGAN': 0, 
    'real': 0, 
    'stable_diffusion': 0, 
    'style_gan_1': 0, 
    'style_gan_2': 0, 
    'style_gan_3': 0, 
    'taming_transformer': 0
}

class HierarchicalClassifier(nn.Module):
    def __init__(self, num_fake_classes,mode='hierarchical', backbone='resnet50', pretrained=True, dropout_rate=0.3):
        super(HierarchicalClassifier, self).__init__()

        # Load ResNet50 as the backbone
        self.backbone = models.resnet50(pretrained=pretrained)
        self.backbone.fc = nn.Identity()  # Remove the final fully connected layer
        self.mode = mode
        self.dropout = nn.Dropout(p=dropout_rate)
        #print("mode: ",self.mode)
        # Binary Classifier: Real vs. Fake
        self.binary_classifier = nn.Linear(2048, 1)  # Single output node for binary classification

        # Multiclass Classifier: Which Fake Model
        if self.mode == 'hierarchical':
          self.fake_classifier = nn.Linear(2048, num_fake_classes)  # Softmax output for fake class

    def forward(self, x):
        features = self.backbone(x)
        features = self.dropout(features)
        real_fake_logits = self.binary_classifier(features)
        real_fake_pred = torch.sigmoid(real_fake_logits)  # Binary classification
        
        if self.mode == 'hierarchical':
          fake_class_logits = self.fake_classifier(features)
          fake_class_pred = torch.softmax(self.fake_classifier(features), dim=1)  # Multiclass classification
          return real_fake_pred, fake_class_pred, real_fake_logits, fake_class_logits
        else:
          return real_fake_pred, real_fake_logits  # Only return binary classification in binary mode

# restNet50 with dropOut
class ResNet50Customed(nn.Module):
    def __init__(self, num_classes=11, dropout_rate=0.5, pretrained=True):
        super(ResNet50Customed, self).__init__()
        self.base_model = models.resnet50(pretrained=pretrained)
        num_features = self.base_model.fc.in_features
        
        # Replace the FC layer with dropout + new FC
        self.base_model.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(num_features, num_classes)
        )  

    def forward(self, x):
        return self.base_model(x)

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = []
        self.activations = []

        # Hook for gradients
        def backward_hook(module, grad_input, grad_output):
            self.gradients.append(grad_output[0])

        # Hook for activations
        def forward_hook(module, input, output):
            self.activations.append(output)

        # Register hooks
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)
    def generate_cam(self, input_tensor, class_idx=None):
            """
            Generate Grad-CAM heatmap for the given input image.
            """
            self.model.eval()
            output = self.model(input_tensor)  # Forward pass

            if class_idx is None:
                class_idx = output.argmax(dim=1).item()  # Get predicted class index

            # Compute gradients for the target class    
            self.model.zero_grad()
            output[:, class_idx].backward(retain_graph=True)

            # Get the gradients and activations
            gradients = self.gradients
            activations = self.activations
            #print("gradients dimension: ", gradients[0].size())
            #print("activations dimension: ", activations.shape)

            # Global average pooling of gradients
            #weights = gradients.mean(dim=[2, 3], keepdim=True)
            weights = torch.mean(gradients[0], dim=[2, 3])
            #print("weight: ",weights.size())
            #print("activation: ",activations[0].size())

            # Compute Grad-CAM
            #weights = weights.view(1, 512, 1, 1)
            weights = weights.view(1, 2048, 1, 1)
            cam = torch.sum(weights * activations[0], dim=1).squeeze()
            cam = np.maximum(cam.cpu().detach().numpy(), 0)
            cam /= np.max(cam)


            #cam = (weights * activations).sum(dim=1, keepdim=True)
            #cam = F.relu(cam)  # Apply ReLU to keep only positive influences
            #cam = cam.squeeze().cpu().numpy()

            # Normalize CAM
            cam = (cam - cam.min()) / (cam.max() - cam.min())

            return cam
    def generate_cam_flexible(self, input_tensor, mode='binary', class_idx=None):
        """
        mode = 'binary' or 'multiclass'
        class_idx: required if mode == 'multiclass'
        """
        self.model.eval()
        self.gradients.clear()
        self.activations.clear()

        
        # Forward pass
        if mode == 'binary':
            #real_fake_pred.backward(retain_graph=True)
            real_fake_pred, _, real_fake_logits, _ = self.model(input_tensor) # real_fake_logits: Raw output (before sigmoid)
            target = real_fake_logits  # raw logits for real/fake
        elif mode == 'multiclass':
            _, fake_class_pred, _, fake_class_logits = self.model(input_tensor) # adding fake class logic
            if class_idx is None:
                class_idx = fake_class_pred.argmax(dim=1).item()
            #fake_class_pred[:, class_idx].backward(retain_graph=True)
            target = fake_class_logits[:, class_idx]  # raw logit for target fake class

        # Zero gradients
        self.model.zero_grad()

        # Backward pass
        target.backward(retain_graph=True)

        # Get gradients and activations
        gradients = self.gradients
        activations = self.activations

        # Global average pooling
        weights = torch.mean(gradients[0], dim=[2, 3]).view(1, -1, 1, 1)

        # Weighted sum of activations
        cam = torch.sum(weights * activations[0], dim=1).squeeze()
        cam = np.maximum(cam.cpu().detach().numpy(), 0)
        cam = (cam - cam.min()) / (cam.max() + 1e-5)

        # Safe normalization
        cam = cam - cam.min()
        if cam.max() != 0:
            cam = cam / cam.max()
        else:
            cam = np.zeros_like(cam)

        return cam

    def overlay_heatmap(self, cam, image, alpha=0.5):
        h, w = image.shape[:2]
        cam = cv2.resize(cam, (w, h))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # NEW: Convert the image to uint8 if it's not already
        if image.dtype != np.uint8:
            image = np.clip(image * 255, 0, 255).astype(np.uint8)
        overlayed_img = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)
        return overlayed_img

def fourier_transform_shift_fft(image, log_scale=True, filter=False):
    # High-pass Laplacian kernel
    kernel = np.array([[0, -1, 0],
                       [-1, 4, -1],
                       [0, -1, 0]])

    # Convert to numpy (C, H, W) → grayscale (H, W)
    image_np = image.cpu().numpy()
    gray = 0.2989 * image_np[0] + 0.5870 * image_np[1] + 0.1140 * image_np[2]

    # Apply high-pass filter if requested
    if filter:
        gray = cv2.filter2D(gray, -1, kernel)

    # Compute 2D FFT and shift the zero-frequency component to center
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)

    # Get magnitude
    magnitude = np.abs(fft_shift)

    # Log scale for better visibility
    mag_log = np.log1p(magnitude) if log_scale else magnitude

    # Normalize (visualssation)
    norm_magnitude = (mag_log - mag_log.min()) / (mag_log.max() - mag_log.min())

    return norm_magnitude  # shape: (H, W)

def fourier_transform_shift_dct(image, log_scale=True, filter = False):

    """
    Computes 2D DCT of a 3-channel image and returns the log-magnitude spectrum.
    Assumes input is a PyTorch tensor of shape (3, H, W) with values in [0,1] or [0,255].
    """
    kernel = np.array([[0, -1, 0],
                    [-1, 4, -1],
                    [0, -1, 0]])

    # Convert to numpy (C, H, W) → (H, W) grayscale
    image_np = image.cpu().numpy()
    gray = 0.2989 * image_np[0] + 0.5870 * image_np[1] + 0.1140 * image_np[2]
    if filter == True:
        gray = cv2.filter2D(gray, -1, kernel)

    # Compute 2D DCT
    dct = scipy.fftpack.dct(scipy.fftpack.dct(gray, axis=0, norm='ortho'), axis=1, norm='ortho')

    # Apply log scaling for visualization
    dct_log = np.log1p(np.abs(dct)) if log_scale else np.abs(dct)
    #dct_log_clipped = np.clip(dct_log, a_min=1, a_max=8)
    dct_min, dct_max = dct_log.min(), dct_log.max()
    normalized = (dct_log - dct_min) / (dct_max - dct_min)
    #normalized = (dct_log_clipped - dct_log_clipped.min()) / (dct_log_clipped.max() - dct_log_clipped.min())
    return normalized  # shape: (H, W)

# function to perform thresholding of heatmap and draw boxes around heated area
def analyse_gradcam(heatmap, image, threshold=0.7, draw_box=True, min_area = 100, alpha = 0.5):
    # function to crop the boxes
    def crop_boxes(regions_collection):
        region_crops = []
        #print("image: ", image.shape)
        for region_entity in regions_collection:
            min_row, min_col, max_row, max_col = region_entity['bbox']
            crop = image[min_row:max_row, min_col:max_col]
            #print("croped shape: ",crop.shape)
            region_crops.append(crop)
        return region_crops
    
    # Resize heatmap to image size
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    # Threshold: keep only high-activation regions
    high_act_mask = heatmap_resized > threshold
    labeled_area = label(high_act_mask) # label regions, extracts key information about each blob
    # measures properties for each labeled region in label image L
    props = regionprops(labeled_area, intensity_image=heatmap_resized)

    regions_collection = []

    for i,region in enumerate(props):
        if region.area >= min_area:  # filter small noise  
            regions_collection.append({ # append region that pass the mask
                "bbox": region.bbox,  # (min_row, min_col, max_row, max_col)
                "area": region.area,
                "mean_activation": region.mean_intensity,
                "id" : i+1
            })

    # Overlay heatmap
    heatmap_rgb = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_rgb, cv2.COLOR_BGR2RGB) # covert from BGR to RGB
    overlay = cv2.addWeighted(image, alpha, heatmap_rgb, 1 - alpha, 0)

    if draw_box:
        for region_entity in regions_collection:
            min_row, min_col, max_row, max_col = region_entity['bbox']
            region_id = region_entity["id"]
            cv2.rectangle(overlay, (min_col, min_row), (max_col, max_row), (0, 255, 0), 2)
            # add id to the box (region)
            cv2.putText(overlay,
                f"ID {region_id}", (min_col, min_row - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1
            )
    region_crops = crop_boxes(regions_collection)
    frequency_img_DCT = []
    frequency_img_FFT = []
    DCT_no_log = []
    FFT_no_log = []
    for region in region_crops:
        region_resized = cv2.resize(region, (128, 128), interpolation=cv2.INTER_CUBIC)
        toTensor = torch.from_numpy(region_resized)
        toTensor = toTensor.permute(2,0,1)
        frequency_img_DCT.append(fourier_transform_shift_dct(toTensor, True, False))
        frequency_img_FFT.append(fourier_transform_shift_fft(toTensor, True, True))
        DCT_no_log.append(fourier_transform_shift_dct(toTensor, False, False))
        FFT_no_log.append(fourier_transform_shift_fft(toTensor, False, True))
    return regions_collection, overlay, frequency_img_DCT, frequency_img_FFT, DCT_no_log, FFT_no_log

def find_className(pred=None, label=None, labelling_dict=None):
  pred_class = None
  label_class = None
  # find key by label in the customised dict
  for key,value in labelling_dict.items():
      if value == pred:
        pred_class = key
      if value == label:
        label_class = key
  return pred_class, label_class


# Load CLIP model (at the top of your script, after imports)
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

# Define CLIP concept matcher
def clip_concept_match(image_crop, concepts):
    image_pil = Image.fromarray(image_crop)
    image_input = clip_preprocess(image_pil).unsqueeze(0).to(device)
    text_tokens = clip.tokenize(concepts).to(device)

    with torch.no_grad():
        image_features = clip_model.encode_image(image_input)
        text_features = clip_model.encode_text(text_tokens)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarities = (image_features @ text_features.T).squeeze(0).cpu().numpy()

    results = list(zip(concepts, similarities))
    return sorted(results, key=lambda x: x[1], reverse=True)

# CLIP concepts definition
clip_concepts = [
    # GAN artifacts
    "smooth skin", "asymmetrical eyes", "blurry teeth", "weird mouth", "broken eyeglasses",
    "strange background blur", "overly symmetrical face", "repetitive hair texture",
    "weird object shape", "blurred object edges", "melting object",

    # Diffusion Model artifacts
    "painted look", "dream-like background", "hallucinated object",
    "incoherent text", "weird hands", "wrong number of fingers",
    "warped reflection", "inconsistent shadows",

    # Fusion Model artifacts (e.g., Taming Transformer)
    "inpainting artifact", "semantic mismatch", "confused object identity",
    "blending issue", "different styles in one image", "poorly blended region",

    # Real image characteristics
    "natural skin texture", "realistic lighting", "subtle shadows", "detailed pores",
    "fine wrinkles", "natural hair strands", "correct number of fingers",
    "aligned facial features", "realistic reflections", "natural eye reflections",
    "natural background lighting", "authentic fabric texture", "readable text",
    "correct object shape", "clear signboards", "objects in natural positions",
    "consistent object size", "natural positioning", "non-repeating background",
    "real photo lighting", "camera blur", "realistic photo layout",
    "lens flare", "correct lighting direction", "visible blemishes",
    "normal imperfections"
]

def unnormalize(img, mean, std):
   # Ensure the mean and std are broadcastable with the image
   # std[:, None, None] reshape it from shape 3, to 3,1,1 for per-channel unnormalisation
    mean = torch.tensor(mean).to(img.device)  # Ensure mean is a tensor on the same device as img
    std = torch.tensor(std).to(img.device)    # Ensure std is a tensor on the same device as img

    # Reshape mean and std to [C, 1, 1] for broadcasting
    std_1 = std.view(3, 1, 1)  # Reshape std to [3, 1, 1]
    mean_1 = mean.view(3, 1, 1)  # Reshape mean to [3, 1, 1]

    # Unnormalize the image
    img = img * std_1 + mean_1
    return img

def patch_gradcam_regions(image_tensor, gradcam_mask, replacement_tensor, threshold=0.7):
    """
    Patch **all** high-activation Grad-CAM regions with a resized replacement image.
    """
    _, H, W = image_tensor.shape  # in the format of (C, H, W)

    # Step 1: Resize Grad-CAM to match input size
    heatmap_resized = cv2.resize(gradcam_mask, (W, H))

    # Step 2: Threshold
    binary_mask = (heatmap_resized >= threshold).astype(np.uint8)

    # Step 3: Find connected components
    labeled = label(binary_mask)
    props = regionprops(labeled)

    if not props:
        print("⚠️ No significant regions found. Consider lowering threshold.")
        return image_tensor

    # Step 4: Copy the original image
    counterfactual = image_tensor.clone()

    # Step 5: For each region, patch it
    for region in props:
        minr, minc, maxr, maxc = region.bbox

        region_height = maxr - minr
        region_width = maxc - minc

        # Resize replacement image to match region size
        replacement_resized = F.interpolate(
            replacement_tensor.unsqueeze(0),
            size=(region_height, region_width),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)  # (3, region_height, region_width)

        # Replace region
        counterfactual[:, minr:maxr, minc:maxc] = replacement_resized

    # Optional: Unnormalize for visualization if needed
    unnormalized_img = unnormalize(counterfactual, mean, std)
    unnormalized_img = unnormalized_img.permute(1, 2, 0)

    return counterfactual

def plotting(image, predicted_class_name,
             frequency_domain,frequency_domain_fft,overlay_multi,frequency_img_heated_DCT_multi, 
             frequency_img_heated_FFT_multi, overlay_bin, frequency_img_heated_DCT_bin, 
             frequency_img_heated_FFT_bin,overlayed_cf_bin, overlayed_cf_multi, cf_className):
    # Display the image with the predicted class name
    plt.figure(figsize=(30, 20))
    plt.subplot(4, 5, 1)
    image = np.array(image)
    plt.imshow(image)
    plt.axis('off')
    plt.text(10, 10, f'Predicted: {predicted_class_name}', fontsize=12, color='white', backgroundcolor='red')

    plt.subplot(4, 5, 2)
    vmin, vmax = np.percentile(frequency_domain, [1, 99])
    plt.imshow((frequency_domain),  cmap='plasma',vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title("Whole image DCT")
    plt.axis("off")

    plt.subplot(4, 5, 3)
    vmin, vmax = np.percentile(frequency_domain_fft, [1, 99])
    plt.imshow((frequency_domain_fft),  cmap='magma',vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title("Whole image FFT")
    plt.axis("off")

    if overlay_multi is not None:
        plt.subplot(4, 5, 6)
        plt.imshow(overlay_multi)
        plt.axis("off")
        plt.title("Grad-CAM multiclass")
        temp1 = len(frequency_img_heated_DCT_multi)
        temp2 = len(frequency_img_heated_FFT_multi)
        temp3 = 7+temp1+temp2

        for i,img  in enumerate(frequency_img_heated_DCT_multi[:2]):
            plt.subplot(4, 5, 7+i)
            vmin, vmax = np.percentile(img, [1, 99])
            plt.imshow((img), cmap='magma', vmin=vmin, vmax=vmax)
            plt.colorbar()
            plt.axis("off")
            plt.title(f"(Multiclass) DCT box {i+1}")

        for i,img  in enumerate(frequency_img_heated_FFT_multi[:2]):
            plt.subplot(4, 5, 7+temp1+i)
            vmin, vmax = np.percentile(img, [1, 99])
            plt.imshow((img), cmap='magma', vmin=vmin, vmax=vmax)
            plt.colorbar()
            plt.axis("off")
            plt.title(f"(Multiclass) FFT box {i+1}")
    if overlay_bin is not None:
        plt.subplot(4, 5, 11)
        plt.imshow(overlay_bin)
        plt.axis("off")
        plt.title("Grad-CAM binary")

        for i,img  in enumerate(frequency_img_heated_DCT_bin[:2]):
            plt.subplot(4, 5, 12+i)
            vmin, vmax = np.percentile(img, [1, 99])
            plt.imshow((img), cmap='magma', vmin=vmin, vmax=vmax)
            plt.colorbar()
            plt.axis("off")
            plt.title(f"(Binary) DCT box {i+1}")

        for i,img  in enumerate(frequency_img_heated_FFT_bin[:2]):
            plt.subplot(4, 5, 12+len(frequency_img_heated_DCT_bin)+i)
            vmin, vmax = np.percentile(img, [1, 99])
            plt.imshow((img), cmap='magma', vmin=vmin, vmax=vmax)
            plt.colorbar()
            plt.axis("off")
            plt.title(f"(Binary) FFT box {i+1}")
    if overlayed_cf_bin is not None:
        plt.subplot(4,5,16)
        plt.imshow(overlayed_cf_bin)
        plt.axis("off")
        plt.title("Grad-CAM counterfactual binary")
        plt.text(10, 10, f'Predicted: {cf_className}', fontsize=12, color='white', backgroundcolor='red')
    if overlayed_cf_multi is not None:
        plt.subplot(4,5,17)
        plt.imshow(overlayed_cf_multi)
        plt.axis("off")
        plt.title("Grad-CAM counterfactual multiclass")
        plt.text(10, 10, f'Predicted: {cf_className}', fontsize=12, color='white', backgroundcolor='red')

    plt.savefig(F"result_example/{fileName_for_image}_result.png")
    plt.show()

def clipScore(regions = None, wholeImg = None):
    # Analyze each cropped Grad-CAM region with CLIP
    region_clip_results = []
    if wholeImg is not None:
        region_crop_resized = cv2.resize(wholeImg, (224, 224))
        clip_matches = clip_concept_match(region_crop_resized, clip_concepts)
        top_3_clip_concept = clip_matches[:3]  # Get top-1 match
        region_clip_results.append((0, top_3_clip_concept))
        insert_log("whole image Clipscore")
    else:
        for region in regions:
            minr, minc, maxr, maxc = region['bbox']
            region_crop = image_np[minr:maxr, minc:maxc]  # Crop from original image (not normalized)
            region_crop_resized = cv2.resize(region_crop, (224, 224))  # Resize for CLIP
            
            clip_matches = clip_concept_match(region_crop_resized, clip_concepts)
            top_3_clip_concept = clip_matches[:3]  # Get top-1 match
            region['clip_concept'] = top_3_clip_concept
            region_clip_results.append((region['id'], top_3_clip_concept))
        

    # Display CLIP results for each region
    insert_log("\nCLIP Semantic Concept Matches for Grad-CAM Regions:")
    for rid, concepts in region_clip_results: 
        for (concept, score) in concepts:
            insert_log(f"Region ID {rid}: {concept} ({score:.2f})")

def traces_detection(freq_image):
    # Measures energy in high-frequencies
    def high_freq_energy(magnitude):
        h, w = magnitude.shape
        center_crop = magnitude[h//4:3*h//4, w//4:3*w//4]
        border_crop = magnitude.copy()
        border_crop[h//4:3*h//4, w//4:3*w//4] = 0
        return border_crop.sum() / (center_crop.sum() + 1e-5)
    # Measures spatial repetition artifacts
    def checkerboard_score(magnitude):
        sobel_x = cv2.Sobel(magnitude, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(magnitude, cv2.CV_64F, 0, 1, ksize=3)
        return np.mean(np.abs(sobel_x)) + np.mean(np.abs(sobel_y))
    # Measures frequency irregularity
    def radial_variance(magnitude):
        h, w = magnitude.shape
        y, x = np.indices((h, w))
        r = np.sqrt((x - w//2)**2 + (y - h//2)**2).astype(int)
        radial_profile = [magnitude[r == radius].mean() for radius in np.unique(r)]
        return np.var(radial_profile)

    return {
        'high_freq_energy': F"{high_freq_energy(freq_image):.4f}",
        'checkerboard_score': F"{checkerboard_score(freq_image):.4f}",
        'radial_variance': F"{radial_variance(freq_image):.4f}"
    }

def choose_model(type):
    if type == "hierarchical":
        model_choosing = HierarchicalClassifier(num_fake_classes,mode=mode,dropout_rate=0.3 )
        tl = model_choosing.backbone.layer4[-1] # Last convolutional layer in ResNet-50
        return model_choosing, tl
    elif type == "multi":
        model_choosing = ResNet50Customed(num_classes=num_fake_classes+1, dropout_rate= 0.5)
        tl = model_choosing.base_model.layer4[-1]
        return model_choosing , tl
    else:
        ValueError("Wrong type of model chosen")

class App:
    def __init__(self, root):
        self.root = root
        self.root.geometry("1000x900")
        self.root.title("🖼️ AI Image Authenticity Analysis")

        # Title Label
        self.title_label = tk.Label(root, text="Image Authenticity Analysis", font=("Helvetica", 18, "bold"))
        self.title_label.pack(pady=10)

        # Frame for Image Display
        self.image_frame = tk.Frame(root)
        self.image_frame.pack(pady=5)

        # Image Panel
        self.panel = tk.Label(self.image_frame)
        self.panel.pack()

        # Frame for Buttons
        self.button_frame = tk.Frame(root)
        self.button_frame.pack(pady=10)

        # Select Image Button
        self.img_button = ctk.CTkButton(self.button_frame, text="📂 Select Image", font=("Helvetica", 20), command=self.select_image, corner_radius=20, 
                                        fg_color="#333333", text_color="white",width = 200, height=50)
        self.img_button.grid(row=0, column=0, padx=10)

        # Run Analysis Button
        self.analysis_button = ctk.CTkButton(self.button_frame, text="🚀 Run Analysis", font=("Helvetica", 20), command=self.run_analysis, corner_radius=20, 
                                        fg_color="#333333", text_color="white",width = 200, height=50)
        self.analysis_button.grid(row=0, column=1, padx=10)

        self.choosing_model = ctk.CTkButton(root, text="Model type: Hierarchical",font=("Helvetica", 20), command=self.toggle_model, corner_radius=20, 
                                       fg_color="#333333", text_color="white", width = 250, height=50)
        self.choosing_model.pack(pady=10)


        # Frame for Output Text
        self.output_frame = tk.Frame(root)
        self.output_frame.pack(pady=10)

        # Output Text Box (Scrollable)
        App.output_log = tk.Text(self.output_frame, height=40, width=120, font=("Courier", 10),bg="#e6e7e8", fg="black", insertbackground="black")
        App.output_log.pack(side=tk.LEFT, padx=5)

        self.scrollbar = tk.Scrollbar(self.output_frame, command=App.output_log.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        App.output_log.configure(yscrollcommand=self.scrollbar.set)

    def select_image(self):
        global image_path, image, input_tensor, input_batch, image_np, fileName_for_image

        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")]
        )
        if path:
            image_path = path
            fileName_for_image = path.split("/")[-1].split(".")[0]  # Extract file name without extension
            print(f"Selected image: {image_path}")
            print(f"fileName_for_image: {fileName_for_image}")

            image = Image.open(image_path)
            image_display = image.resize((224, 224))  # Resize for display
            img = ImageTk.PhotoImage(image_display)
            self.panel.config(image=img)
            self.panel.image = img  # Keep reference

            input_tensor = preprocess(image)
            input_batch = input_tensor.unsqueeze(0)
            image_np = np.array(image.resize((256, 224)))

    def run_analysis(self):
        self.output_log.delete('1.0', tk.END)
        if image == None:
            insert_log("Error: No image imported.")
        else:
            insert_log("Start running anaylsis...")
            full_pipeline()

    def toggle_model(self):
        global model_type, model_chosen, target_layer, gradcam

        if model_type == "hierarchical":
            model_type = "multi"
            self.choosing_model.configure(text="Model type: Multiclass")

            # Load multiclass model
            model_chosen, target_layer = choose_model(model_type)
            model_chosen.load_state_dict(torch.load('final_model/resnet50_multi_lr=5e-05_epochs=40_fake-class=10_large.pth'))
            model_chosen.eval()
            gradcam = GradCAM(model_chosen, target_layer)

            insert_log("Switched to Multiclass Model.")

        else:
            model_type = "hierarchical"
            self.choosing_model.configure(text="Model type: Hierarchical")

            # Load hierarchical model
            model_chosen, target_layer = choose_model(model_type)
            model_chosen.load_state_dict(torch.load('final_model/resnet50_hierarchical_lr=5e-05_epochs=40_fake-class=10_large.pth'))
            model_chosen.eval()
            gradcam = GradCAM(model_chosen, target_layer)

            insert_log("Switched to Hierarchical Model.")
# load model (constant)
num_fake_classes = 10
mode = 'hierarchical' 
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# model setting
model_type = "hierarchical"
counterfactual_model = True
patch = True

target_layer = None
model_chosen, target_layer = choose_model(model_type)

# Initialize Grad-CAM
# ResNet-50 has multiple layers (layer1, layer2, layer3, layer4).
# The last convolutional layer before global average pooling is layer4[-1].
gradcam = GradCAM(model_chosen, target_layer)
model_chosen.load_state_dict(torch.load('final_model/resnet50_hierarchical_lr=5e-05_epochs=40_fake-class=10_large.pth'))
model_chosen.eval()

fileName_for_image = None
image_path = None
image = None
preprocess = transforms.Compose([  # necessary preprocessing for the image
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
input_tensor = None
input_batch = None
# Convert image to numpy
image_np = None
predicted_class_name = None

def full_pipeline():
    # Perform inference
    if model_type == "hierarchical":
        counterfactual_model = True
    else:
        counterfactual_model = False
    if model_type == "hierarchical":
        with torch.no_grad():
            #real_fake_pred, fake_class_pred = model(input_batch)
            real_fake_pred, fake_class_pred, real_fake_logits, fake_class_logits = model_chosen(input_batch)
        
        # Determine if the image is real or fake
        is_fake = real_fake_pred.item() > 0.5
        cam_binary = None
        cam_multiclass = None
        overlayed_cf_bin, overlayed_cf_multi, counterfactual_class_name_patch = None, None, None
        # generate fake class classification grad cam
        if is_fake:
            predicted_class = fake_class_pred.argmax(dim=1).item()
            predicted_class_name, _ = find_className(pred=predicted_class,labelling_dict=fake_class_labelling)

            # get probability of each classes
            # class_probabilities: list of architecture with probabilities where indices map with the fake classes
            class_probabilities = fake_class_pred.squeeze().tolist()
            for idx in range(len(class_probabilities)):
                y_predicted, _ = find_className(pred=idx,labelling_dict=fake_class_labelling)
                probability_fake_class[y_predicted] = class_probabilities[idx]
            
            cam_multiclass = gradcam.generate_cam_flexible(input_batch, mode='multiclass', class_idx=predicted_class)
            cam_binary = gradcam.generate_cam_flexible(input_batch, mode='binary')
            
        else:
            cam_binary = gradcam.generate_cam_flexible(input_batch, mode='binary')
            predicted_class_name = 'real'

        # manage the image for display
        #overlayed_image = gradcam.overlay_heatmap(cam, image_np)

        # whole image fft and dct
        frequency_domain_dct = fourier_transform_shift_dct(input_tensor,True, False)
        frequency_domain_fft = fourier_transform_shift_fft(input_tensor,True, True)
        regions_multi, overlay_multi, frequency_img_heated_DCT_multi, frequency_img_heated_FFT_multi, noLogDCT_multi, noLogFFT_multi = None, None, None, None , None, None
        regions_bin, overlay_bin, frequency_img_heated_DCT_bin, frequency_img_heated_FFT_bin, noLogDCT_bin, noLogFFT_bin = None, None, None, None , None, None
        if cam_multiclass is not None:
            regions_multi, overlay_multi, frequency_img_heated_DCT_multi, frequency_img_heated_FFT_multi, noLogDCT_multi, noLogFFT_multi = analyse_gradcam(cam_multiclass, image_np, threshold=0.7)
        if cam_binary is not None:
            regions_bin, overlay_bin, frequency_img_heated_DCT_bin, frequency_img_heated_FFT_bin, noLogDCT_bin, noLogFFT_bin = analyse_gradcam(cam_binary, image_np, threshold=0.7)

        # printing results
        insert_log("\n=================================================================================")
        insert_log(f'The predicted class is: {predicted_class_name}')
        insert_log(f"Real vs fake prediction confidence: {is_fake:.4f}")
        if is_fake:
            insert_log("\nProbability of each class")
            for k,v in probability_fake_class.items():
                insert_log(F"{k} class probability: {v:.4f}")
            insert_log(f"Fake class prediction confidence: {probability_fake_class[predicted_class_name]:.4f}")
        insert_log("\n=================================================================================")
        
        insert_log("Grad CAM thresholding region overview for binary classification: ")
        # insert_log region info multi
        for i, r in enumerate(regions_bin):
            insert_log(f"Region {i+1}: BBox={r['bbox']}, Area={r['area']}, Activation={r['mean_activation']:.2f}")
        
        if regions_multi is not None:
            insert_log("Grad CAM thresholding region overview for multiclass classification: ")
            # insert_log region info multi
            for i, r in enumerate(regions_multi):
                insert_log(f"Region {i+1}: BBox={r['bbox']}, Area={r['area']}, Activation={r['mean_activation']:.2f}")
        
        insert_log("\n==================================================================================")
        
        insert_log("\nForensic scores for DCT real vs fake detection")
        for i,img in enumerate(noLogDCT_bin):
            result = traces_detection(img)
            insert_log(F"forensic scores for region {i+1}: {result}")
        
        insert_log("\nForensic scores for FFT real vs fake detection")
        for i,img in enumerate(noLogFFT_bin):
            result = traces_detection(img)
            insert_log(F"forensic scores for region {i+1}: {result}")
        
        # insert_log forensic scores multi
        if noLogDCT_multi is not None:
            insert_log("\nForensic scores for DCT fake classes detection")
            for i,img in enumerate(noLogDCT_multi):
                result = traces_detection(img)
                insert_log(F"forensic scores for region {i+1}: {result}")
        if noLogFFT_multi is not None:
            insert_log("\nForensic scores for FFT fake classes detection")
            for i,img in enumerate(noLogFFT_multi):
                result = traces_detection(img)
                insert_log(F"forensic scores for region {i+1}: {result}")
        insert_log("\n==================================================================================")
        insert_log("\n CLIP score matching session for binary classification: ")
        # CLIP score multi
        if (len(regions_bin) > 0):
            clipScore(regions=regions_bin)
        else:
            clipScore(regions=None, wholeImg=image_np)
        if regions_multi is not None:
            insert_log("\n CLIP score matching session for multiclass classification: ")
            # CLIP score multi
            if (len(regions_multi) > 0):
                clipScore(regions=regions_multi)
            else:
                clipScore(regions=None, wholeImg=image_np)
        insert_log("\n==================================================================================")

        if counterfactual_model:
            print("\nApplying Counterfactual Perturbations...")
            # Load another replacement image (for example from another fake model)
            if patch:
                print("patching..........")
                counterfactual_class_name_patch = "real"
                print("predicted_class_name: ", predicted_class_name)
                if predicted_class_name == "real":
                    replacement_path = "counterfactual_patch/fake.png"
                else:
                    replacement_path = "counterfactual_patch/real.png"  
                replacement_img = Image.open(replacement_path)
                replacement_tensor = preprocess(replacement_img)
                # analyse binary only
                patched_image_tensor = patch_gradcam_regions(input_tensor, cam_binary, replacement_tensor)
                patched_image_batch = patched_image_tensor.unsqueeze(0)

                unnormalized_img = unnormalize(patched_image_tensor, mean, std)
                unnormalized_img = unnormalized_img.permute(1, 2, 0).numpy()
                cf_is_fake = None
                with torch.no_grad():
                    real_fake_pred_cf_patch, fake_class_pred_cf_patch, _, _ = model_chosen(patched_image_batch)
                    cf_is_fake = real_fake_pred_cf_patch.item() > 0.5
                cf_prob = probability_fake_class
                if cf_is_fake:
                    counterfactual_patch_pred = fake_class_pred_cf_patch.argmax(dim=1).item()
                    counterfactual_class_name_patch, _ = find_className(pred=counterfactual_patch_pred, labelling_dict=fake_class_labelling)
                    cam_multi_cf = gradcam.generate_cam_flexible(patched_image_batch, mode='multiclass')
                    overlayed_cf_multi = gradcam.overlay_heatmap(cam_multi_cf, unnormalized_img)
                    cam_bin_cf = gradcam.generate_cam_flexible(patched_image_batch, mode='binary')
                    overlayed_cf_bin = gradcam.overlay_heatmap(cam_bin_cf, unnormalized_img)

                else:
                    cam_bin_cf = gradcam.generate_cam_flexible(patched_image_batch, mode='binary')
                    overlayed_cf_bin = gradcam.overlay_heatmap(cam_bin_cf, unnormalized_img)

                class_probabilities_cf = fake_class_pred.squeeze().tolist()
                for idx in range(len(class_probabilities_cf)):
                    y_predicted, _ = find_className(pred=idx,labelling_dict=fake_class_labelling)
                    cf_prob[y_predicted] = class_probabilities_cf[idx]

                insert_log("\n=================================================================================")
                insert_log("counterfactual analysis")
                insert_log(f'The counterfactual predicted class is: {counterfactual_class_name_patch}')
                insert_log(f"Real vs fake prediction confidence: {cf_is_fake:.4f}")
                if cf_is_fake:
                    insert_log("\nProbability of each class")
                    for k,v in cf_prob.items():
                        insert_log(F"{k} class probability: {v:.4f}")
                    insert_log(f"Fake class prediction confidence: {cf_prob[counterfactual_class_name_patch]:.4f}")
                insert_log("\n=================================================================================")

        plotting(
            image, predicted_class_name,
            frequency_domain_dct,frequency_domain_fft,
            overlay_multi,frequency_img_heated_DCT_multi,
            frequency_img_heated_FFT_multi, overlay_bin, 
            frequency_img_heated_DCT_bin, frequency_img_heated_FFT_bin, overlayed_cf_bin, overlayed_cf_multi, counterfactual_class_name_patch)
    else:
        with torch.no_grad():
            output = model_chosen(input_batch)
        # Get the predicted class
        _, predicted_class = output.max(1)
        # Get predicted class
        predicted_class = output.argmax(dim=1).item()
        predicted_class_name, _ = find_className(pred=predicted_class, labelling_dict=multi_class_labelling)

        # get probability of each classes
        # class_probabilities: list of architecture with probabilities where indices map with the fake classes
        output_probs = F.softmax(output, dim=1)
        class_probabilities = output_probs.squeeze().tolist()
        
        for idx in range(len(class_probabilities)):
            y_predicted, _ = find_className(pred=idx,labelling_dict=multi_class_labelling)
            probability_fake_class_multi[y_predicted] = class_probabilities[idx]

        # Generate Grad-CAM (for multiclass classifier)
        cam_multiclass = gradcam.generate_cam(input_batch, class_idx=predicted_class)
        overlayed_multiclass = gradcam.overlay_heatmap(cam_multiclass, image_np)

        # No need for binary CAM here (pure multiclass model)

        # Whole image FFT and DCT
        frequency_domain_dct = fourier_transform_shift_dct(input_tensor, True, False)
        frequency_domain_fft = fourier_transform_shift_fft(input_tensor, True, True)

        # Grad-CAM region analysis (multiclass)
        regions_multi, overlay_multi, frequency_img_heated_DCT_multi, frequency_img_heated_FFT_multi, noLogDCT_multi, noLogFFT_multi = analyse_gradcam(cam_multiclass, image_np, threshold=0.7)

        # Printing results
        insert_log("\n=================================================================================")
        insert_log(f'The predicted class is: {predicted_class_name}')
        insert_log("\nProbability of each class")
        for k,v in probability_fake_class_multi.items():
            insert_log(F"{k} class probability: {v:.4f}")
        insert_log(f"\nPrediction confidence: {probability_fake_class_multi[predicted_class_name]:.2f}")
        insert_log("\n=================================================================================")
        insert_log("Grad CAM thresholding region overview: ")
        # Print region info
        for i, r in enumerate(regions_multi):
            insert_log(f"Region {i+1}: BBox={r['bbox']}, Area={r['area']}, Activation={r['mean_activation']:.2f}")

        # Print forensic scores
        insert_log("\n==================================================================================")
        insert_log("\nForensic scores for DCT multiclass detection")
        for i,img in enumerate(noLogDCT_multi):
            result = traces_detection(img)
            insert_log(F"forensic scores for region {i+1}: {result}")
        if frequency_img_heated_FFT_multi is not None:
            insert_log("\nForensic scores for FFT multiclass detection")
            for i,img in enumerate(noLogFFT_multi):
                result = traces_detection(img)
                insert_log(F"forensic scores for region {i+1}: {result}")
        insert_log("\n==================================================================================")
        # CLIP semantic concept matching
        if (len(regions_multi) > 0):
            clipScore(regions=regions_multi)
        else:
            clipScore(regions=None, wholeImg=image_np)
        insert_log("\n==================================================================================")
        # Plotting (only multiclass cam, no binary cam in pure multiclass model)
        plotting(
            image, predicted_class_name,
            frequency_domain_dct, frequency_domain_fft,
            overlay_multi, frequency_img_heated_DCT_multi, frequency_img_heated_FFT_multi,
            overlay_bin=None, frequency_img_heated_DCT_bin=[], frequency_img_heated_FFT_bin=[], overlayed_cf_bin=None, overlayed_cf_multi=None, cf_className=None
        )

def insert_log(message):
    app.output_log.insert(tk.END, message + "\n")
    app.output_log.see(tk.END)  # Auto-scroll to the bottom

root = tk.Tk()
app = App(root)
root.mainloop()