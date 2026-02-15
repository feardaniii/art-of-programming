"""
================================================================================
BEYOND CNNs: The Horizon of Computer Vision
================================================================================

Course: The Art of Programming - CNN & Transfer Learning (Session 38)
Lesson: Style Transfer, GANs, Diffusion, CLIP, and Vision Transformers

PREREQUISITES:
    - Scripts 1-3: Convolution, CNN training, Transfer Learning

THE BIG PICTURE:
    In Scripts 1-3, you mastered the CNN pipeline:
    convolution -> training -> transfer learning -> deployment* (soon we'll learn more about deployment).

    But the field has not stood still. Today we explore what comes NEXT:
    - Style Transfer: making a photo look like a painting
    - GANs: generating images from noise
    - Diffusion Models: the technology behind DALL-E and Stable Diffusion
    - CLIP: understanding images AND text together
    - Vision Transformers: replacing convolution with attention

    This script is more CONCEPTUAL than the previous ones.
    The goal is to understand WHERE the field is heading,
    so you can learn these techniques when you need them.

================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ==============================================================================
# PART 1: NEURAL STYLE TRANSFER - Art Meets AI
# ==============================================================================

def explain_style_transfer():
    """
    Style Transfer: Combine the CONTENT of one image with the STYLE of another.

    How it works:
    1. Use a pre-trained VGG19 to extract features
    2. Content features (high-level layers): WHAT objects are in the image
    3. Style features (low-level layers): HOW it looks (textures, colors, patterns)
    4. Generate a new image that matches content of photo + style of painting

    The KEY insight: we optimize the IMAGE, not the NETWORK.
    The CNN is frozen. The PIXELS are the variables.
    """

    print("=" * 70)
    print("PART 1: NEURAL STYLE TRANSFER")
    print("Your Photo in Van Gogh's Style")
    print("=" * 70)
    print()

    print("THE CONCEPT:")
    print()
    print("  Content Image (your photo) + Style Image (Van Gogh painting)")
    print("                      |")
    print("             CNN Feature Extraction (VGG19)")
    print("                      |")
    print("     Content Features          Style Features")
    print("     (what is present)         (how it looks)")
    print("                      |")
    print("             Optimization Process")
    print("                      |")
    print("     Your photo painted in Van Gogh's style!")
    print()

    print("THE MATHEMATICS:")
    print()
    print("  Content Loss = || F_content - F_generated ||^2")
    print("    'The generated image should have the same objects/structure'")
    print()
    print("  Style Loss = || Gram(F_style) - Gram(F_generated) ||^2")
    print("    'The generated image should have the same textures/patterns'")
    print("    Gram matrix captures correlations between feature maps")
    print("    (texture = which features tend to appear together)")
    print()
    print("  Total Loss = alpha * Content_Loss + beta * Style_Loss")
    print("    alpha: how much to preserve content")
    print("    beta: how much to apply style")
    print()

    print("THE KEY INSIGHT:")
    print("  We are NOT training the network.")
    print("  We are optimizing the INPUT IMAGE.")
    print("  The CNN is FROZEN. The PIXELS are the variables!")
    print("  Gradient descent on pixel values, not weights.")
    print()

    print("REAL-WORLD APPLICATIONS:")
    print("  - Prisma app: real-time style transfer on mobile")
    print("  - Adobe Neural Filters: built into Photoshop")
    print("  - TikTok/Instagram: artistic video filters")
    print("  - Game development: procedural texture generation")
    print()


def style_transfer_demo():
    """
    Hands-on demo: Extract style features from a real image using VGG19.
    We show WHAT the network uses for 'content' vs 'style' representation.
    """

    print("HANDS-ON: Extracting Style Features with VGG19")
    print("-" * 50)
    print()

    # Load VGG19
    print("Loading VGG19 pre-trained on ImageNet...")
    vgg = keras.applications.VGG19(weights='imagenet', include_top=False)
    vgg.trainable = False

    # Style layers (low/mid-level: textures, patterns)
    style_layers = [
        'block1_conv1',  # Very early: basic textures
        'block2_conv1',  # Early: more complex textures
        'block3_conv1',  # Mid: style patterns
        'block4_conv1',  # Mid-high: larger style patterns
        'block5_conv1',  # High: overall style
    ]

    # Content layer (high-level: semantic content)
    content_layers = ['block5_conv2']

    print()
    print("  Style layers (capture textures and patterns):")
    for layer in style_layers:
        print(f"    - {layer}")
    print()
    print("  Content layers (capture objects and structure):")
    for layer in content_layers:
        print(f"    - {layer}")
    print()

    # Load a CIFAR-10 image and extract features
    (X_train, _), _ = keras.datasets.cifar10.load_data()
    sample_image = X_train[42]  # A recognizable image

    # Preprocess for VGG19
    image = tf.image.resize(sample_image, (224, 224))
    image = keras.applications.vgg19.preprocess_input(image[np.newaxis, ...])

    # Build feature extraction model
    all_layer_names = style_layers + content_layers
    outputs = [vgg.get_layer(name).output for name in all_layer_names]
    feature_model = keras.Model(inputs=vgg.input, outputs=outputs)
    features = feature_model.predict(image, verbose=0)

    # Visualize style features from different layers
    fig, axes = plt.subplots(2, 5, figsize=(18, 7))

    # First row: style features from each layer
    for i, (feat, name) in enumerate(zip(features[:5], style_layers)):
        # Show the first feature map
        ax = axes[0, i]
        ax.imshow(feat[0, :, :, 0], cmap='viridis')
        ax.set_title(f'{name}\n{feat.shape[1]}x{feat.shape[2]}', fontsize=9)
        ax.axis('off')

    # Second row: more feature maps from layer 3 (to show diversity)
    mid_features = features[2]  # block3_conv1
    for i in range(5):
        ax = axes[1, i]
        if i < mid_features.shape[-1]:
            ax.imshow(mid_features[0, :, :, i], cmap='viridis')
            ax.set_title(f'block3_conv1\nmap {i}', fontsize=9)
        ax.axis('off')

    plt.suptitle('Style Features Extracted by VGG19\n'
                 'Top: One feature from each style layer | Bottom: Multiple features from block3',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('style_features_demo.png', dpi=150, bbox_inches='tight')
    print("  Style features visualization saved: style_features_demo.png")
    plt.close()
    print()

    print("  OBSERVE:")
    print("    Early layers (block1): detect fine textures, small patterns")
    print("    Middle layers (block3): detect medium-scale style elements")
    print("    Late layers (block5): capture overall composition and style")
    print()
    print("  In full style transfer, we compute a GRAM MATRIX for each layer.")
    print("  The Gram matrix captures which features tend to co-occur,")
    print("  which defines the 'style' or 'texture' of the image.")
    print()


# ==============================================================================
# PART 2: GENERATIVE MODELS - Creating Images from Nothing
# ==============================================================================

def explain_generative_models():
    """
    From classification to CREATION: how AI learned to generate images.

    Evolution of generative models:
    1. GANs (2014): Two networks competing -- generator vs discriminator
    2. Diffusion Models (2020): Iterative denoising -- the tech behind DALL-E
    """

    print("=" * 70)
    print("PART 2: GENERATIVE AI - From Classification to Creation")
    print("=" * 70)
    print()

    print("Until now, CNNs have been doing CLASSIFICATION:")
    print("  Image --> CNN --> 'This is a cat'")
    print()
    print("But what about the REVERSE direction?")
    print("  Random noise --> ??? --> Photorealistic image of a cat")
    print()
    print("Two breakthrough approaches solved this:")
    print()

    # ---- GANs ----
    print("-" * 50)
    print("GENERATIVE ADVERSARIAL NETWORKS (GANs, 2014)")
    print("-" * 50)
    print()
    print("  Two networks, competing in a game:")
    print()
    print("  GENERATOR (The Forger):")
    print("    Input: random noise vector (100 numbers)")
    print("    Output: a fake image")
    print("    Goal: create images so real the discriminator cannot tell")
    print()
    print("  DISCRIMINATOR (The Detective):")
    print("    Input: an image (real OR generated)")
    print("    Output: 'Real' or 'Fake' probability")
    print("    Goal: catch the forger's fakes")
    print()
    print("  TRAINING (The Arms Race):")
    print("    1. Generator creates fake images")
    print("    2. Discriminator tries to distinguish real from fake")
    print("    3. Generator improves to fool the discriminator")
    print("    4. Discriminator improves to catch the generator")
    print("    5. Repeat until the generator creates photorealistic images")
    print()
    print("  This is a MIN-MAX GAME:")
    print("    Generator tries to MAXIMIZE discriminator's errors")
    print("    Discriminator tries to MINIMIZE its classification errors")
    print("    At Nash equilibrium: generated images are indistinguishable from real")
    print()

    # ---- Diffusion Models ----
    print("-" * 50)
    print("DIFFUSION MODELS (2020) - DALL-E, Stable Diffusion, Midjourney")
    print("-" * 50)
    print()
    print("  FORWARD PROCESS (Destroying an image):")
    print("    Clean image -> Add noise -> Add noise -> ... -> Pure noise")
    print("    Like dropping ink into water: structure gradually dissolves")
    print()
    print("  REVERSE PROCESS (Creating an image):")
    print("    Pure noise -> Denoise -> Denoise -> ... -> Clean image")
    print("    The model LEARNS to undo one step of noise at a time")
    print()
    print("  TEXT CONDITIONING:")
    print("    Add a text description to guide the denoising:")
    print('    "A blue cat sitting on a rainbow" --> text encoder --> guide denoising')
    print()
    print("  ARCHITECTURE:")
    print("    - U-Net with attention mechanisms (processes the noisy image)")
    print("    - Text encoder (CLIP or T5, converts text to numbers)")
    print("    - Works in 'latent space' (compressed, not full-resolution)")
    print("    - 50-100 denoising steps to generate one image")
    print()

    print("WHY DIFFUSION WON OVER GANs:")
    print("  GANs: hard to train (mode collapse, instability)")
    print("  Diffusion: stable training, better diversity, text conditioning")
    print("  Result: DALL-E, Midjourney, Stable Diffusion are ALL diffusion models")
    print()

    print("APPLICATIONS:")
    print("  - Creative: art generation, concept visualization")
    print("  - Design: logos, product mockups, architecture renderings")
    print("  - Entertainment: game assets, film pre-visualization")
    print("  - Medical: synthetic training data (privacy-preserving)")
    print("  - Science: protein structure generation, molecule design")
    print()


# ==============================================================================
# PART 3: CLIP - Vision Meets Language
# ==============================================================================

def explain_clip():
    """
    CLIP (OpenAI, 2021): Contrastive Language-Image Pre-training

    The breakthrough: a model that understands BOTH images AND text
    in a shared embedding space.

    Training: 400 million (image, caption) pairs from the internet
    Result: zero-shot image classification, image search, and more
    """

    print("=" * 70)
    print("PART 3: CLIP - When Vision Meets Language")
    print("=" * 70)
    print()

    print("THE ARCHITECTURE:")
    print()
    print("  Image Encoder (Vision Transformer or ResNet):")
    print("    Input: image (224x224x3)")
    print("    Output: image embedding (512 numbers)")
    print()
    print("  Text Encoder (Transformer):")
    print("    Input: text description ('a photo of a cat')")
    print("    Output: text embedding (512 numbers)")
    print()
    print("  CONTRASTIVE LEARNING:")
    print("    - 'a photo of a cat' + image of a cat   --> embeddings CLOSE together")
    print("    - 'a photo of a cat' + image of a dog   --> embeddings FAR apart")
    print("    - Trained on 400 MILLION image-text pairs from the internet")
    print()

    print("BREAKTHROUGH CAPABILITIES:")
    print()
    print("  1. ZERO-SHOT CLASSIFICATION:")
    print('     Image --> CLIP --> Compare with ["a dog", "a cat", "a car"]')
    print("     No training on these specific classes needed!")
    print("     The model has never seen your classes before,")
    print("     but it UNDERSTANDS what the words mean.")
    print()
    print("  2. NATURAL LANGUAGE IMAGE SEARCH:")
    print('     Query: "sunset over mountains with purple sky"')
    print("     CLIP finds matching images in your database")
    print("     No labels, no tags, no metadata needed.")
    print()
    print("  3. IMAGE-TO-TEXT MATCHING:")
    print("     Show an image, CLIP finds the best text description")
    print("     Powers: alt-text generation, accessibility features")
    print()

    print("REAL-WORLD APPLICATIONS:")
    print()
    print("  Search & Discovery:")
    print("    - Pinterest visual search (find similar images)")
    print("    - Google Lens (understand what you photograph)")
    print("    - Video moment retrieval ('find when the dog jumps')")
    print()
    print("  Content Moderation:")
    print("    - Detect harmful content WITHOUT explicit training examples")
    print('    - "violent imagery" as text query catches images it never saw')
    print()
    print("  Accessibility:")
    print("    - Generate image descriptions for screen readers")
    print("    - Help visually impaired users understand visual content")
    print()
    print("  Creative AI:")
    print("    - DALL-E uses CLIP to understand text prompts")
    print("    - Stable Diffusion uses CLIP for text conditioning")
    print("    - CLIP guides the generation: is this image matching the text?")
    print()


# ==============================================================================
# PART 4: VISION TRANSFORMERS - Attention for Images
# ==============================================================================

def explain_vision_transformers():
    """
    Vision Transformers (ViT, 2020): applying the Transformer architecture
    (originally designed for text) to images.

    Key insight: an image is a sequence of PATCHES.

    Instead of sliding filters (convolution), ViT uses ATTENTION:
    every patch attends to every other patch simultaneously.
    This captures long-range dependencies that CNNs miss.
    """

    print("=" * 70)
    print("PART 4: VISION TRANSFORMERS - Attention for Images")
    print("=" * 70)
    print()

    print("THE SHIFT: From Local to Global Understanding")
    print()
    print("  CNNs (1998-2020):")
    print("    - Local receptive fields (3x3 convolutions)")
    print("    - See neighbors, not distant parts of the image")
    print("    - Must stack MANY layers to see the whole image")
    print("    - Translation invariant through weight sharing")
    print()
    print("  Vision Transformers (2020+):")
    print("    - Global attention (every patch sees every other patch)")
    print("    - Even in the FIRST layer, any part can attend to any other part")
    print("    - Learn spatial relationships from data (no built-in assumptions)")
    print("    - Dominant architecture for large-scale vision")
    print()

    print("HOW ViT WORKS:")
    print()
    print("  Input: 224x224x3 image")
    print("    |")
    print("  Split into patches: 14x14 = 196 patches (each 16x16 pixels)")
    print("    |")
    print("  Flatten each patch to a vector: 196 vectors of 768 dimensions")
    print("    |")
    print("  Add [CLS] token + positional embeddings")
    print("    |")
    print("  Transformer Encoder (12 layers):")
    print("    - Multi-head self-attention")
    print("    - Feed-forward network")
    print("    - Layer normalization")
    print("    |")
    print("  [CLS] token representation --> Classification head --> Output")
    print()

    print("THE ATTENTION MECHANISM:")
    print()
    print("  For each patch, compute:")
    print("    Query (Q): 'What am I looking for?'")
    print("    Key (K):   'What do I contain?'")
    print("    Value (V): 'What information do I provide?'")
    print()
    print("    Attention(Q, K, V) = softmax(Q * K^T / sqrt(d)) * V")
    print()
    print("  Result: each patch can attend to ALL other patches")
    print("    - Sky patches attend to cloud patches")
    print("    - Eye patches attend to mouth patches (same face)")
    print("    - Object patches attend to related background patches")
    print()

    print("PERFORMANCE:")
    print()
    print("  ImageNet (2021):")
    print("    ViT-Huge: 88.5% accuracy (state-of-the-art)")
    print("    EfficientNet-L2: 88.4% accuracy")
    print()
    print("  Benefits:")
    print("    - Scales better with more data (100M+ images)")
    print("    - More interpretable (attention maps show what the model focuses on)")
    print("    - Transfers well across modalities (vision, text, audio)")
    print()
    print("  Trade-offs:")
    print("    - Requires MORE data than CNNs to train well")
    print("    - Computationally expensive (quadratic attention)")
    print("    - CNNs still win on small datasets (<10K images)")
    print()

    print("THE TREND:")
    print("  2025-2026: Hybrid models (CNN + Transformer) are emerging")
    print("  CNNs for efficient local processing + Transformers for global context")
    print("  Neither approach is 'dead' -- they complement each other")
    print()


# ==============================================================================
# PART 5: THE JOURNEY SO FAR
# ==============================================================================

def the_journey():
    """
    From Script 1 to Script 4: the complete computer vision story.
    """

    print("=" * 70)
    print("PART 5: THE COMPLETE JOURNEY")
    print("From Pixels to Multi-Modal Intelligence")
    print("=" * 70)
    print()

    print("THE EVOLUTION OF COMPUTER VISION:")
    print()
    print("  1998: LeNet-5 classifies 28x28 handwritten digits")
    print("        (Script 1: you built this from scratch in NumPy)")
    print()
    print("  2012: AlexNet wins ImageNet with 8 conv layers")
    print("        (The deep learning revolution begins)")
    print()
    print("  2014: GANs learn to GENERATE images")
    print("        (From classification to creation)")
    print()
    print("  2015: ResNet enables 152-layer networks with skip connections")
    print("        (Script 3: you used MobileNetV2 for transfer learning)")
    print()
    print("  2019: EfficientNet achieves state-of-the-art with fewer parameters")
    print("        (Compound scaling: depth + width + resolution)")
    print()
    print("  2020: Vision Transformers challenge CNNs")
    print("        (Attention replaces convolution for large-scale tasks)")
    print()
    print("  2021: CLIP unifies vision and language")
    print("        (One model that sees AND reads)")
    print()
    print("  2022: Diffusion models create photorealistic images from text")
    print("        (DALL-E, Stable Diffusion, Midjourney)")
    print()
    print("  2024-2026: Multi-modal models see, read, reason, and create")
    print("        (The convergence of all modalities)")
    print()

    print("YOUR JOURNEY IN SESSIONS 37-38:")
    print()
    print("  Script 1: You understood CONVOLUTION")
    print("    The detective's flashlight scanning for patterns")
    print("    Weight sharing: 9 parameters instead of millions")
    print("    Built a CNN from scratch in NumPy")
    print()
    print("  Script 2: You built a COMPLETE PIPELINE")
    print("    Load -> Preprocess -> Augment -> Train -> Evaluate -> Visualize")
    print("    Trained a real CNN on Fashion-MNIST")
    print("    Opened the black box: saw what the network learned")
    print()
    print("  Script 3: You mastered TRANSFER LEARNING")
    print("    500 images from scratch: barely better than chance")
    print("    500 images with transfer learning: production quality")
    print("    Fine-tuning to squeeze out maximum accuracy")
    print()
    print("  Script 4: You explored THE HORIZON")
    print("    Style transfer: art meets AI")
    print("    Generative models: creating images from noise")
    print("    CLIP: understanding images and text together")
    print("    Vision Transformers: attention replaces convolution")
    print()

    print("WHAT YOU CAN NOW DO:")
    print()
    print("  Build any image classification system:")
    print("    1. Load and explore your dataset")
    print("    2. Preprocess and augment")
    print("    3. Choose: train from scratch (if large data) or transfer learn")
    print("    4. Train, evaluate, visualize")
    print("    5. Fine-tune if needed")
    print("    6. Save and deploy")
    print()
    print("  This pipeline works for:")
    print("    - Medical imaging (X-rays, retinal scans)")
    print("    - Satellite imagery (pollution, land use)")
    print("    - Manufacturing (defect detection)")
    print("    - Finance (document classification)")
    print("    - Any domain with images and categories")
    print()

    print("=" * 70)
    print("SESSIONS 37-38 COMPLETE")
    print("=" * 70)
    print()
    print("You started with: 'What is convolution?'")
    print("You ended with: training models, transfer learning, and a vision of the future.")
    print()
    print("From 150 million parameters to 9.")
    print("From weeks of training to 30 minutes.")
    print("From 100,000 images to 500.")
    print()
    print("This is the CNN revolution.")
    print("This is computer vision democratized.")
    print("This is AI accessible to everyone.")
    print("=" * 70)


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print()
    print("=" * 70)
    print("SESSION 38 - SCRIPT 4: BEYOND CNNs")
    print("The Horizon: Style Transfer, GANs, CLIP, Vision Transformers")
    print("=" * 70)
    print()

    # Part 1: Style Transfer
    explain_style_transfer()
    style_transfer_demo()
    input("Press Enter to continue to Part 2 (Generative Models)...")
    print()

    # Part 2: GANs and Diffusion
    explain_generative_models()
    input("Press Enter to continue to Part 3 (CLIP)...")
    print()

    # Part 3: CLIP
    explain_clip()
    input("Press Enter to continue to Part 4 (Vision Transformers)...")
    print()

    # Part 4: Vision Transformers
    explain_vision_transformers()
    input("Press Enter to continue to the Summary...")
    print()

    # Part 5: The Journey
    the_journey()


if __name__ == '__main__':
    main()
