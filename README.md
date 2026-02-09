Food Image Generation using Conditional GAN (cGAN)



Overview
This project implements a Conditional Generative Adversarial Network (cGAN using PyTorch to generate realistic food images conditioned on class labels. The model is trained on a subset of the Food-101 dataset and learns to synthesize food images corresponding to specific categories. This project demonstrates adversarial learning, conditional image synthesis, and handling of real-world image data.


Key Highlights
- Implemented a label-conditioned image generation pipeline** using cGANs in PyTorch.
- Designed custom Generator and Discriminator architectures for RGB food images.
- Visualized generated images to monitor training quality and convergence.
- Experimented with real-world dataset constraints such as limited compute resources.
- Conceptually inspired by prior research in food image synthesis using cGANs.


Reference Paper (Inspired This Work)
This project draws conceptual inspiration from the following research that applies **Conditional GANs for food image generation**:

Food Image Generation using a Large Amount of Food Images with Conditional GAN: RamenGAN and RecipeGAN
https://mm.cs.uec.ac.jp/e/pub/conf18/180715shimok2_0.pdf

The referenced work demonstrates the effectiveness of cGANs for category-conditioned food image synthesis, aligning closely with the methodology used in this project.


How It Works
- The Generator takes random noise along with a class label and generates a food image.
- The Discriminator evaluates image–label pairs and predicts whether the image is real or generated.
- Both networks are trained adversarially using BCEWithLogitsLoss and the Adam optimizer.
- Label conditioning enables controlled image generation for specific food categories.


Dataset
- Dataset: Food-101  
- Source: Hugging Face Datasets  
- Preprocessing Steps:
  - Convert all images to RGB
  - Resize images to 64×64
  - Normalize pixel values to [-1, 1]

A small subset of food categories is used to reduce computational cost and training time.


Training Details & Constraints
- In this implementation, the model is trained for 5 epochs due to GPU and time constraints.
- While this is sufficient to demonstrate learning behavior and conditional generation, the generated images are not fully refined.
- For significantly sharper and more realistic images, it is recommended to train the model for 50–100 epochs, preferably using a GPU.

This trade-off highlights practical considerations when training deep generative models under limited resources.


Installation & Requirements

Ensure you have Python 3.9 or higher installed.  
Install all required dependencies using:

```bash
pip install torch torchvision datasets numpy pandas matplotlib tqdm
