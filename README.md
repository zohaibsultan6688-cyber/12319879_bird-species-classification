# Fine-Grained Bird Species Classification Using Swin Transformer and a Learnable Cross-Attention Feature Selection Module

**Author:** Zohaib Sultan  
**Matriculation Number:** 12319879  
**Course:** Applied Deep Learning  
**Assignment:** Assignment 1: Initiate  
**Project Type:** Bring Your Own Method

---

# 1. Project Overview

## 1.1 Problem Formulation

Fine-Grained Visual Categorization (FGVC) is a computer vision task that involves distinguishing between categories that look very similar but belong to the same broader group. Unlike standard image classification, where classes often have obvious differences, FGVC asks a model to pick up on subtle visual details.

Bird species classification is a good example. Different species can share similar body shapes, feather colours and sizes, so telling them apart may depend on details such as beak shape, wing patterns, head markings or tail structure. These features can be harder to identify in busy backgrounds, from different viewing angles or under varying lighting.

Convolutional Neural Networks (CNNs) have performed well in image classification. However, their feature embeddings may not always capture the small, localized details needed to distinguish closely related species. Vision Transformer models offer another approach by learning relationships between different parts of an image.

This project proposes a fine-grained bird classification system built on a pretrained Swin Transformer-Tiny backbone, with a custom, learnable Cross-Attention Feature Selection Module.

The module will use a small set of learnable queries to focus on spatial attributes extracted by the Swin Transformer. These queries will produce several localized feature embeddings, which will be combined with a global representation of the image before classification.

The project will compare three architectures:

1. ResNet-50 as a convolutional baseline.
2. Swin Transformer-Tiny as a transformer-based baseline.
3. Swin Transformer-Tiny with the proposed Cross-Attention Feature Selection Module.

All models will be evaluated using the CUB-200-2011 dataset. The main purpose is to investigate whether combining global and learnable localized feature representations can improve the recognition of visually similar bird species.

The project will focus on experimentally evaluating the proposed method rather than assuming that the additional attention module will necessarily improve accuracy.

## 1.2 Project Objectives

The main objectives of this project are:

1. To implement a ResNet-50 model as a conventional deep-learning baseline.
2. To fine-tune an ImageNet-pretrained Swin Transformer-Tiny on the CUB-200-2011 dataset.
3. To design and implement a custom Cross-Attention Feature Selection Module.
4. To investigate how learnable attention queries interact with spatial image features.
5. To combine global and localized feature representations for final classification.
6. To evaluate the effect of different attention configurations and data augmentation strategies.
7. To compare classification performance and computational requirements across the selected models.
8. Visualize attention patterns and examine which image regions receive the most attention.
9. Build a simple interactive application for testing the trained model.


## 1.3 Research Questions

**RQ1: Global and Local Feature Representations**

Does combining global image features with localized features selected through learnable cross-attention improve Top-1 accuracy compared with conventional global average pooling?

**RQ2: Number of Attention Queries**

How does using 2, 4 or 8 learnable queries affect classification performance, computational cost and inference latency?

**RQ3: Data Augmentation**

How do Random Resized Cropping, horizontal flipping, color jitter and random erasing affect the model’s performance on unseen bird images?

**RQ4: Model Performance and Computational Efficiency**

How do ResNet-50, Swin Transformer-Tiny and the proposed model compare in accuracy, training time, number of parameters, GPU memory use and inference latency?

---

# 2. Proposed Methodology and Technical Architecture

## 2.1 Overall Architecture

The project will compare three model configurations using the same dataset splits and evaluation procedures.

The proposed architecture will contain four main components:

1. Image preprocessing and feature preparation.
2. Swin Transformer-Tiny feature extraction.
3. Learnable Cross-Attention Feature Selection Module.
4. Global-local feature fusion and classification.

ResNet-50 and standard Swin Transformer-Tiny will serve as the CNN and transformer baselines. The third model will add the custom attention module so its contribution can be measured.

## 2.2 Model 1: ResNet-50 Baseline

ResNet-50 will be used as the convolutional neural network baseline.

ResNet introduces residual connections that help information and gradients propagate through deeper neural networks. This architecture has been widely used for image classification and provides a suitable reference for comparison.

The model will use ImageNet-1K pretrained weights. Its original classification layer will be replaced with a new fully connected layer containing 200 output neurons.

**Selected configuration:**

| Parameter | Configuration |
|---|---|
| Architecture | ResNet-50 |
| Initial weights | ImageNet-1K pretrained |
| Input resolution | 224 × 224 |
| Feature extraction | ResNet-50 backbone |
| Feature aggregation | Global Average Pooling |
| Classification layer | Fully connected layer |
| Output classes | 200 |
| Loss function | Cross-Entropy with label smoothing |

The ResNet-50 model will provide a conventional CNN reference for evaluating the performance of transformer-based architectures.

## 2.3 Model 2: Swin Transformer-Tiny Baseline

The second model will use Swin Transformer-Tiny as its feature extraction backbone.

Swin Transformer uses hierarchical feature representations and shifted-window self-attention. Instead of calculating self-attention across every image position simultaneously, it performs attention within local windows and uses shifted windows to exchange information between neighbouring regions.

This design provides a hierarchical visual representation while controlling the computational cost of attention.

The model will use ImageNet-1K pretrained weights and will be fine-tuned on the CUB-200-2011 dataset.

The original classification head will be replaced with a new classifier containing 200 output classes.

**Selected configuration:**

| Parameter | Configuration |
|---|---|
| Architecture | Swin Transformer-Tiny |
| Initial weights | ImageNet-1K pretrained |
| Input resolution | 224 × 224 |
| Feature extraction | Hierarchical Swin backbone |
| Feature aggregation | Global Average Pooling |
| Classification layer | Fully connected layer |
| Output classes | 200 |
| Loss function | Cross-Entropy with label smoothing |

For a 224 × 224 input, the final Swin-Tiny feature representation is expected to contain a 7 × 7 spatial grid with 768 feature channels. The actual output shape will be checked during implementation using the selected model configuration.

This model will serve as the transformer baseline against which the proposed attention-based architecture will be compared.

## 2.4 Model 3: Proposed Swin Transformer with Cross-Attention Feature Selection

The proposed model builds on Swin Transformer-Tiny by adding a learnable Cross-Attention Feature Selection Module.

Rather than relying on a single globally pooled feature vector, the model will learn to extract several localized feature embeddings from the image.

The module won’t use predefined bird-part locations. Instead, it will learn focus patterns from the training data.

### A. Swin Transformer Feature Extraction

A 224 × 224 × 3 image will be passed through the pretrained Swin Transformer-Tiny backbone.

The final feature vector is expected to have the following dimensions:

\[
F \in \mathbb{R}^{B \times 7 \times 7 \times 768}
\]

Where:

- \(B\) represents the batch size.
- \(7 \times 7\) represents the final spatial feature grid.
- \(768\) represents the feature dimension.

The spatial grid will be flattened into 49 feature tokens:

\[
X \in \mathbb{R}^{B \times 49 \times 768}
\]

These tokens will contain visual information extracted from different spatial regions of the input image.

### B. Feature Projection

Before applying cross-attention, the 768-dimensional spatial features will be projected into a 512-dimensional embedding space.

\[
X_p = W_pX+b_p
\]

Where:

- \(X_p\) represents the projected spatial features.
- \(W_p\) represents the learnable projection matrix.
- \(b_p\) represents the bias term.

The resulting representation will have dimensions:

\[
X_p \in \mathbb{R}^{B \times 49 \times 512}
\]

This projection will make the spatial features compatible with the learnable attention queries.

### C. Learnable Attention Queries

The proposed module will initially use four learnable attention queries.

The query matrix will have the following dimensions:

\[
Q \in \mathbb{R}^{4 \times 512}
\]

For each input batch, these queries will be expanded across the batch dimension.

\[
Q_b \in \mathbb{R}^{B \times 4 \times 512}
\]

Each query will attend to the spatial feature tokens and produce its own representation. The queries will be learnable parameters, updated during training through backpropagation.

To see how the number of queries affects performance, the module will also be tested with 2 and 8 queries.

### D. Multi-Head Cross-Attention

The projected spatial elements will provide the keys and values for cross-attention, while the learnable queries will provide the queries.

The cross-attention operation is defined as:

\[
Attention(Q,K,V)=
Softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\]

Where:

- \(Q\) represents the learnable query vectors.
- \(K\) represents the spatial feature keys.
- \(V\) represents the spatial feature values.
- \(d_k\) represents the dimension of each attention head.

The initial configuration will use:

| Parameter | Selected Value |
|---|---|
| Number of queries | 4 |
| Embedding dimension | 512 |
| Number of attention heads | 8 |
| Dimension per head | 64 |
| Number of spatial tokens | 49 |
| Attention mechanism | Multi-Head Cross-Attention |

Since the embedding dimension is 512 and the module uses 8 attention heads, each head will operate on a 64-dimensional representation.

For each query, attention weights will be calculated across the 49 spatial tokens. The weighted spatial features will then be combined to produce four attention-based feature representations.

The output will have the following dimensions:

\[
F_{local} \in \mathbb{R}^{B \times 4 \times 512}
\]

A residual connection and layer normalization will be applied to the attention output. A feed-forward layer will then refine the resulting query representations.

This design will help preserve the learnable query information while allowing the module to incorporate spatial image features.

### E. Global Feature Extraction

In addition to localized feature representations, the model will retain a global representation of the input image.

The global feature vector will be obtained by average pooling the final Swin Transformer feature map:

\[
F_{global}=GAP(F)
\]

The resulting 768-dimensional representation will be projected into the same 512-dimensional embedding space:

\[
F_{global}^{p}\in\mathbb{R}^{B\times512}
\]

This global representation will provide information about the overall visual structure of the bird.

### F. Global-Local Feature Fusion

The four attention-based representations will be flattened into a single vector:

\[
F_{local}^{flat}\in\mathbb{R}^{B\times2048}
\]

The projected global feature vector will then be concatenated with the localized feature vector:

\[
F_{combined}=
Concat(F_{global}^{p},F_{local}^{flat})
\]

The combined feature dimension will be:

\[
512+(4\times512)=2560
\]

The fused representation will be passed through a classification head consisting of:

1. Fully connected layer: 2560 to 1024 dimensions.
2. GELU activation.
3. Dropout with probability 0.3.
4. Final fully connected layer: 1024 to 200 output classes.

The output layer will generate classification scores for the 200 bird species.

### G. Classification Loss Function

The proposed model will use Cross-Entropy Loss with label smoothing.

The loss function is defined as:

\[
L=-\sum_{i=1}^{C}y_i\log(p_i)
\]

Where:

- \(C\) represents the number of classes.
- \(y_i\) represents the smoothed target probability.
- \(p_i\) represents the predicted probability for class \(i\).

The label smoothing factor will initially be set to 0.1.

Label smoothing will be investigated as a regularization technique to reduce excessively confident predictions.

The same primary loss configuration will be applied to all three models to maintain a consistent experimental comparison.

---

# 3. Selected Project Type

**Project Type: Bring Your Own Method**

This project involves implementing a custom attention-based feature selection module and integrating it into an existing pretrained transformer architecture.

The project does not attempt to develop a completely new transformer backbone. Instead, it investigates whether a specifically designed cross-attention component can improve the feature representation used for fine-grained bird species classification.

The proposed architecture will be evaluated against two reference models:

- ResNet-50.
- Swin Transformer-Tiny without the additional cross-attention module.

The main technical contribution will be the design, implementation and experimental evaluation of the learnable Cross-Attention Feature Selection Module.

---

# 4. Dataset Description

## 4.1 Selected Dataset

**Dataset:** Caltech-UCSD Birds-200-2011 (CUB-200-2011)

CUB-200-2011 is a publicly available benchmark for fine-grained bird species recognition. 

It contains 11,788 images across 200 species, many of which look similar. The dataset also provides bounding boxes and bird-part annotations.

These features make CUB-200-2011 a useful dataset for studying attention-based feature extraction as well as fine-grained image classification.

### Dataset Specifications

| Attribute | Description |
|---|---|
| Dataset name | CUB-200-2011 |
| Total images | 11,788 |
| Number of classes | 200 |
| Official training images | 5,994 |
| Official testing images | 5,794 |
| Image type | RGB |
| Image resolution | Variable |
| Additional annotations | Bounding boxes and part annotations |

## 4.2 Dataset Challenges

The dataset presents several challenges:

**1. High Intra-Class Variation**

Images of the same species can differ in pose, viewing angle, lighting and background.

**2. High Inter-Class Similarity**

Different species may have similar feather patterns, body shapes and colours, making them difficult to distinguish.

**3. Background Clutter**

Images may include trees, leaves, branches and other objects that are not directly relevant to identifying the bird.

**4. Small Discriminative Regions**

Important visual features may occupy only a small part of an image.

These problems make the dataset suitable for investigating whether learnable, localized feature embeddings can improve classification performance.

## 4.3 Data Splitting Strategy

The official training and testing partitions of CUB-200-2011 will be preserved. 

The official training partition will be divided into:

- 90% training data.
- 10% validation data.

A stratified split will help ensure that all 200 species are represented in both subsets. The random seed will be set to 42 to support reproducibility.

The official test partition will remain separate from the training and validation data. It will be used only for final evaluation, after model selection and hyperparameter decisions are complete. Test images will not be used for training, choosing augmentations or selecting model checkpoints.

## 4.4 Image Preprocessing

All input images will be converted into RGB format.

The preprocessing pipeline will include:

1. Loading images using PIL.
2. Applying the required resizing and cropping operations.
3. Converting images into PyTorch tensors.
4. Normalizing the image channels using ImageNet statistics.

The normalization parameters will be:

**Mean:**

\[
\mu=[0.485,0.456,0.406]
\]

**Standard Deviation:**

\[
\sigma=[0.229,0.224,0.225]
\]

Training images will use random transformations, while validation and test images will use deterministic preprocessing.

## 4.5 Data Augmentation

Data augmentation will be applied to training images to improve generalization.

The initial augmentation configuration will be:

| Augmentation | Configuration |
|---|---|
| Random Resized Crop | Output size 224 × 224, scale 0.7–1.0 |
| Horizontal Flip | Probability 0.5 |
| Color Jitter | Brightness, contrast and saturation: 0.2 |
| Random Erasing | Probability 0.25 |

Random Resized Cropping will introduce variation in image scale and the visible portions of each image.

Horizontal flipping will increase variation in bird orientation.

Color Jitter will make moderate adjustments to brightness, contrast and saturation.

Random Erasing will remove small, randomly selected regions during training, allowing us to investigate how well the model learns from incomplete visual information.

Validation and test images will not receive random augmentations. Instead, they will be resized, centre-cropped and normalized along with ImageNet normalization.

An ablation experiment will compare the selected augmentation pipeline with a simpler preprocessing configuration.

---

# 5. Experimental Design

## 5.1 Model Configurations

Three model configurations will be evaluated.

| Model | Architecture | Experimental Purpose |
|---|---|---|
| Model 1 | ResNet-50 | CNN baseline |
| Model 2 | Swin Transformer-Tiny | Transformer baseline |
| Model 3 | Swin Transformer-Tiny + Cross-Attention Feature Selection | Proposed architecture |

All models will use ImageNet-1K pretrained weights.

The same training, validation and testing partitions will be used for all configurations.

The comparison will investigate whether the proposed attention module provides a measurable improvement over global feature extraction.

## 5.2 Training Configuration

The initial training configuration will be:

| Parameter | Selected Value |
|---|---|
| Programming language | Python |
| Deep-learning framework | PyTorch |
| Model library | timm |
| Optimizer | AdamW |
| Backbone learning rate | 0.00001 |
| New classification/attention layers learning rate | 0.0001 |
| Batch size | 16 |
| Maximum epochs | 30 |
| Weight decay | 0.05 |
| Learning-rate scheduler | Cosine Annealing |
| Warm-up period | 3 epochs |
| Loss function | Cross-Entropy |
| Label smoothing | 0.1 |
| Random seed | 42 |
| Mixed precision | Automatic Mixed Precision |
| Early stopping patience | 7 epochs |

AdamW will be used because it supports decoupled weight decay and is commonly applied in transformer training.

Separate learning rates will be assigned to the pretrained backbone and newly introduced layers. The lower learning rate for the pretrained backbone will help limit unnecessary changes to previously learned visual representations.

The learning-rate schedule will include a three-epoch warm-up followed by cosine decay.

Automatic Mixed Precision will be used when supported by the available GPU. It can reduce memory consumption and may improve training speed.

Early stopping will monitor validation Top-1 accuracy. Training will stop if the validation accuracy does not improve for seven consecutive epochs.

The best checkpoint will be saved based on validation Top-1 accuracy.

The batch size may be reduced if GPU memory is insufficient. Any such change will be documented and applied consistently when comparing the relevant models.

## 5.3 Ablation Studies

Ablation studies will be conducted to investigate the contribution of individual components of the proposed architecture.

**Experiment A: Global Feature Pooling**

Swin Transformer-Tiny will be evaluated using global average pooling without the additional attention module.

This experiment will establish the transformer baseline.

**Experiment B: Cross-Attention Integration**

The Cross-Attention Feature Selection Module will be added to Swin Transformer-Tiny.

The results will be compared with the global pooling configuration to investigate the contribution of localized feature representations.

**Experiment C: Number of Learnable Queries**

Three configurations will be evaluated:

| Configuration | Number of Queries |
|---|---:|
| Query Configuration 1 | 2 |
| Query Configuration 2 | 4 |
| Query Configuration 3 | 8 |

The embedding dimension and number of attention heads will remain fixed.

This experiment will investigate whether increasing the number of learnable queries improves classification performance and whether any improvement justifies the additional computational cost.

**Experiment D: Data Augmentation**

The proposed model will be trained using two preprocessing configurations:

- Basic preprocessing with horizontal flipping.
- The complete selected augmentation pipeline.

The results will be compared to investigate the contribution of the additional augmentation operations.

The ablation studies will use the same validation procedure and training configuration wherever possible.

---

# 6. Evaluation Strategy

All models will be evaluated using consistent procedures.

The official test partition will only be used for final evaluation after the best model configurations have been selected using validation data.

## 6.1 Classification Metrics

The following metrics will be reported.

**1. Top-1 Accuracy**

Top-1 accuracy measures the percentage of images for which the model's highest-probability prediction matches the actual bird species.

\[
Accuracy=
\frac{Correct\ Predictions}{Total\ Predictions}
\]

**2. Top-5 Accuracy**

Top-5 accuracy measures whether the correct bird species appears among the five highest-probability predictions.

**3. Macro-Precision**

Macro-precision calculates precision separately for each class and then takes the average across all 200 species.

**4. Macro-Recall**

Macro-recall calculates recall independently for each class and averages the results equally.

**5. Macro-F1 Score**

Macro-F1 combines precision and recall using their harmonic mean.

\[
F1=
2\times
\frac{Precision\times Recall}
{Precision+Recall}
\]

Macro-averaged metrics will provide a class-level view of performance rather than allowing more frequent classes to dominate the evaluation.

A confusion matrix and per-class classification results will also be generated to identify species that are frequently confused with one another.

## 6.2 Computational Performance

Classification accuracy alone will not provide a complete comparison of the selected architectures.

The following computational measurements will also be recorded:

- Total training time.
- Average inference latency per image.
- Peak GPU memory consumption.
- Total trainable and non-trainable parameter counts.
- Model checkpoint size.

Inference latency will be measured using a batch size of one after an initial warm-up period. Where GPU execution is used, device synchronization will be applied during timing to obtain more reliable measurements.

The same hardware and measurement procedure will be used when comparing models.

These measurements will help determine whether any improvement in classification performance justifies the additional computational requirements of the proposed architecture.

## 6.3 Attention Visualization

Attention visualization will be used to investigate how the proposed module distributes attention across different spatial regions.

The cross-attention weights will be extracted from the attention module.

For each query, the attention weights over the 49 spatial tokens will be reshaped into a 7 × 7 attention map.

The maps will then be resized to the original image dimensions and displayed as heatmap overlays.

This will allow visual inspection of the image regions receiving higher attention.

The CUB-200-2011 bounding-box annotations may also be used for an exploratory localization analysis. Attention maps can be thresholded to identify highly attended regions and compared with the corresponding bird bounding boxes using Intersection over Union (IoU).

This analysis will be treated as an exploratory measure rather than proof that the model has learned anatomically correct bird parts. Attention maps show where the module assigns attention, but they do not automatically establish that those regions caused the final prediction.

---

# 7. Work-Breakdown Structure and Time Estimation

The project will be divided into seven major stages.

| Task ID | Task Description | Estimated Hours |
|---|---|---:|
| WBS 1 | Dataset collection, inspection, preprocessing and data splitting | 8 |
| WBS 2 | Implementation of ResNet-50 and Swin Transformer-Tiny baselines | 12 |
| WBS 3 | Development of the Cross-Attention Feature Selection Module and feature fusion classifier | 14 |
| WBS 4 | Model training, hyperparameter configuration and augmentation experiments | 16 |
| WBS 5 | Ablation studies, performance evaluation and attention visualization | 10 |
| WBS 6 | Development of an interactive image classification application | 6 |
| WBS 7 | Final report writing, documentation and presentation preparation | 8 |
| **Total** | **Estimated Project Effort** | **74 hours** |

The implementation will follow a staged process.

First, the dataset and preprocessing pipeline will be prepared. The ResNet-50 and standard Swin Transformer-Tiny models will then be implemented and evaluated. 

After establishing the baseline results, the custom cross-attention module will be developed and integrated into the Swin Transformer architecture. 

The models will then undergo training, validation, feature ablation, and final evaluation

The interactive application will be developed after selecting the final model. 

The estimated schedule may be adjusted depending on GPU availability, training duration, and implementation difficulty.
---

# 8. Planned Tools and Technologies

The project will use Python as its primary programming language.

## 8.1 Development Environment

- Google Colab.
- Jupyter Notebook.
- PyCharm, if required.

## 8.2 Deep-Learning Frameworks and Model Libraries

- PyTorch for model implementation and training.
- Torchvision for image transformations and pretrained CNN models.
- PyTorch Image Models (timm) for Swin Transformer-Tiny.

## 8.3 Image Processing and Numerical Libraries

- PIL for image loading and processing.
- OpenCV for additional image operations, if required.
- NumPy for numerical operations.
- Matplotlib for plotting and visualization.

## 8.4 Model Evaluation

- Scikit-learn for classification metrics and confusion matrices.
- Pandas for experimental result management.
- Matplotlib for performance graphs and attention visualization.

## 8.5 Interactive Application

Streamlit or Gradio will be used to develop a simple image classification interface.

The application is expected to provide:

1. Image upload functionality.
2. Predicted bird species.
3. Top-5 predicted species.
4. Prediction confidence scores.
5. Attention visualization for the proposed model.

## 8.6 Version Control

Git and GitHub will be used to maintain the source code, track implementation changes and document experimental configurations.

Only Python and PyTorch will be used as the primary implementation environment to maintain consistency across the project.

---

# 9. Expected Outcomes

The expected outcomes of this project include:

1. A functioning ResNet-50 baseline for classifying 200 bird species.
2. A fine-tuned Swin Transformer-Tiny model trained on CUB-200-2011.
3. A custom Cross-Attention Feature Selection Module integrated into the Swin Transformer architecture.
4. A global-local feature fusion mechanism for combining overall image information with localized feature representations.
5. A quantitative comparison between conventional global pooling and cross-attention-based feature extraction.
6. An ablation study investigating the effect of different query configurations.
7. An evaluation of classification performance and computational requirements.
8. Attention heatmaps for visual inspection of the proposed module.
9. A simple interactive application for testing the final trained model.

The central hypothesis is that combining global image representations with learnable localized feature representations may improve the recognition of visually similar bird species.

However, the proposed architecture will not be assumed to outperform the baseline models in advance. Its effectiveness will be determined through the experimental results.

If the proposed model achieves higher classification performance, the additional computational cost will also be considered. If it does not outperform the baseline, the results will still be analyzed to identify possible reasons and limitations.

---

# 10. Project Scope and Limitations

## 10.1 Project Scope

This project will focus specifically on fine-grained bird species classification using the CUB-200-2011 dataset.

The study will cover:

- Implementation of ResNet-50 and Swin Transformer-Tiny.
- Fine-tuning of pretrained ImageNet models.
- Development of a custom learnable cross-attention module.
- Global-local feature fusion.
- Comparative evaluation and ablation studies.
- Attention visualization.
- Development of a basic image classification application.

The project will focus on image-level bird species classification rather than real-time object detection or video-based recognition.

The Swin Transformer backbone will be used as a pretrained feature extractor. The project will not attempt to develop a completely new transformer backbone from scratch.

## 10.2 Project Limitations

The project may face several limitations.

**Dataset Size**

CUB-200-2011 contains 11,788 images across 200 classes. The relatively limited number of images per class may affect the ability of models to generalize to different environments.

**Computational Resources**

Training transformer-based architectures may require considerable GPU memory and processing time. The final training configuration may therefore depend on available hardware.

**Image Resolution**

The selected input resolution of 224 × 224 pixels may reduce the visibility of very small visual details.

**Attention Interpretation**

Learnable attention queries may produce different spatial attention patterns. However, these patterns cannot automatically be interpreted as specific anatomical bird parts.

**Generalization**

The model will be trained and evaluated on CUB-200-2011. Its performance on bird images collected from unrelated sources may differ because of changes in background, lighting, image quality and bird appearance.

The study will report the actual experimental findings, including cases where the proposed architecture does not outperform the baseline models.

---

# 11. Scientific References

1. Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., & Guo, B. (2021). *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows*. Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 10012–10022. https://doi.org/10.1109/ICCV48922.2021.00986

2. Wah, C., Branson, S., Welinder, P., Perona, P., & Belongie, S. (2011). *The Caltech-UCSD Birds-200-2011 Dataset*. California Institute of Technology, CNS-TR-2011-001.

3. He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition*. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 770–778. https://doi.org/10.1109/CVPR.2016.90

4. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). *Attention Is All You Need*. Advances in Neural Information Processing Systems, 30.

5. Dosovitskiy, A., Beyer, L., Kolesnikov, A., et al. (2021). *An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale*. International Conference on Learning Representations (ICLR). https://arxiv.org/abs/2010.11929

6. Loshchilov, I., & Hutter, F. (2019). *Decoupled Weight Decay Regularization*. International Conference on Learning Representations (ICLR). https://arxiv.org/abs/1711.05101

---

# 12. Final Project Goal

The goal of this project is to find out whether adding a learnable Cross-Attention Feature Selection Module to a pretrained Swin Transformer-Tiny can improve fine-grained bird species classification.

To investigate this, the project will compare a conventional CNN, a standard transformer and the proposed attention-based model. The comparison will look at how global and localized feature encodings relate to classification performance and computational capability.

The conclusions will be drawn from the experimental results, not from assumed improvements.
