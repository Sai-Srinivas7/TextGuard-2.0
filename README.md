# End-to-End NLP Pipeline: Governance & Compliance

This repository contains an end-to-end Natural Language Processing (NLP) pipeline designed for governance and compliance tasks. It demonstrates how to process project documents, train a risk classification model, and annotate PDFs with identified risks.

## Table of Contents
1. [Introduction](#introduction)
2. [Setup and Installation](#setup-and-installation)
3. [Data Loading & Preprocessing](#data-loading--preprocessing)
4. [Exploratory NLP (NER & Topic Modeling)](#exploratory-nlp-ner--topic-modeling)
5. [Advanced Risk Classification Model Training](#advanced-risk-classification-model-training)
    - [Hierarchical Attention Network (HAN) Architecture](#hierarchical-attention-network-han-architecture)
    - [HAN Data Preparation](#han-data-preparation)
    - [HAN Training Loop](#han-training-loop)
    - [HAN Evaluation](#han-evaluation)
6. [PDF Spatial Annotation](#pdf-spatial-annotation)
7. [HAN PDF Inference & Attention Extraction](#han-pdf-inference--attention-extraction)
8. [Multi-Label Risk Categorization](#multi-label-risk-categorization)
    - [Multi-Label Dataset & Training Loop](#multi-label-dataset--training-loop)
    - [Multi-Label Model Evaluation](#multi-label-model-evaluation)
    - [Multi-Label PDF Inference](#multi-label-pdf-inference)
    - [Continue Fine-Tuning](#continue-fine-tuning)
9. [HAN-C: Data Augmentation & Focal Loss Integration](#han-c-data-augmentation--focal-loss-integration)
    - [Back-Translation Data Augmentation](#back-translation-data-augmentation)
    - [HAN-C Training with Focal Loss](#han-c-training-with-focal-loss)
    - [HAN-C Evaluation](#han-c-evaluation)
10. [End-to-End Pipeline Demo](#end-to-end-pipeline-demo)
11. [Results Summary](#results-summary)

## Introduction
This project provides a comprehensive NLP pipeline for analyzing governance and compliance documents. It covers essential steps from data preparation to advanced model training and deployment for PDF annotation. Key features include:
- **Text Cleaning and Preprocessing**
- **Named Entity Recognition (NER)** and **Topic Modeling** for document understanding.
- **Risk Classification** using fine-tuned DistilBERT models with custom loss functions (Focal Loss).
- **Hierarchical Attention Networks (HAN)** for document-level classification and interpretability.
- **Multi-Label Classification** to identify multiple risk categories simultaneously.
- **Data Augmentation** using back-translation to address class imbalance.
- **PDF Annotation** to visually highlight risky sentences and critical sections.

## Setup and Installation
To run this notebook, you'll need to install the following Python libraries and download a spaCy language model. All necessary steps are included in the notebook's first code cell (`cb1e840c`).

```bash
!pip install -q python-docx spacy transformers datasets evaluate accelerate PyMuPDF
!python -m spacy download en_core_web_sm -q
```

## Data Loading & Preprocessing
The pipeline starts by loading a `Labeled_docs_data.csv` dataset. The text data is cleaned (lowercased, non-alphabetic characters removed, whitespace normalized), and labels are mapped numerically.

## Exploratory NLP (NER & Topic Modeling)
Initial NLP exploration involves:
- **Named Entity Recognition (NER)** using `spaCy` to identify entities like organizations, dates, and locations in sample text.
- **Topic Modeling** using Latent Dirichlet Allocation (LDA) to discover latent themes within the document corpus.

## Advanced Risk Classification Model Training
This section focuses on training a robust risk classification model:

### Document-Level Splits
To prevent data leakage, document-level splits are used for training and testing. Chunks belonging to the same document are kept together to ensure the model generalizes well to unseen documents.

### Tokenization
`DistilBERT` tokenizer is used to convert text into numerical input suitable for the model.

### Custom Focal Loss Trainer
A custom `FocalLossTrainer` is implemented, extending HuggingFace's `Trainer`, to handle class imbalance by down-weighting easy examples and focusing on hard-to-classify instances.

### Hierarchical Attention Network (HAN) Architecture
A custom PyTorch `HierarchicalAttentionNetwork` model is defined. This model processes individual text chunks using `DistilBERT` and then aggregates these chunk embeddings into a document-level representation using a soft-attention mechanism. This allows for both document-level prediction and interpretability by highlighting influential chunks.

### HAN Data Preparation
To train the HAN, chunk-level data is grouped by `document_id` and transformed into 3D tensors `(batch_size, num_chunks, sequence_length)` using a custom `HANDataset`.

### HAN Training & Evaluation
The HAN model is trained using a standard PyTorch training loop and evaluated on the test set using accuracy and a classification report.

## PDF Spatial Annotation
The fine-tuned DistilBERT model is used to identify and highlight high-risk sentences directly within PDF documents, providing visual cues for compliance officers.

## Multi-Label Risk Categorization
To identify specific risk types (Financial, Social, Environmental), a multi-label classification approach is implemented:

### Multi-Label HAN Architecture
A `MultiLabelHAN` model extends the binary HAN to predict multiple risk categories simultaneously. It uses `BCEWithLogitsLoss` for independent binary predictions per category.

### Data Preparation & Training
`MultiLabelHANDataset` is created to output a vector of labels per document. The model is trained with `BCEWithLogitsLoss`.

### Evaluation and PDF Inference
The multi-label model is evaluated using a classification report, and its predictions are demonstrated on a PDF document, showing detected risk categories and the most critical chunk.

## HAN-C: Data Augmentation & Focal Loss Integration
This advanced version of the HAN (`HAN-C`) incorporates two key improvements:

### Back-Translation Data Augmentation
Minority class (risk) chunks are augmented using a back-translation technique (English → Spanish → English) via `MarianMT` models to create paraphrases and increase training data diversity, thereby addressing class imbalance.

### HAN-C Training with Focal Loss
The `HAN-C` model is trained with a standalone `FocalLoss` implementation, specifically tuned with `alpha=0.75` and `gamma=2.0` to strongly prioritize the minority (risk) class and hard-to-classify examples.

### HAN-C Evaluation
Evaluation of HAN-C demonstrates its effectiveness in achieving higher recall for the minority risk class, crucial for governance and compliance applications.

## End-to-End Pipeline Demo
This section orchestrates the entire NLP pipeline, taking a raw PDF as input and performing:
1. NER Extraction
2. Binary Risk Classification (HAN)
3. Multi-Label Risk Categorization
4. PDF Spatial Annotation

## Results Summary
| Model | Risk Recall | Risk F1 | Accuracy |
|--------------------------|-------------|---------|----------|
| HAN (Cross-Entropy)      | 1.00        | 0.98    | 0.95     |
| HAN-C (Focal + Back-Translation) | 0.90        | 0.92    | 0.86     |

HAN-C demonstrates improved recall for the minority risk class, aligning with the project's target of >75% recall for identifying non-compliant documents.
