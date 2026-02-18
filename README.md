# Intelligent Object Recognition and Query System for Warehouse Robotics

## Overview

This project implements an end-to-end intelligent warehouse automation system that integrates Computer Vision, Machine Learning, and Retrieval-Augmented Generation (RAG).

The system simulates how a warehouse robot can:

1. Detect a package using computer vision
2. Classify it into categories (fragile, heavy, hazardous)
3. Retrieve relevant handling instructions automatically
4. Provide context-aware responses based on predicted category

The architecture is modular and scalable, making it suitable for real-world robotics applications.

---

## Project Structure

warehouse_robot_ai
│
├── vision
│ ├── detect.py
│ ├── box_clean.jpg
│ └── ...
│
├── ml_model
│ ├── train.py
│ ├── inference.py
│ ├── model.pth
│ └── data/
│
├── rag
│ ├── rag_system.py
│ └── documents/
│
├── integration
│ └── main.py
│
├── results
│ └── detected_output.jpg
│
├── requirements.txt
└── README.md

---

## System Architecture

### 1. Vision Module (Object Detection)

- Implemented using OpenCV
- Uses edge detection and contour analysis
- Detects package boundaries
- Computes:
  - Bounding boxes
  - Pixel dimensions
  - Center coordinates
- Saves detection result in `results/`

---

### 2. Machine Learning Module (Classification)

- Transfer learning using MobileNetV2
- Pretrained feature extractor frozen
- Final classification layer fine-tuned
- Classes:
  - Fragile
  - Heavy
  - Hazardous

Evaluation Metrics:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Note: Dataset size is intentionally small for demonstration purposes.

---

### 3. RAG Module (Document Retrieval)

- Custom warehouse knowledge base (handling protocols, safety guidelines)
- SentenceTransformers for semantic embeddings
- FAISS vector index for similarity search
- Natural language query retrieval

The system retrieves the top-k most semantically relevant documents.

---

### 4. Full Integration Pipeline

The final pipeline performs:

1. Object detection
2. Category classification
3. Context-aware query generation
4. Semantic document retrieval

Example:
- Input: Image of fragile package
- Output: Fragile handling instructions retrieved automatically

---

## Installation

### 1. Create Virtual Environment

bash
python3 -m venv venv
source venv/bin/activate

---

## Author

Nandita Udupa  
BTech Computer Science  
AI Research Internship Technical Assessment  
Email: nanditaudupa16@gmail.com

