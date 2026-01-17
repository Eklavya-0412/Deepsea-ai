# 🌊 DEEPSEA-AI

**Unveiling the Mysteries of the Deep with Environmental DNA & Artificial Intelligence**

## 📖 Overview

**DEEPSEA-AI** is a cutting-edge bioinformatic pipeline designed to analyze environmental DNA (eDNA) from deep-sea ecosystems. By leveraging advanced Deep Learning architectures—including Transformer-based DNA encoders and Variational Autoencoders (VAEs)—this tool automates taxonomic classification, assesses biodiversity, and identifies potential novel species in one of Earth's least explored frontiers.

This project bridges the gap between marine biology and artificial intelligence, providing researchers with a powerful dashboard to visualize ecosystem health and species diversity.

## ✨ Key Features

*   **🧬 Advanced DNA Encoding:** Utilizes a custom Transformer model to generate rich embeddings for DNA sequences.
*   **🏷️ Hierarchical Taxonomy:** Predicts Kingdom, Phylum, and Class with confidence scores using a multi-head neural network.
*   **🔍 Novel Species Detection:** Detects outliers and potential new species using VAEs and DBSCAN clustering.
*   **📊 Biodiversity Metrics:** Automatically calculates Species Richness, Shannon Diversity, and Simpson Diversity indices.
*   **📈 Interactive Dashboard:** A modern React-based frontend using Recharts to visualize abundance, correlations, and ecosystem health.
*   **🚀 High-Performance API:** Built with FastAPI for rapid and asynchronous processing of FASTA files.

## 🛠️ Tech Stack

### Backend & AI
*   **Python**: Core programming language.
*   **FastAPI**: High-performance web framework for APIs.
*   **PyTorch**: Deep learning framework for Transformer and VAE models.
*   **Biopython**: Tools for biological computation.
*   **Scikit-learn**: Clustering and dimensionality reduction.

### Frontend
*   **React**: UI library for building the dashboard.
*   **TypeScript**: Type-safe development.
*   **Recharts**: Composable charting library.
*   **Lucide React**: Beautiful & consistent icons.
*   **Axios**: Promise based HTTP client.

## 🚀 Getting Started

Follow these steps to set up the project locally.

### Prerequisites
*   Node.js (v14+)
*   Python (v3.8+)
*   Git

### Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/shreyashi2609/deepsea-ai.git
    cd deepsea-ai
    ```

2.  **Frontend Setup**
    Navigate to the frontend directory and install dependencies.
    ```bash
    cd frontend
    npm install
    ```

3.  **Backend Setup**
    Navigate to the backend directory, create a virtual environment, and install dependencies.
    ```bash
    cd ../backend
    
    # Create virtual environment
    python -m venv venv
    
    # Activate virtual environment
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    
    # Install dependencies
    pip install -r requirements.txt
    ```

## 🖥️ Usage

1.  **Start the Backend Server**
    Ensure your virtual environment is activated.
    ```bash
    cd src
    python main.py
    ```
    The API will run at `http://0.0.0.0:8000`.

2.  **Start the Frontend Application**
    Open a new terminal, navigate to the frontend directory, and start the development server.
    ```bash
    cd frontend
    npm start
    ```
    The application will open at `http://localhost:3000`.

3.  **Analyze Data**
    *   Upload a `.fasta` file containing eDNA sequences.
    *   Or use the **Demo Analysis** feature to see the pipeline in action with sample data.

## 🧠 Model Architecture

The core of DEEPSEA-AI consists of three main components:
1.  **DNATransformer**: Encodes raw DNA sequences into high-dimensional vectors.
2.  **HierarchicalClassifier**: A multi-head MLP that takes embeddings and predicts taxonomic ranks.
3.  **NoveltyVAE**: A Variational Autoencoder that learns the latent distribution of known species to flag anomalies as potential novel discoveries.
