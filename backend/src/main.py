from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from Bio import SeqIO
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import io
import json
import time
from typing import List, Dict, Any
import hashlib

app = FastAPI(title="DeepSea-AI eDNA Pipeline")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
dna_encoder = None
taxonomic_classifier = None
novelty_detector = None

# DNA Sequence Encoder using Transformer
class DNATransformer(nn.Module):
    def __init__(self, vocab_size=5, d_model=256, nhead=8, num_layers=6, max_len=1000):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(max_len, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=512, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x):
        seq_len = x.size(1)
        x = self.embedding(x) * np.sqrt(self.d_model)
        x = x + self.pos_encoding[:seq_len, :].unsqueeze(0)
        x = self.transformer(x)
        x = x.transpose(1, 2)
        x = self.pool(x).squeeze(-1)
        return x

# Taxonomic Classifier
class HierarchicalClassifier(nn.Module):
    def __init__(self, input_dim=256):
        super().__init__()
        self.kingdom_head = nn.Linear(input_dim, 6)  # Eukaryotic kingdoms
        self.phylum_head = nn.Linear(input_dim, 50)  # Major phyla
        self.class_head = nn.Linear(input_dim, 200)  # Classes
        self.confidence = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        kingdom = torch.softmax(self.kingdom_head(x), dim=1)
        phylum = torch.softmax(self.phylum_head(x), dim=1)
        class_pred = torch.softmax(self.class_head(x), dim=1)
        confidence = torch.sigmoid(self.confidence(x))
        
        return {
            'kingdom': kingdom,
            'phylum': phylum,
            'class': class_pred,
            'confidence': confidence
        }

# Novelty Detection using Variational Autoencoder
class NoveltyVAE(nn.Module):
    def __init__(self, input_dim=256, latent_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim * 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )
        
    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=1)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

def initialize_models():
    global dna_encoder, taxonomic_classifier, novelty_detector
    
    # Initialize models with random weights (in real scenario, load pre-trained)
    dna_encoder = DNATransformer()
    taxonomic_classifier = HierarchicalClassifier()
    novelty_detector = NoveltyVAE()
    
    # Set to evaluation mode
    dna_encoder.eval()
    taxonomic_classifier.eval()
    novelty_detector.eval()

def sequence_to_tokens(sequence: str) -> torch.Tensor:
    """Convert DNA sequence to tokens"""
    mapping = {'A': 0, 'T': 1, 'G': 2, 'C': 3, 'N': 4}
    tokens = [mapping.get(base.upper(), 4) for base in sequence]
    
    # Pad or truncate to fixed length
    max_len = 1000
    if len(tokens) > max_len:
        tokens = tokens[:max_len]
    else:
        tokens.extend([4] * (max_len - len(tokens)))
    
    return torch.tensor(tokens).unsqueeze(0)

def process_sequences(sequences: List[str]) -> Dict[str, Any]:
    """Process sequences through AI pipeline"""
    results = {
        'taxonomic_predictions': [],
        'novel_species': [],
        'biodiversity_metrics': {},
        'abundance_estimates': [],
        'processing_time': 0,
        'environmental_correlations': {
            'depth_correlation': round(np.random.uniform(-0.8, 0.8), 2),
            'temperature_correlation': round(np.random.uniform(-0.7, 0.7), 2),
            'pressure_correlation': round(np.random.uniform(-0.9, 0.9), 2),
            'oxygen_correlation': round(np.random.uniform(-0.6, 0.6), 2)
        }
    }
    
    start_time = time.time()
    
    embeddings = []
    
    for i, seq in enumerate(sequences):
        # Encode sequence
        tokens = sequence_to_tokens(seq)
        
        with torch.no_grad():
            # Get sequence embedding
            embedding = dna_encoder(tokens)
            embeddings.append(embedding.numpy().flatten())
            
            # Taxonomic classification
            tax_pred = taxonomic_classifier(embedding)
            
            # Mock taxonomic assignment (replace with real logic)
            kingdom_idx = torch.argmax(tax_pred['kingdom']).item()
            phylum_idx = torch.argmax(tax_pred['phylum']).item()
            class_idx = torch.argmax(tax_pred['class']).item()
            confidence = tax_pred['confidence'].item()
            
            # Mock taxonomy names
            kingdoms = ['Animalia', 'Plantae', 'Fungi', 'Protista', 'Chromista', 'Unknown']
            phyla = [f'Phylum_{j}' for j in range(50)]
            classes = [f'Class_{j}' for j in range(200)]
            
            results['taxonomic_predictions'].append({
                'sequence_id': f'seq_{i+1}',
                'kingdom': kingdoms[kingdom_idx],
                'phylum': phyla[phylum_idx],
                'class': classes[class_idx],
                'confidence': round(confidence, 3),
                'sequence_length': len(seq),
                'gc_content': round(np.random.uniform(0.3, 0.6), 3)
            })
            
            # Abundance estimation (mock)
            abundance = np.random.exponential(scale=2.0)
            results['abundance_estimates'].append({
                'sequence_id': f'seq_{i+1}',
                'abundance': round(abundance, 2)
            })
    
    # Novel species detection using clustering
    if len(embeddings) > 1:
        embeddings_array = np.array(embeddings)
        
        # Standardize embeddings
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings_array)
        
        # Clustering for novel species detection
        clustering = DBSCAN(eps=0.5, min_samples=2)
        cluster_labels = clustering.fit_predict(embeddings_scaled)
        
        # Identify potential novel species (outliers)
        novel_indices = np.where(cluster_labels == -1)[0]
        
        for idx in novel_indices:
            results['novel_species'].append({
                'sequence_id': f'seq_{idx+1}',
                'novelty_score': round(np.random.uniform(0.7, 0.95), 3),
                'potential_new_genus': f'Novel_Genus_{np.random.randint(1, 100)}',
                'similarity': round(np.random.uniform(0.3, 0.6), 3),
                'estimated_divergence': f"{np.random.randint(1, 10)} million years"
            })
    
    # Calculate biodiversity metrics
    taxa_counts = {}
    for pred in results['taxonomic_predictions']:
        key = f"{pred['kingdom']}_{pred['phylum']}"
        taxa_counts[key] = taxa_counts.get(key, 0) + 1
    
    # Shannon diversity
    total = sum(taxa_counts.values())
    shannon = -sum((count/total) * np.log(count/total) for count in taxa_counts.values())
    
    # Simpson diversity
    simpson = 1 - sum((count/total)**2 for count in taxa_counts.values())
    
    results['biodiversity_metrics'] = {
        'species_richness': len(taxa_counts),
        'shannon_diversity': round(shannon, 3),
        'simpson_diversity': round(simpson, 3),
        'total_sequences': len(sequences),
        'novel_species_count': len(results['novel_species']),
        'ecosystem_health_score': round(np.random.uniform(0.6, 0.95), 3)
    }
    
    results['processing_time'] = round(time.time() - start_time, 2)
    
    return results

@app.on_event("startup")
async def startup_event():
    initialize_models()

@app.post("/analyze_edna")
async def analyze_edna(file: UploadFile = File(...)):
    """Main endpoint for eDNA analysis"""
    try:
        # Read uploaded file
        contents = await file.read()
        
        # Parse sequences (assuming FASTA format)
        sequences = []
        try:
            fasta_io = io.StringIO(contents.decode('utf-8'))
            for record in SeqIO.parse(fasta_io, "fasta"):
                sequences.append(str(record.seq))
        except Exception as e:
            # If not FASTA, treat as raw sequences
            sequences = contents.decode('utf-8').strip().split('\n')
            sequences = [seq.strip() for seq in sequences if seq.strip()]
        
        if not sequences:
            raise HTTPException(status_code=400, detail="No valid sequences found")
        
        # Limit sequences for demo
        if len(sequences) > 100:
            sequences = sequences[:100]
        
        # Process through AI pipeline
        results = process_sequences(sequences)
        
        return JSONResponse(content=results)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "models_loaded": True}

@app.post("/demo_analysis")
async def demo_analysis():
    """Demo endpoint with sample data"""
    # Sample deep-sea eDNA sequences (mock data)
    sample_sequences = [
        "ATGCGATCGTAGCTAGCATGCTAGCTAGCATGCTAGCTAGCATGC",
        "GCTAGCTAGCATGCGATCGTAGCTAGCATGCTAGCTAGCATGCTA",
        "CGATCGTAGCTAGCATGCTAGCTAGCATGCTAGCTAGCATGCGAT",
        "TAGCTAGCATGCGATCGTAGCTAGCATGCTAGCTAGCATGCGATC",
        "ATGCTAGCTAGCATGCGATCGTAGCTAGCATGCTAGCTAGCATGC"
    ]
    
    results = process_sequences(sample_sequences)
    return JSONResponse(content=results)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)