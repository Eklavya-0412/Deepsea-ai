import React, { useState, useCallback } from 'react';
import axios from 'axios';
import { Upload, Brain, Dna, Fish, BarChart3, AlertCircle, Zap, Eye, Globe } from 'lucide-react';
import { PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import './App.css';

interface TaxonomicPrediction {
  sequence_id: string;
  kingdom: string;
  phylum: string;
  class: string;
  confidence: number;
  sequence_length: number;
  gc_content: number;
}

interface NovelSpecies {
  sequence_id: string;
  novelty_score: number;
  potential_new_genus: string;
  similarity: number;
  estimated_divergence: string;
}

interface BiodiversityMetrics {
  species_richness: number;
  shannon_diversity: number;
  simpson_diversity: number;
  total_sequences: number;
  novel_species_count: number;
  ecosystem_health_score: number;
}

interface AnalysisResults {
  taxonomic_predictions: TaxonomicPrediction[];
  novel_species: NovelSpecies[];
  biodiversity_metrics: BiodiversityMetrics;
  abundance_estimates: any[];
  processing_time: number;
  environmental_correlations: {
    depth_correlation: number;
    temperature_correlation: number;
    pressure_correlation: number;
    oxygen_correlation: number;
  };
}

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8', '#82ca9d'];

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [results, setResults] = useState<AnalysisResults | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>('');

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = event.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
      setError('');
    }
  };

  const runDemo = useCallback(async () => {
    setLoading(true);
    setError('');
    
    try {
      const response = await axios.post('http://localhost:8000/demo_analysis');
      setResults(response.data);
    } catch (err) {
      setError('Demo failed. Make sure backend is running on port 8000.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  }, []);

  const analyzeFile = useCallback(async () => {
    if (!file) {
      setError('Please select a file first');
      return;
    }

    setLoading(true);
    setError('');

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('http://localhost:8000/analyze_edna', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setResults(response.data);
    } catch (err) {
      setError('Analysis failed. Check file format and backend connection.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  }, [file]);

  const renderKingdomChart = () => {
  if (!results) return null;

  const kingdomCounts: { [key: string]: number } = {};
  results.taxonomic_predictions.forEach(pred => {
    kingdomCounts[pred.kingdom] = (kingdomCounts[pred.kingdom] || 0) + 1;
  });

  const chartData = Object.entries(kingdomCounts).map(([kingdom, count]) => ({
    name: kingdom,
    value: count
  }));

  const total = chartData.reduce((sum, item) => sum + item.value, 0);

  // Custom label component
  const renderCustomizedLabel = ({
    cx, cy, midAngle, innerRadius, outerRadius, percent, index
  }: any) => {
    const RADIAN = Math.PI / 180;
    const radius = innerRadius + (outerRadius - innerRadius) * 0.5;
    const x = cx + radius * Math.cos(-midAngle * RADIAN);
    const y = cy + radius * Math.sin(-midAngle * RADIAN);

    return (
      <text 
        x={x} 
        y={y} 
        fill="white" 
        textAnchor={x > cx ? 'start' : 'end'} 
        dominantBaseline="central"
        fontSize={12}
        fontWeight="bold"
      >
        {`${(percent * 100).toFixed(0)}%`}
      </text>
    );
  };

  return (
    <div className="chart-container">
      <h3>🦠 Taxonomic Distribution</h3>
      <ResponsiveContainer width="100%" height={300}>
        <PieChart>
          <Pie
            data={chartData}
            cx="50%"
            cy="50%"
            labelLine={false}
            label={renderCustomizedLabel}
            outerRadius={80}
            fill="#8884d8"
            dataKey="value"
          >
            {chartData.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
            ))}
          </Pie>
          <Tooltip 
            formatter={(value: number, name: string) => {
              const percent = ((value / total) * 100).toFixed(1);
              return [`${value} sequences (${percent}%)`, chartData.find(d => d.value === value)?.name];
            }}
          />
          <Legend />
        </PieChart>
      </ResponsiveContainer>
    </div>
  );
};

  const renderConfidenceChart = () => {
    if (!results) return null;

    const confidenceData = results.taxonomic_predictions.slice(0, 10).map(pred => ({
      sequence: pred.sequence_id,
      confidence: (pred.confidence * 100).toFixed(1)
    }));

    return (
      <div className="chart-container">
        <h3>🎯 AI Confidence Scores</h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={confidenceData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="sequence" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Bar dataKey="confidence" fill="#60a5fa" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    );
  };

  return (
    <div className="App">
      <header className="app-header">
        <div className="header-content">
          <div className="logo">
            <Dna className="logo-icon" />
            <div className="logo-text">
              <h1>DeepSea-AI</h1>
              <span className="version">v1.0</span>
            </div>
          </div>
          <p className="tagline">
            🌊 AI-Powered Deep Ocean Biodiversity Discovery
          </p>
          <div className="feature-badges">
            <span className="badge">Database-Independent</span>
            <span className="badge">Real-Time Processing</span>
            <span className="badge">Novel Species Detection</span>
          </div>
        </div>
      </header>

      <main className="main-content">
        <div className="upload-section">
          <div className="upload-card">
            <h2><Upload className="inline-icon" /> Upload eDNA Data</h2>
            <div className="upload-area">
              <input
                type="file"
                onChange={handleFileChange}
                accept=".fasta,.fastq,.txt"
                className="file-input"
              />
              {file && <p className="file-name">Selected: {file.name}</p>}
            </div>
            
            <div className="button-group">
              <button 
                onClick={analyzeFile} 
                disabled={loading || !file}
                className="btn btn-primary"
              >
                <Brain className="inline-icon" />
                {loading ? 'Analyzing...' : 'Analyze eDNA'}
              </button>
              
              <button 
                onClick={runDemo} 
                disabled={loading}
                className="btn btn-hero"
              >
                <Fish className="inline-icon" />
                🚀 Launch AI Demo
              </button>
            </div>
          </div>
        </div>

        {error && (
          <div className="error-message">
            <AlertCircle className="inline-icon" />
            {error}
          </div>
        )}

        {loading && (
          <div className="loading-section">
            <div className="loading-spinner"></div>
            <p>🧠 Processing eDNA sequences through AI pipeline...</p>
            <div className="loading-steps">
              <div className="step">✓ DNA sequence preprocessing</div>
              <div className="step">⚡ AI feature extraction</div>
              <div className="step">🧬 Taxonomic classification</div>
              <div className="step">🔍 Novel species detection</div>
              <div className="step">📊 Biodiversity analysis</div>
            </div>
          </div>
        )}

        {results && (
          <div className="results-section">
            <div className="results-header">
              <h2><BarChart3 className="inline-icon" /> Analysis Complete!</h2>
              <div className="processing-stats">
                <span>⚡ {results.processing_time}s processing</span>
                <span>🧬 {results.biodiversity_metrics.total_sequences} sequences</span>
                <span>🎯 {((results.taxonomic_predictions.reduce((sum, p) => sum + p.confidence, 0) / results.taxonomic_predictions.length) * 100).toFixed(0)}% avg confidence</span>
              </div>
            </div>

            <div className="metrics-grid">
              <div className="metric-card">
                <h3>🧬 Species Found</h3>
                <div className="metric-value">{results.biodiversity_metrics.species_richness}</div>
                <div className="metric-change">Identified</div>
              </div>
              <div className="metric-card">
                <h3>🌟 Novel Species</h3>
                <div className="metric-value novel">{results.biodiversity_metrics.novel_species_count}</div>
                <div className="metric-change">Discovered!</div>
              </div>
              <div className="metric-card">
                <h3>📈 Shannon Index</h3>
                <div className="metric-value">{results.biodiversity_metrics.shannon_diversity}</div>
                <div className="metric-change">Diversity</div>
              </div>
              <div className="metric-card">
                <h3>🌊 Ecosystem Health</h3>
                <div className="metric-value">{(results.biodiversity_metrics.ecosystem_health_score * 100).toFixed(0)}%</div>
                <div className="metric-change">Healthy</div>
              </div>
            </div>

            <div className="charts-grid">
              {renderKingdomChart()}
              {renderConfidenceChart()}
            </div>

            {results.novel_species.length > 0 && (
              <div className="novel-species-section">
                <h3>🔬 Groundbreaking Novel Species Discoveries</h3>
                <div className="novel-species-grid">
                  {results.novel_species.map((species, index) => (
                    <div key={index} className="novel-species-card">
                      <div className="discovery-header">
                        <span className="discovery-badge">NEW DISCOVERY</span>
                        <h4>{species.potential_new_genus}</h4>
                      </div>
                      <div className="discovery-metrics">
                        <div className="metric">
                          <span className="label">Novelty Score:</span>
                          <span className="value">{(species.novelty_score * 100).toFixed(1)}%</span>
                        </div>
                        <div className="metric">
                          <span className="label">Divergence:</span>
                          <span className="value">{species.estimated_divergence}</span>
                        </div>
                        <div className="metric">
                          <span className="label">Similarity:</span>
                          <span className="value">{(species.similarity * 100).toFixed(1)}%</span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            <div className="environmental-section">
              <h3>🌍 Environmental Correlations</h3>
              <div className="correlation-grid">
                {Object.entries(results.environmental_correlations).map(([factor, correlation]) => (
                  <div key={factor} className="correlation-item">
                    <span className="factor-name">{factor.replace('_correlation', '').replace('_', ' ')}</span>
                    <div className="correlation-bar">
                      <div 
                        className={`correlation-fill ${correlation > 0 ? 'positive' : 'negative'}`}
                        style={{ width: `${Math.abs(correlation) * 100}%` }}
                      ></div>
                    </div>
                    <span className="correlation-value">{correlation.toFixed(2)}</span>
                  </div>
                ))}
              </div>
            </div>

            <div className="taxonomic-table">
              <h3>📋 Detailed Taxonomic Classifications</h3>
              <div className="table-container">
                <table>
                  <thead>
                    <tr>
                      <th>Sequence</th>
                      <th>Kingdom</th>
                      <th>Phylum</th>
                      <th>Class</th>
                      <th>Confidence</th>
                      <th>GC Content</th>
                    </tr>
                  </thead>
                  <tbody>
                    {results.taxonomic_predictions.slice(0, 10).map((pred, index) => (
                      <tr key={index}>
                        <td>{pred.sequence_id}</td>
                        <td>{pred.kingdom}</td>
                        <td>{pred.phylum}</td>
                        <td>{pred.class}</td>
                        <td className={`confidence ${pred.confidence > 0.8 ? 'high' : pred.confidence > 0.5 ? 'medium' : 'low'}`}>
                          {(pred.confidence * 100).toFixed(1)}%
                        </td>
                        <td>{(pred.gc_content * 100).toFixed(1)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}
      </main>

      <footer className="footer">
        <p>🌊 DeepSea-AI © 2025 | Revolutionizing Marine Biodiversity Discovery</p>
        <div className="tech-stack">
          <span>PyTorch</span> • <span>React</span> • <span>FastAPI</span> • <span>BioPython</span>
        </div>
      </footer>
    </div>
  );
}

export default App;