import React, { useState } from 'react';
import { BrowserRouter as Router, Route, Routes, useNavigate } from 'react-router-dom';
import axios from 'axios';
import { PieChart, Pie, Cell, Tooltip, Legend } from 'recharts';
import './App.css';

// Start Page Component
function StartPage() {
  const navigate = useNavigate();
  return (
    <div className="start-container">
      <h1 className="catchy-text">Discover the Power of Sentiment Analysis</h1>
      <p>Uncover emotions hidden in text instantly!</p>
      <button className="get-started" onClick={() => navigate('/analyze')}>Get Started</button>
    </div>
  );
}

// Sentiment Analysis Page
function SentimentAnalysis() {
  const [text, setText] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async () => {
    setLoading(true);
    setError('');
    try {
      const response = await axios.post('http://localhost:5000/analyze', { text });
      setResult(response.data);
    } catch (err) {
      setError('An error occurred while fetching the data.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  // Pie Chart Data
  const data = result
    ? [
        { name: 'Positive', value: result.probabilities.positive },
        { name: 'Negative', value: result.probabilities.negative },
        { name: 'Neutral', value: result.probabilities.neutral },
      ]
    : [];

  return (
    <div className="App">
      <div className="page-container">
        <h1>Sentiment Analysis</h1>
        <textarea
          className="text-input"
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Enter text to analyze sentiment"
        />
        <button className="analyze-button" onClick={handleSubmit} disabled={loading}>
          {loading ? 'Analyzing...' : 'Analyze Sentiment'}
        </button>

        {error && <div className="error-message">{error}</div>}

        {result && (
          <div className="result">
            <h3>Sentiment: {result.sentiment}</h3>
            <PieChart width={300} height={300}>
              <Pie data={data} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={100} fill="#8884d8">
                <Cell key="positive" fill="#4CAF50" />
                <Cell key="negative" fill="#FF5733" />
                <Cell key="neutral" fill="#FFC107" />
              </Pie>
              <Tooltip />
              <Legend />
            </PieChart>

            {/* Displaying Probability Results as Text */}
            <div className="probabilities">
              <p>Positive: {result.probabilities.positive}%</p>
              <p>Negative: {result.probabilities.negative}%</p>
              <p>Neutral: {result.probabilities.neutral}%</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// App Component with Routing
function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<StartPage />} />
        <Route path="/analyze" element={<SentimentAnalysis />} />
      </Routes>
    </Router>
  );
}

export default App;
