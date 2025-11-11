"""
SEC Filings RAG Copilot - Flask Web Application
Author: Abhyudaya Lohani
University of Maryland, College Park
"""

from flask import Flask, render_template, request, jsonify
import sys
import os
from pathlib import Path
import time
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from query import answer_query

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Global metrics tracking
class Metrics:
    def __init__(self):
        self.total_queries = 0
        self.total_latency = 0.0
        self.queries_with_citations = 0
        self.query_history = []
    
    @property
    def avg_latency(self):
        if self.total_queries == 0:
            return 0.0
        return round(self.total_latency / self.total_queries, 2)
    
    @property
    def accuracy(self):
        if self.total_queries == 0:
            return 100.0
        return round((self.queries_with_citations / self.total_queries) * 100, 1)
    
    def add_query(self, latency, has_citations):
        self.total_queries += 1
        self.total_latency += latency
        if has_citations:
            self.queries_with_citations += 1
    
    def get_stats(self):
        return {
            'total_queries': self.total_queries,
            'avg_latency': self.avg_latency,
            'accuracy': self.accuracy
        }

metrics = Metrics()

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Process query and return results"""
    try:
        data = request.get_json()
        query_text = data.get('query', '').strip()
        top_k = data.get('top_k', 5)
        
        if not query_text:
            return jsonify({
                'success': False,
                'error': 'Please enter a question.'
            }), 400
        
        # Process query
        start_time = time.time()
        answer, citations = answer_query(query_text, top_k=top_k)
        elapsed_time = time.time() - start_time
        
        # Check for safety blocks
        is_blocked = answer.startswith("⚠️")
        has_citations = citations and len(citations) > 0
        
        # Update metrics (only for successful queries)
        if not is_blocked:
            metrics.add_query(elapsed_time, has_citations)
        
        # Format response
        response = {
            'success': True,
            'answer': answer,
            'citations': citations if citations else [],
            'latency': round(elapsed_time, 2),
            'num_sources': len(citations) if citations else 0,
            'is_blocked': is_blocked,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'metrics': metrics.get_stats()
        }
        
        # Add to history (keep last 10)
        if not is_blocked:
            metrics.query_history.insert(0, {
                'query': query_text,
                'timestamp': datetime.now().strftime('%H:%M:%S'),
                'latency': round(elapsed_time, 2),
                'sources': len(citations) if citations else 0
            })
            if len(metrics.query_history) > 10:
                metrics.query_history.pop()
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Get current metrics"""
    return jsonify({
        'success': True,
        'metrics': metrics.get_stats()
    })

@app.route('/history', methods=['GET'])
def get_history():
    """Return query history"""
    return jsonify({
        'success': True,
        'history': metrics.query_history[:5]
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
