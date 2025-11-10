from flask import Flask, jsonify, request
from flask_cors import CORS
from datetime import datetime, timedelta
import json
import os
from models.ml_models import TaskPredictor
from routes.tasks import tasks_bp
from routes.predictions import predictions_bp

app = Flask(__name__)
CORS(app)

# Register blueprints
app.register_blueprint(tasks_bp, url_prefix='/api/tasks')
app.register_blueprint(predictions_bp, url_prefix='/api/predictions')

# Initialize ML model
task_predictor = TaskPredictor()

@app.route('/')
def home():
    return jsonify({
        "message": "AI-Powered Task Manager API",
        "version": "1.0.0",
        "author": "Nadeem Ahmed",
        "endpoints": {
            "tasks": "/api/tasks",
            "predictions": "/api/predictions",
            "health": "/health"
        }
    })

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "ml_model_loaded": task_predictor.is_loaded()
    })

@app.route('/api/dashboard')
def dashboard():
    """Get dashboard statistics"""
    try:
        # Load tasks from file (in production, this would be from a database)
        with open('../data/sample_tasks.json', 'r') as f:
            tasks = json.load(f)
        
        total_tasks = len(tasks)
        completed_tasks = len([t for t in tasks if t['status'] == 'completed'])
        in_progress_tasks = len([t for t in tasks if t['status'] == 'in_progress'])
        pending_tasks = len([t for t in tasks if t['status'] == 'pending'])
        
        # Calculate completion rate
        completion_rate = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
        
        # Get recent tasks
        recent_tasks = sorted(tasks, key=lambda x: x['created_at'], reverse=True)[:5]
        
        return jsonify({
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "in_progress_tasks": in_progress_tasks,
            "pending_tasks": pending_tasks,
            "completion_rate": round(completion_rate, 2),
            "recent_tasks": recent_tasks
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
