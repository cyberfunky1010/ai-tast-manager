import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score
import joblib
import os
from datetime import datetime, timedelta
import json

class TaskPredictor:
    def __init__(self):
        self.duration_model = None
        self.priority_model = None
        self.difficulty_model = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Try to load existing models
        self.load_models()
    
    def prepare_features(self, task_data):
        """Prepare features for ML models"""
        features = []
        
        # Task characteristics
        task_type = task_data.get('type', 'general')
        description_length = len(task_data.get('description', ''))
        
        # Time-based features
        created_hour = datetime.now().hour
        is_weekend = datetime.now().weekday() >= 5
        
        # Complexity indicators
        has_subtasks = len(task_data.get('subtasks', [])) > 0
        num_subtasks = len(task_data.get('subtasks', []))
        
        # Dependencies
        num_dependencies = len(task_data.get('dependencies', []))
        
        features = [
            description_length,
            created_hour,
            int(is_weekend),
            int(has_subtasks),
            num_subtasks,
            num_dependencies
        ]
        
        # Encode categorical features
        if task_type in self.label_encoders.get('task_type', {}):
            features.append(self.label_encoders['task_type'][task_type])
        else:
            features.append(0)  # Default encoding
        
        return np.array(features).reshape(1, -1)
    
    def train_models(self, tasks_data):
        """Train ML models with task data"""
        if not tasks_data:
            print("No training data available")
            return False
        
        df = pd.DataFrame(tasks_data)
        
        # Prepare features
        X = []
        y_duration = []
        y_priority = []
        y_difficulty = []
        
        for _, task in df.iterrows():
            features = self.prepare_features(task.to_dict())[0]
            X.append(features)
            
            # Extract target variables
            y_duration.append(task.get('estimated_duration', 1))
            y_priority.append(task.get('priority_score', 3))
            y_difficulty.append(task.get('difficulty_score', 3))
        
        X = np.array(X)
        y_duration = np.array(y_duration)
        y_priority = np.array(y_priority)
        y_difficulty = np.array(y_difficulty)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train duration prediction model
        self.duration_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.duration_model.fit(X_scaled, y_duration)
        
        # Train priority prediction model
        self.priority_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.priority_model.fit(X_scaled, y_priority)
        
        # Train difficulty prediction model
        self.difficulty_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.difficulty_model.fit(X_scaled, y_difficulty)
        
        self.is_trained = True
        
        # Save models
        self.save_models()
        
        print("ML models trained successfully!")
        return True
    
    def predict_task_metrics(self, task_data):
        """Predict task duration, priority, and difficulty"""
        if not self.is_trained:
            return self.get_default_predictions()
        
        try:
            features = self.prepare_features(task_data)
            features_scaled = self.scaler.transform(features)
            
            duration = max(0.5, self.duration_model.predict(features_scaled)[0])
            priority = max(1, min(5, int(self.priority_model.predict(features_scaled)[0])))
            difficulty = max(1, min(5, int(self.difficulty_model.predict(features_scaled)[0])))
            
            # Calculate completion probability based on historical data
            completion_prob = self.calculate_completion_probability(task_data)
            
            return {
                "estimated_duration": round(duration, 2),
                "predicted_priority": priority,
                "predicted_difficulty": difficulty,
                "completion_probability": round(completion_prob, 2),
                "recommendation": self.generate_recommendation(duration, priority, difficulty)
            }
        except Exception as e:
            print(f"Prediction error: {e}")
            return self.get_default_predictions()
    
    def calculate_completion_probability(self, task_data):
        """Calculate probability of task completion based on various factors"""
        base_prob = 0.7  # Base 70% completion rate
        
        # Adjust based on task complexity
        description_length = len(task_data.get('description', ''))
        if description_length < 50:
            base_prob += 0.1
        elif description_length > 200:
            base_prob -= 0.1
        
        # Adjust based on dependencies
        num_dependencies = len(task_data.get('dependencies', []))
        base_prob -= (num_dependencies * 0.05)
        
        # Adjust based on time of creation
        current_hour = datetime.now().hour
        if 9 <= current_hour <= 17:  # Working hours
            base_prob += 0.05
        
        return max(0.1, min(0.95, base_prob))
    
    def generate_recommendation(self, duration, priority, difficulty):
        """Generate task recommendation based on predictions"""
        recommendations = []
        
        if duration > 4:
            recommendations.append("Consider breaking this task into smaller subtasks")
        
        if priority >= 4:
            recommendations.append("High priority - schedule this task soon")
        
        if difficulty >= 4:
            recommendations.append("Complex task - allocate extra time and resources")
        
        if priority >= 4 and difficulty >= 4:
            recommendations.append("Critical task requiring immediate attention")
        
        if not recommendations:
            recommendations.append("Standard task - can be scheduled normally")
        
        return recommendations
    
    def get_default_predictions(self):
        """Return default predictions when ML models are not available"""
        return {
            "estimated_duration": 2.0,
            "predicted_priority": 3,
            "predicted_difficulty": 3,
            "completion_probability": 0.7,
            "recommendation": ["Default estimation - update after gathering more data"]
        }
    
    def save_models(self):
        """Save trained models to disk"""
        models_dir = 'saved_models'
        os.makedirs(models_dir, exist_ok=True)
        
        if self.duration_model:
            joblib.dump(self.duration_model, f'{models_dir}/duration_model.pkl')
            joblib.dump(self.priority_model, f'{models_dir}/priority_model.pkl')
            joblib.dump(self.difficulty_model, f'{models_dir}/difficulty_model.pkl')
            joblib.dump(self.scaler, f'{models_dir}/scaler.pkl')
            joblib.dump(self.label_encoders, f'{models_dir}/label_encoders.pkl')
    
    def load_models(self):
        """Load saved models from disk"""
        models_dir = 'saved_models'
        try:
            if os.path.exists(f'{models_dir}/duration_model.pkl'):
                self.duration_model = joblib.load(f'{models_dir}/duration_model.pkl')
                self.priority_model = joblib.load(f'{models_dir}/priority_model.pkl')
                self.difficulty_model = joblib.load(f'{models_dir}/difficulty_model.pkl')
                self.scaler = joblib.load(f'{models_dir}/scaler.pkl')
                self.label_encoders = joblib.load(f'{models_dir}/label_encoders.pkl')
                self.is_trained = True
                print("ML models loaded successfully!")
        except Exception as e:
            print(f"Could not load models: {e}")
            self.is_trained = False
    
    def is_loaded(self):
        """Check if models are loaded"""
        return self.is_trained

class SmartScheduler:
    """AI-powered task scheduling system"""
    
    def __init__(self):
        self.task_predictor = TaskPredictor()
    
    def optimize_schedule(self, tasks, available_hours_per_day=8):
        """Optimize task schedule based on priority, difficulty, and duration"""
        if not tasks:
            return []
        
        # Get predictions for all tasks
        enhanced_tasks = []
        for task in tasks:
            predictions = self.task_predictor.predict_task_metrics(task)
            task_copy = task.copy()
            task_copy.update(predictions)
            enhanced_tasks.append(task_copy)
        
        # Sort by priority score and completion probability
        enhanced_tasks.sort(
            key=lambda x: (x['predicted_priority'], x['completion_probability']), 
            reverse=True
        )
        
        # Generate optimized schedule
        schedule = []
        current_day = 0
        current_day_hours = 0
        
        for task in enhanced_tasks:
            duration = task['estimated_duration']
            
            # Check if task fits in current day
            if current_day_hours + duration > available_hours_per_day:
                current_day += 1
                current_day_hours = 0
            
            start_date = datetime.now() + timedelta(days=current_day)
            
            schedule_item = {
                "task": task,
                "scheduled_date": start_date.strftime("%Y-%m-%d"),
                "scheduled_time": f"{int(current_day_hours):02d}:00",
                "estimated_end_time": f"{int(current_day_hours + duration):02d}:00"
            }
            
            schedule.append(schedule_item)
            current_day_hours += duration
        
        return schedule
