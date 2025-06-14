"""
Meta-Model for Hyperparameter Prediction
=======================================

This module implements a meta-learning approach to predict optimal hyperparameters
based on dataset characteristics and partial training results.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import joblib
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class DatasetFeatureExtractor:
    """
    Extracts statistical features from a dataset to use as input for the meta-model.
    """
    
    def __init__(self):
        """Initialize the feature extractor."""
        self.feature_names = [
            # Basic statistics
            'num_samples', 'num_features', 'num_classes',
            # Class distribution statistics
            'class_imbalance', 'class_entropy',
            # Feature statistics
            'feature_mean', 'feature_std', 'feature_skew', 'feature_kurtosis',
            # Complexity measures
            'feature_correlation_mean', 'feature_correlation_std'
        ]
    
    def extract_features(self, train_loader, val_loader=None) -> Dict[str, float]:
        """
        Extract statistical features from the dataset.
        
        Args:
            train_loader: DataLoader for the training set
            val_loader: Optional DataLoader for the validation set
            
        Returns:
            Dictionary of dataset features
        """
        logger.info("Extracting dataset features...")
        
        # Initialize feature dictionary
        features = {}
        
        # Collect all data for analysis
        all_data = []
        all_labels = []
        
        for inputs, targets in train_loader:
            all_data.append(inputs.cpu().numpy().reshape(inputs.shape[0], -1))
            all_labels.append(targets.cpu().numpy())
        
        all_data = np.vstack(all_data)
        all_labels = np.concatenate(all_labels)
        
        # Basic statistics
        features['num_samples'] = len(all_labels)
        features['num_features'] = all_data.shape[1]
        features['num_classes'] = len(np.unique(all_labels))
        
        # Class distribution statistics
        class_counts = np.bincount(all_labels.astype(int))
        class_probs = class_counts / len(all_labels)
        features['class_imbalance'] = np.max(class_probs) / np.min(class_probs) if np.min(class_probs) > 0 else 1.0
        features['class_entropy'] = -np.sum(class_probs * np.log(class_probs + 1e-10))
        
        # Feature statistics
        features['feature_mean'] = np.mean(all_data)
        features['feature_std'] = np.std(all_data)
        features['feature_skew'] = np.mean(((all_data - np.mean(all_data, axis=0)) / (np.std(all_data, axis=0) + 1e-10)) ** 3)
        features['feature_kurtosis'] = np.mean(((all_data - np.mean(all_data, axis=0)) / (np.std(all_data, axis=0) + 1e-10)) ** 4)
        
        # Complexity measures - sample correlation matrix for a subset of features
        if all_data.shape[1] > 1:
            # Sample up to 1000 features for correlation analysis to keep computation manageable
            sample_size = min(1000, all_data.shape[1])
            sampled_features = np.random.choice(all_data.shape[1], sample_size, replace=False)
            sampled_data = all_data[:, sampled_features]
            
            # Calculate correlation matrix
            corr_matrix = np.corrcoef(sampled_data.T)
            # Get upper triangle without diagonal
            upper_triangle = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
            
            features['feature_correlation_mean'] = np.mean(np.abs(upper_triangle))
            features['feature_correlation_std'] = np.std(np.abs(upper_triangle))
        else:
            features['feature_correlation_mean'] = 0.0
            features['feature_correlation_std'] = 0.0
        
        logger.info(f"Extracted {len(features)} dataset features")
        return features


class TrainingResultFeatureExtractor:
    """
    Extracts features from partial training results to enhance the meta-model.
    """
    
    def __init__(self):
        """Initialize the training result feature extractor."""
        self.feature_names = [
            # Training dynamics
            'initial_loss', 'final_loss', 'loss_decrease_rate',
            'initial_accuracy', 'final_accuracy', 'accuracy_increase_rate',
            # Sharpness measures
            'sharpness', 'perturbation_robustness',
            # Resource usage
            'resource_level', 'epochs'
        ]
    
    def extract_features(self, training_history: List[Dict[str, Any]], sharpness: float) -> Dict[str, float]:
        """
        Extract features from training history and sharpness measurement.
        
        Args:
            training_history: List of dictionaries containing training metrics per epoch
            sharpness: Measured sharpness of the loss landscape
            
        Returns:
            Dictionary of training result features
        """
        features = {}
        
        if not training_history:
            logger.warning("Empty training history provided")
            return {name: 0.0 for name in self.feature_names}
        
        # Extract loss dynamics
        losses = [entry.get('loss', 0.0) for entry in training_history if 'loss' in entry]
        if losses:
            features['initial_loss'] = losses[0]
            features['final_loss'] = losses[-1]
            features['loss_decrease_rate'] = (losses[0] - losses[-1]) / len(losses) if len(losses) > 1 else 0.0
        else:
            features['initial_loss'] = 0.0
            features['final_loss'] = 0.0
            features['loss_decrease_rate'] = 0.0
        
        # Extract accuracy dynamics
        accuracies = [entry.get('accuracy', 0.0) for entry in training_history if 'accuracy' in entry]
        if accuracies:
            features['initial_accuracy'] = accuracies[0]
            features['final_accuracy'] = accuracies[-1]
            features['accuracy_increase_rate'] = (accuracies[-1] - accuracies[0]) / len(accuracies) if len(accuracies) > 1 else 0.0
        else:
            features['initial_accuracy'] = 0.0
            features['final_accuracy'] = 0.0
            features['accuracy_increase_rate'] = 0.0
        
        # Sharpness measures
        features['sharpness'] = sharpness
        features['perturbation_robustness'] = 1.0 / (sharpness + 1e-10)
        
        # Resource usage
        features['resource_level'] = training_history[0].get('resource_level', 0.0) if training_history else 0.0
        features['epochs'] = len(training_history)
        
        return features


class HyperparameterPredictor:
    """
    Meta-model that predicts optimal hyperparameters based on dataset features
    and partial training results.
    """
    
    def __init__(self, model_dir: Optional[str] = None):
        """
        Initialize the hyperparameter predictor.
        
        Args:
            model_dir: Directory to save/load model files
        """
        self.dataset_feature_extractor = DatasetFeatureExtractor()
        self.training_feature_extractor = TrainingResultFeatureExtractor()
        self.model_dir = model_dir or os.path.join(os.path.dirname(__file__), '../models/meta')
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Initialize models for each hyperparameter
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        
        # Track hyperparameter types and ranges
        self.hyperparameter_types = {}
        self.hyperparameter_ranges = {}
        self.categorical_values = {}
        
        logger.info(f"Initialized HyperparameterPredictor with model directory: {self.model_dir}")
    
    def _get_feature_vector(self, dataset_features: Dict[str, float], 
                           training_features: Dict[str, float]) -> np.ndarray:
        """
        Combine dataset and training features into a single feature vector.
        
        Args:
            dataset_features: Features extracted from the dataset
            training_features: Features extracted from training results
            
        Returns:
            Combined feature vector
        """
        # Combine all features
        all_features = {}
        all_features.update(dataset_features)
        all_features.update(training_features)
        
        # Convert to vector in a consistent order
        feature_names = list(self.dataset_feature_extractor.feature_names) + \
                        list(self.training_feature_extractor.feature_names)
        
        feature_vector = np.array([all_features.get(name, 0.0) for name in feature_names])
        return feature_vector
    
    def add_training_example(self, 
                            dataset_features: Dict[str, float],
                            training_features: Dict[str, float],
                            hyperparameters: Dict[str, Any],
                            performance: float):
        """
        Add a training example to the meta-model.
        
        Args:
            dataset_features: Features extracted from the dataset
            training_features: Features extracted from training results
            hyperparameters: Hyperparameter configuration used
            performance: Performance metric achieved (higher is better)
        """
        # Store the example for later training
        example = {
            'dataset_features': dataset_features,
            'training_features': training_features,
            'hyperparameters': hyperparameters,
            'performance': performance
        }
        
        # Save the example to disk
        example_path = os.path.join(self.model_dir, f'example_{len(os.listdir(self.model_dir))}.joblib')
        joblib.dump(example, example_path)
        
        logger.info(f"Added training example with performance {performance:.4f}")
        
        # Update hyperparameter types and ranges
        for name, value in hyperparameters.items():
            if name not in self.hyperparameter_types:
                self.hyperparameter_types[name] = type(value)
                self.hyperparameter_ranges[name] = [value, value]
                
                # Track categorical values
                if isinstance(value, str):
                    self.categorical_values[name] = {value}
            else:
                if isinstance(value, (int, float)):
                    self.hyperparameter_ranges[name][0] = min(self.hyperparameter_ranges[name][0], value)
                    self.hyperparameter_ranges[name][1] = max(self.hyperparameter_ranges[name][1], value)
                elif isinstance(value, str):
                    if name in self.categorical_values:
                        self.categorical_values[name].add(value)
                    else:
                        self.categorical_values[name] = {value}
    
    def train(self):
        """
        Train the meta-model on all collected examples.
        """
        logger.info("Training hyperparameter prediction meta-model...")
        
        # Load all examples
        examples = []
        for filename in os.listdir(self.model_dir):
            if filename.startswith('example_') and filename.endswith('.joblib'):
                example_path = os.path.join(self.model_dir, filename)
                example = joblib.load(example_path)
                examples.append(example)
        
        if not examples:
            logger.warning("No training examples found, cannot train meta-model")
            return
        
        logger.info(f"Training meta-model with {len(examples)} examples")
        
        # Extract all hyperparameter names
        all_hyperparams = set()
        for example in examples:
            all_hyperparams.update(example['hyperparameters'].keys())
        
        # Train a separate model for each hyperparameter
        for hyperparam in all_hyperparams:
            logger.info(f"Training model for hyperparameter: {hyperparam}")
            
            # Prepare training data
            X = []
            y = []
            
            for example in examples:
                if hyperparam in example['hyperparameters']:
                    feature_vector = self._get_feature_vector(
                        example['dataset_features'],
                        example['training_features']
                    )
                    X.append(feature_vector)
                    y.append(example['hyperparameters'][hyperparam])
            
            if not X:
                logger.warning(f"No examples for hyperparameter {hyperparam}, skipping")
                continue
            
            X = np.array(X)
            y = np.array(y)
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Determine if this is a categorical or numerical hyperparameter
            is_categorical = False
            if hyperparam in self.categorical_values:
                is_categorical = True
                
            if is_categorical:
                # For categorical hyperparameters, use classification
                label_encoder = LabelEncoder()
                y_encoded = label_encoder.fit_transform(y)
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_scaled, y_encoded)
                self.label_encoders[hyperparam] = label_encoder
            else:
                # For numerical hyperparameters, use regression
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_scaled, y)
            
            # Save model and scaler
            self.models[hyperparam] = model
            self.scalers[hyperparam] = scaler
            
            # Save to disk
            model_path = os.path.join(self.model_dir, f'model_{hyperparam}.joblib')
            scaler_path = os.path.join(self.model_dir, f'scaler_{hyperparam}.joblib')
            joblib.dump(model, model_path)
            joblib.dump(scaler, scaler_path)
        
        # Save hyperparameter types, ranges, and categorical values
        types_path = os.path.join(self.model_dir, 'hyperparameter_types.joblib')
        ranges_path = os.path.join(self.model_dir, 'hyperparameter_ranges.joblib')
        categorical_path = os.path.join(self.model_dir, 'categorical_values.joblib')
        joblib.dump(self.hyperparameter_types, types_path)
        joblib.dump(self.hyperparameter_ranges, ranges_path)
        joblib.dump(self.categorical_values, categorical_path)
        
        # Save label encoders
        for hyperparam, encoder in self.label_encoders.items():
            encoder_path = os.path.join(self.model_dir, f'encoder_{hyperparam}.joblib')
            joblib.dump(encoder, encoder_path)
        
        logger.info(f"Meta-model training complete, trained models for {len(self.models)} hyperparameters")
    
    def load(self):
        """
        Load trained models from disk.
        """
        logger.info("Loading hyperparameter prediction meta-model...")
        
        # Load hyperparameter types, ranges, and categorical values
        types_path = os.path.join(self.model_dir, 'hyperparameter_types.joblib')
        ranges_path = os.path.join(self.model_dir, 'hyperparameter_ranges.joblib')
        categorical_path = os.path.join(self.model_dir, 'categorical_values.joblib')
        
        if os.path.exists(types_path) and os.path.exists(ranges_path):
            self.hyperparameter_types = joblib.load(types_path)
            self.hyperparameter_ranges = joblib.load(ranges_path)
            
            if os.path.exists(categorical_path):
                self.categorical_values = joblib.load(categorical_path)
        else:
            logger.warning("Hyperparameter metadata not found")
            
        # Load label encoders
        for hyperparam in self.categorical_values.keys() if hasattr(self, 'categorical_values') else []:
            encoder_path = os.path.join(self.model_dir, f'encoder_{hyperparam}.joblib')
            if os.path.exists(encoder_path):
                self.label_encoders[hyperparam] = joblib.load(encoder_path)
        
        # Find all model files
        for filename in os.listdir(self.model_dir):
            if filename.startswith('model_') and filename.endswith('.joblib'):
                hyperparam = filename[6:-7]  # Extract name from 'model_NAME.joblib'
                
                model_path = os.path.join(self.model_dir, filename)
                scaler_path = os.path.join(self.model_dir, f'scaler_{hyperparam}.joblib')
                
                if os.path.exists(model_path) and os.path.exists(scaler_path):
                    self.models[hyperparam] = joblib.load(model_path)
                    self.scalers[hyperparam] = joblib.load(scaler_path)
                    logger.info(f"Loaded model for hyperparameter: {hyperparam}")
        
        logger.info(f"Loaded {len(self.models)} hyperparameter prediction models")
    
    def predict(self, 
               dataset_features: Dict[str, float],
               training_features: Dict[str, float]) -> Dict[str, Any]:
        """
        Predict optimal hyperparameters based on dataset and training features.
        
        Args:
            dataset_features: Features extracted from the dataset
            training_features: Features extracted from training results
            
        Returns:
            Dictionary of predicted optimal hyperparameters
        """
        if not self.models:
            logger.warning("No trained models available, loading from disk...")
            self.load()
            
            if not self.models:
                logger.error("No trained models available, cannot make predictions")
                return {}
        
        logger.info("Predicting optimal hyperparameters...")
        
        # Get feature vector
        feature_vector = self._get_feature_vector(dataset_features, training_features)
        
        # Make predictions for each hyperparameter
        predictions = {}
        
        for hyperparam, model in self.models.items():
            scaler = self.scalers.get(hyperparam)
            if scaler is None:
                logger.warning(f"No scaler found for {hyperparam}, skipping")
                continue
            
            # Scale features
            X_scaled = scaler.transform(feature_vector.reshape(1, -1))
            
            # Check if this is a categorical hyperparameter
            is_categorical = hyperparam in self.categorical_values if hasattr(self, 'categorical_values') else False
            
            if is_categorical:
                # For categorical hyperparameters, predict class
                label_encoder = self.label_encoders.get(hyperparam)
                if label_encoder is None:
                    logger.warning(f"No label encoder found for {hyperparam}, skipping")
                    continue
                    
                pred_class = model.predict(X_scaled)[0]
                pred = label_encoder.inverse_transform([pred_class])[0]
            else:
                # For numerical hyperparameters, predict value
                pred = model.predict(X_scaled)[0]
                
                # Convert to appropriate type
                if hyperparam in self.hyperparameter_types:
                    if self.hyperparameter_types[hyperparam] == int:
                        pred = int(round(pred))
                    elif self.hyperparameter_types[hyperparam] == bool:
                        pred = bool(round(pred))
            
            predictions[hyperparam] = pred
        
        logger.info(f"Predicted values for {len(predictions)} hyperparameters")
        return predictions


class MetaModelTrainer:
    """
    Manages the training of the meta-model using results from hyperparameter evaluations.
    """
    
    def __init__(self, predictor: Optional[HyperparameterPredictor] = None):
        """
        Initialize the meta-model trainer.
        
        Args:
            predictor: Optional HyperparameterPredictor instance
        """
        self.predictor = predictor or HyperparameterPredictor()
        self.dataset_feature_extractor = DatasetFeatureExtractor()
        self.training_feature_extractor = TrainingResultFeatureExtractor()
    
    def process_evaluation_result(self, 
                                 train_loader,
                                 val_loader,
                                 config: Dict[str, Any],
                                 training_history: List[Dict[str, Any]],
                                 sharpness: float,
                                 performance: float):
        """
        Process an evaluation result and add it to the meta-model training data.
        
        Args:
            train_loader: DataLoader for the training set
            val_loader: DataLoader for the validation set
            config: Hyperparameter configuration
            training_history: List of training metrics per epoch
            sharpness: Measured sharpness of the loss landscape
            performance: Performance metric (higher is better)
        """
        # Extract dataset features
        dataset_features = self.dataset_feature_extractor.extract_features(train_loader, val_loader)
        
        # Extract training features
        training_features = self.training_feature_extractor.extract_features(training_history, sharpness)
        
        # Add example to meta-model
        self.predictor.add_training_example(
            dataset_features=dataset_features,
            training_features=training_features,
            hyperparameters=config,
            performance=performance
        )
    
    def train_meta_model(self):
        """
        Train the meta-model on all collected examples.
        """
        self.predictor.train()
    
    def predict_hyperparameters(self, train_loader, val_loader) -> Dict[str, Any]:
        """
        Predict optimal hyperparameters for a new dataset.
        
        Args:
            train_loader: DataLoader for the training set
            val_loader: DataLoader for the validation set
            
        Returns:
            Dictionary of predicted optimal hyperparameters
        """
        # Extract dataset features
        dataset_features = self.dataset_feature_extractor.extract_features(train_loader, val_loader)
        
        # Use empty training features for initial prediction
        empty_training_features = {name: 0.0 for name in self.training_feature_extractor.feature_names}
        
        # Make prediction
        return self.predictor.predict(dataset_features, empty_training_features)
