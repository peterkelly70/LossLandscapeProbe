    def optimize_hyperparameters(self) -> Dict[str, Any]:
        """
        Train the meta-model using hyperparameter configurations and their evaluations.
        
        This method samples hyperparameter configurations, evaluates them on data subsets,
        extracts meta-features, and trains a meta-model to predict optimal hyperparameters.
        
        Returns:
            Dictionary containing the best hyperparameters found
        """
        try:
            from tqdm import tqdm
            use_tqdm = True
        except ImportError:
            use_tqdm = False
            
        logger.info("Starting meta-model hyperparameter optimization...")
        print("🔍 Starting meta-model hyperparameter optimization...")
        
        # Sample hyperparameter configurations to evaluate
        # Use configs_per_sample attribute if available, otherwise default to 10
        num_configs = getattr(self.config, 'configs_per_sample', 
                         getattr(self.config, 'num_configs', 10))
        
        logger.info(f"Sampling {num_configs} hyperparameter configurations...")
        print(f"🔍 Sampling {num_configs} hyperparameter configurations...")
        configs = self._sample_hyperparameter_configs(
            num_configs=num_configs
            # Note: The method uses self.config.random_seed internally
        )
        
        # Initialize storage for evaluation results
        best_val_accuracy = 0.0
        best_config = None
        meta_features = []
        meta_targets = []
        
        # Evaluate each configuration on multiple data subsets
        total_configs = len(configs)
        total_subsets = len(self.data_subsets) if hasattr(self, 'data_subsets') else 1
        total_evaluations = total_configs * total_subsets
        evaluation_counter = 0
        
        # Store these values as instance variables for progress tracking
        self.total_evaluations = total_evaluations
        self.current_evaluation = 0
        
        print(f"⏳ Total configurations to evaluate: {total_configs}")
        print(f"⏳ Total data subsets: {total_subsets}")
        print(f"⏳ Total evaluations: {total_evaluations}")
        
        # Create progress bar for configurations
        if use_tqdm:
            config_pbar = tqdm(total=total_configs, desc="Meta-Model Configs", 
                              bar_format='{desc}: {bar} {percentage:3.0f}% | {n_fmt}/{total_fmt}')
        
        for config_idx, config in enumerate(configs):
            logger.info(f"\nEvaluating configuration {config_idx+1}/{total_configs} [{(config_idx/total_configs)*100:.1f}% configurations]")
            print(f"\n🔄 Evaluating configuration {config_idx+1}/{total_configs} [{(config_idx/total_configs)*100:.1f}% configurations]")
            print(f"📊 Configuration: {convert_to_serializable(config)}")
            
            # Create progress bar for datasets
            if use_tqdm:
                dataset_pbar = tqdm(total=total_subsets, desc=f"Dataset Progress", 
                                  bar_format='{desc}: {bar} {percentage:3.0f}% | {n_fmt}/{total_fmt}')
            
            # Evaluate on multiple data subsets
            subset_results = []
            
            for subset_idx in range(total_subsets):
                # Update counter for overall progress tracking
                evaluation_counter += 1
                self.current_evaluation = evaluation_counter
                
                # Simple overall progress calculation
                overall_progress = (evaluation_counter / total_evaluations) * 100
                
                logger.info(f"Evaluating dataset {subset_idx+1}/{total_subsets} "
                          f"[Overall Progress: {evaluation_counter}/{total_evaluations} ({overall_progress:.1f}%)]")
                print(f"📈 Evaluating dataset {subset_idx+1}/{total_subsets} "
                      f"[Overall Progress: {evaluation_counter}/{total_evaluations} ({overall_progress:.1f}%)]")
                
                # Update dataset progress bar
                if use_tqdm:
                    dataset_pbar.update(1)
                    dataset_pbar.refresh()
                
                # Use reduced resource level for faster evaluation
                resource_level = getattr(self.config, 'min_resource', 0.2)
                
                # Evaluate this configuration on this subset
                try:
                    result = self._evaluate_configuration(
                        config=config,
                        subset_idx=subset_idx,
                        resource_level=resource_level
                    )
                    subset_results.append(result)
                    print(f"✅ Evaluation complete - Accuracy: {result.get('val_accuracy', 0.0):.4f}")
                except Exception as e:
                    logger.error(f"Error evaluating configuration on subset {subset_idx+1}: {str(e)}")
                    print(f"❌ Error evaluating configuration on subset {subset_idx+1}: {str(e)}")
                    # Add a placeholder result with error information
                    result = {
                        'error': str(e),
                        'val_accuracy': 0.0,
                        'config': config,
                        'subset_idx': subset_idx
                    }
                    subset_results.append(result)
            
            # Log memory usage if using CUDA
            if use_tqdm:
                dataset_pbar.update(1)
                dataset_pbar.refresh()
                
            # Use reduced resource level for faster evaluation
            resource_level = getattr(self.config, 'min_resource', 0.2)
                
            # Evaluate this configuration on this subset
            try:
                result = self._evaluate_configuration(
                    config=config,
                    subset_idx=subset_idx,
                    resource_level=resource_level
                )
                subset_results.append(result)
                print(f"✅ Evaluation complete - Accuracy: {result.get('val_accuracy', 0.0):.4f}")
            except Exception as e:
                logger.error(f"Error evaluating configuration on subset {subset_idx+1}: {str(e)}")
                print(f"❌ Error evaluating configuration on subset {subset_idx+1}: {str(e)}")
                # Add a placeholder result with error information
                result = {
                    'error': str(e),
                    'val_accuracy': 0.0,
                    'config': config,
                    'subset_idx': subset_idx
                }
                subset_results.append(result)
        
        # Log memory usage if using CUDA
        if self.device.type == 'cuda':
            logger.info(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            
        # Close dataset progress bar
        if use_tqdm:
            dataset_pbar.close()
            
        # Update configuration progress bar
        if use_tqdm:
            config_pbar.update(1)
            config_pbar.refresh()
            
        # Extract meta-features and targets from results
        for result in subset_results:
            if 'error' not in result and 'meta_features' in result:
                meta_features.append(result['meta_features'])
                meta_targets.append(result['val_accuracy'])
                    
                # Track best configuration
                if result['val_accuracy'] > best_val_accuracy:
                    best_val_accuracy = result['val_accuracy']
                    best_config = config
                    print(f"🏆 New best configuration found! Accuracy: {best_val_accuracy:.4f}")
        
    # Store meta-features and targets for later use
    self.meta_features = meta_features
    self.meta_targets = meta_targets
        self.meta_targets = meta_targets
        
        # Define number of epochs based on resource level
        max_epochs = getattr(self.config, 'max_epochs', 10)  # Default to 10 if not specified
        actual_epochs = int(max_epochs * resource_level)
        
        # Use a simple model for meta-model training
        # We don't need the actual model_name or num_classes for meta-model training
        # Just create a simple MLP for the meta-model
        input_dim = len(self.meta_features[0]) if hasattr(self, 'meta_features') and self.meta_features else 10
        output_dim = 1  # For regression tasks like hyperparameter optimization
        
        # Create a simple MLP model
        model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, output_dim)
        ).to(self.device)
        
        # Prepare optimizer configuration
        optimizer_config = {
            'optimizer': 'adam',  # Default to Adam
            'learning_rate': 0.001,
            'weight_decay': 0.0001
        }
        
        # Prepare data loaders for meta-model training
        if hasattr(self, 'meta_features') and hasattr(self, 'meta_targets') and self.meta_features and self.meta_targets:
            # Create dataset from meta-features and meta-targets
            meta_features = torch.tensor(self.meta_features, dtype=torch.float32)
            meta_targets = torch.tensor(self.meta_targets, dtype=torch.float32).view(-1, 1)
            meta_dataset = torch.utils.data.TensorDataset(meta_features, meta_targets)
            
            # Create data loaders
            dataset_size = len(meta_dataset)
            train_size = int(0.8 * dataset_size)
            val_size = dataset_size - train_size
            
            # Split into train and validation sets
            train_dataset, val_dataset = torch.utils.data.random_split(
                meta_dataset, [train_size, val_size]
            )
            
            # Create data loaders
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=8, shuffle=True, num_workers=0
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=8, shuffle=False, num_workers=0
            )
        else:
            # Create dummy data if no meta-features are available
            logger.warning("No meta-features available, creating dummy data for meta-model training")
            dummy_features = torch.randn(20, input_dim)
            dummy_targets = torch.randn(20, 1)
            dummy_dataset = torch.utils.data.TensorDataset(dummy_features, dummy_targets)
            
            # Split into train and validation sets
            train_size = 16
            val_size = 4
            train_dataset, val_dataset = torch.utils.data.random_split(
                dummy_dataset, [train_size, val_size]
            )
            
            # Create data loaders
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=4, shuffle=True, num_workers=0
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=4, shuffle=False, num_workers=0
            )
        
        # Define local training and evaluation functions
