<?php
/**
 * Generic Training Progress Visualization Script
 * 
 * This script generates real-time visualizations of training progress for CIFAR models.
 * It reads training log files from the reports directory structure and generates
 * interactive charts to display the training progress.
 */

// Set error reporting for debugging
error_reporting(E_ALL);
ini_set('display_errors', 1);

// Get the model parameter from the query string
$model = isset($_GET['model']) ? $_GET['model'] : '';

// Validate model parameter
if (empty($model)) {
    // Default to CIFAR-10 if no model specified
    $model = 'cifa10';
}

// Sanitize model parameter to prevent directory traversal
$model = preg_replace('/[^a-zA-Z0-9_]/', '', $model);

// Define the reports directory path
$reports_dir = '../../../reports';

// Check if the model directory exists
$model_dir = $reports_dir . '/' . $model;
if (!is_dir($model_dir)) {
    echo "<div class='error'>Model directory not found: $model</div>";
    exit;
}

// Look for the training log file
$log_file = $model_dir . '/' . $model . '_training_log.txt';
if (!file_exists($log_file)) {
    echo "<div class='error'>Training log file not found: $log_file</div>";
    exit;
}

// Read the training log file
$log_content = file_get_contents($log_file);
if ($log_content === false) {
    echo "<div class='error'>Failed to read training log file</div>";
    exit;
}

// Parse the log file
$lines = explode("\n", $log_content);
$header_found = false;
$data = array(
    'epochs' => array(),
    'training_loss' => array(),
    'training_accuracy' => array(),
    'validation_loss' => array(),
    'validation_accuracy' => array(),
    'elapsed_time' => array()
);

$config = array();
$in_config = false;

foreach ($lines as $line) {
    // Skip empty lines
    if (empty(trim($line))) {
        continue;
    }
    
    // Check for configuration section
    if (strpos($line, 'Configuration:') === 0) {
        $in_config = true;
        continue;
    }
    
    // Parse configuration JSON
    if ($in_config && strpos($line, 'Epoch,') === 0) {
        $in_config = false;
        $header_found = true;
        continue;
    }
    
    // Collect configuration data
    if ($in_config) {
        $config[] = $line;
        continue;
    }
    
    // Skip header line
    if (strpos($line, 'Epoch,') === 0) {
        $header_found = true;
        continue;
    }
    
    // Parse data lines
    if ($header_found) {
        $columns = explode(',', $line);
        if (count($columns) >= 6) {
            $data['epochs'][] = intval($columns[0]);
            $data['training_loss'][] = floatval($columns[1]);
            $data['training_accuracy'][] = floatval($columns[2]);
            $data['validation_loss'][] = !empty($columns[3]) ? floatval($columns[3]) : null;
            $data['validation_accuracy'][] = !empty($columns[4]) ? floatval($columns[4]) : null;
            $data['elapsed_time'][] = floatval($columns[5]);
        }
    }
}

// Format model name for display
$model_display = str_replace('_', ' ', $model);
$model_display = ucfirst($model_display);

// If model has a resource level (e.g., cifa10_10), extract and format it
if (strpos($model, '_') !== false) {
    $parts = explode('_', $model);
    $base_model = $parts[0];
    $resource_level = $parts[1];
    
    // Format base model name
    if ($base_model == 'cifa10') {
        $base_model = 'CIFAR-10';
    } elseif ($base_model == 'cifa100') {
        $base_model = 'CIFAR-100';
    }
    
    $model_display = "$base_model ($resource_level% Resource Level)";
} else {
    // Format base model name
    if ($model == 'cifa10') {
        $model_display = 'CIFAR-10 (100% Resource Level)';
    } elseif ($model == 'cifa100') {
        $model_display = 'CIFAR-100 (100% Resource Level)';
    }
}

// Convert data to JSON for JavaScript
$json_data = json_encode($data);
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Training Progress - <?php echo htmlspecialchars($model_display); ?></title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
    <style>
        body {
            font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: white;
        }
        
        h1, h2, h3 {
            color: #2c3e50;
        }
        
        .chart-container {
            position: relative;
            height: 300px;
            margin-bottom: 30px;
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .metric-card {
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #3498db;
        }
        
        .metric-label {
            color: #7f8c8d;
            font-size: 14px;
        }
        
        .error {
            background-color: #f8d7da;
            color: #721c24;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        
        .refresh-button {
            background-color: #3498db;
            color: white;
            border: none;
            padding: 8px 15px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            margin-top: 10px;
        }
        
        .refresh-button:hover {
            background-color: #2980b9;
        }
        
        .last-updated {
            font-size: 12px;
            color: #7f8c8d;
            margin-top: 5px;
        }
        
        .progress-info {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
        
        .auto-refresh {
            display: flex;
            align-items: center;
            gap: 10px;
        }
    </style>
</head>
<body>
    <h1>Training Progress: <?php echo htmlspecialchars($model_display); ?></h1>
    
    <div class="progress-info">
        <div>
            <button id="refreshButton" class="refresh-button">Refresh Data</button>
            <div class="last-updated">Last updated: <span id="lastUpdated"><?php echo date('Y-m-d H:i:s'); ?></span></div>
        </div>
        <div class="auto-refresh">
            <label for="autoRefresh">Auto refresh:</label>
            <select id="autoRefresh">
                <option value="0">Off</option>
                <option value="5" selected>5 seconds</option>
                <option value="10">10 seconds</option>
                <option value="30">30 seconds</option>
                <option value="60">1 minute</option>
            </select>
        </div>
    </div>
    
    <!-- Summary Metrics -->
    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-value" id="currentEpoch">0</div>
            <div class="metric-label">Current Epoch</div>
        </div>
        <div class="metric-card">
            <div class="metric-value" id="latestLoss">0.000</div>
            <div class="metric-label">Latest Training Loss</div>
        </div>
        <div class="metric-card">
            <div class="metric-value" id="latestAccuracy">0.00%</div>
            <div class="metric-label">Latest Training Accuracy</div>
        </div>
        <div class="metric-card">
            <div class="metric-value" id="totalTime">0s</div>
            <div class="metric-label">Total Training Time</div>
        </div>
    </div>
    
    <!-- Training Loss Chart -->
    <h2>Training Loss</h2>
    <div class="chart-container">
        <canvas id="lossChart"></canvas>
    </div>
    
    <!-- Training Accuracy Chart -->
    <h2>Training Accuracy</h2>
    <div class="chart-container">
        <canvas id="accuracyChart"></canvas>
    </div>
    
    <script>
        // Parse the training data from PHP
        const trainingData = <?php echo $json_data; ?>;
        let refreshInterval = null;
        
        // Update summary metrics
        function updateMetrics() {
            if (trainingData.epochs.length > 0) {
                const lastIndex = trainingData.epochs.length - 1;
                
                document.getElementById('currentEpoch').textContent = trainingData.epochs[lastIndex];
                document.getElementById('latestLoss').textContent = trainingData.training_loss[lastIndex].toFixed(4);
                document.getElementById('latestAccuracy').textContent = (trainingData.training_accuracy[lastIndex] * 100).toFixed(2) + '%';
                
                // Calculate total training time
                const totalSeconds = trainingData.elapsed_time.reduce((a, b) => a + b, 0);
                const hours = Math.floor(totalSeconds / 3600);
                const minutes = Math.floor((totalSeconds % 3600) / 60);
                const seconds = Math.floor(totalSeconds % 60);
                
                let timeString = '';
                if (hours > 0) timeString += `${hours}h `;
                if (minutes > 0 || hours > 0) timeString += `${minutes}m `;
                timeString += `${seconds}s`;
                
                document.getElementById('totalTime').textContent = timeString;
            }
        }
        
        // Create loss chart
        const lossCtx = document.getElementById('lossChart').getContext('2d');
        const lossChart = new Chart(lossCtx, {
            type: 'line',
            data: {
                labels: trainingData.epochs,
                datasets: [{
                    label: 'Training Loss',
                    data: trainingData.training_loss,
                    borderColor: 'rgb(255, 99, 132)',
                    backgroundColor: 'rgba(255, 99, 132, 0.1)',
                    tension: 0.1,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Training Loss Over Epochs'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: false
                    }
                }
            }
        });
        
        // Create accuracy chart
        const accCtx = document.getElementById('accuracyChart').getContext('2d');
        const accChart = new Chart(accCtx, {
            type: 'line',
            data: {
                labels: trainingData.epochs,
                datasets: [{
                    label: 'Training Accuracy',
                    data: trainingData.training_accuracy,
                    borderColor: 'rgb(54, 162, 235)',
                    backgroundColor: 'rgba(54, 162, 235, 0.1)',
                    tension: 0.1,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Training Accuracy Over Epochs'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1.0
                    }
                }
            }
        });
        
        // Update metrics on page load
        updateMetrics();
        
        // Function to refresh data
        function refreshData() {
            fetch(`read_training_log.php?model=<?php echo urlencode($model); ?>&log_type=training`)
                .then(response => response.json())
                .then(data => {
                    if (data.success && data.content) {
                        // Parse the new log content
                        const lines = data.content.split('\n');
                        const newData = {
                            epochs: [],
                            training_loss: [],
                            training_accuracy: [],
                            validation_loss: [],
                            validation_accuracy: [],
                            elapsed_time: []
                        };
                        
                        let headerFound = false;
                        
                        for (const line of lines) {
                            // Skip empty lines
                            if (line.trim() === '') continue;
                            
                            // Skip header line
                            if (line.startsWith('Epoch,')) {
                                headerFound = true;
                                continue;
                            }
                            
                            // Skip configuration and metadata
                            if (!headerFound) continue;
                            
                            // Parse data lines
                            const columns = line.split(',');
                            if (columns.length >= 6) {
                                newData.epochs.push(parseInt(columns[0]));
                                newData.training_loss.push(parseFloat(columns[1]));
                                newData.training_accuracy.push(parseFloat(columns[2]));
                                newData.validation_loss.push(columns[3] ? parseFloat(columns[3]) : null);
                                newData.validation_accuracy.push(columns[4] ? parseFloat(columns[4]) : null);
                                newData.elapsed_time.push(parseFloat(columns[5]));
                            }
                        }
                        
                        // Update the training data
                        trainingData.epochs = newData.epochs;
                        trainingData.training_loss = newData.training_loss;
                        trainingData.training_accuracy = newData.training_accuracy;
                        trainingData.validation_loss = newData.validation_loss;
                        trainingData.validation_accuracy = newData.validation_accuracy;
                        trainingData.elapsed_time = newData.elapsed_time;
                        
                        // Update the charts
                        lossChart.data.labels = trainingData.epochs;
                        lossChart.data.datasets[0].data = trainingData.training_loss;
                        lossChart.update();
                        
                        accChart.data.labels = trainingData.epochs;
                        accChart.data.datasets[0].data = trainingData.training_accuracy;
                        accChart.update();
                        
                        // Update metrics
                        updateMetrics();
                        
                        // Update last updated time
                        document.getElementById('lastUpdated').textContent = new Date().toLocaleString();
                    }
                })
                .catch(error => {
                    console.error('Error refreshing data:', error);
                });
        }
        
        // Set up refresh button
        document.getElementById('refreshButton').addEventListener('click', refreshData);
        
        // Set up auto-refresh
        document.getElementById('autoRefresh').addEventListener('change', function() {
            const interval = parseInt(this.value);
            
            // Clear existing interval
            if (refreshInterval) {
                clearInterval(refreshInterval);
                refreshInterval = null;
            }
            
            // Set new interval if not 0
            if (interval > 0) {
                refreshInterval = setInterval(refreshData, interval * 1000);
            }
        });
        
        // Start auto-refresh by default (5 seconds)
        refreshInterval = setInterval(refreshData, 5000);
    </script>
</body>
</html>
