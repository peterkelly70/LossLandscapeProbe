<?php
// PHP script to generate CIFAR-100 progress report on-demand

// Set paths
$project_dir = dirname(__DIR__);
$log_file = $project_dir . '/cifar100_training.log';
$output_file = 'cifar100_progress_report.html';
$python_script = $project_dir . '/examples/visualize_cifar100_progress.py';

// Check for alternative log file locations
$possible_log_files = [
    $project_dir . '/cifar100_transfer.log',
    $project_dir . '/cifar100_training.log',
    $project_dir . '/examples/cifar100_transfer.log',
    $project_dir . '/examples/cifar100_training.log',
    $project_dir . '/logs/cifar100_training.log'
];

$log_file_found = false;
$log_file = '';
foreach ($possible_log_files as $potential_log) {
    if (file_exists($potential_log)) {
        $log_file = $potential_log;
        $log_file_found = true;
        break;
    }
}

// If no log file exists, check if we have a results file we can use instead
if (!$log_file_found) {
    // Check for various possible model file names
    $possible_model_files = [
        $project_dir . '/cifar100_transfer_results.pth',
        $project_dir . '/cifar100_multisamplesize_meta_model.pth',
        $project_dir . '/cifar100_multisamplesize_trained.pth',
        $project_dir . '/cifar100_best_model.pth',
        $project_dir . '/examples/cifar100_transfer_results.pth',
        $project_dir . '/examples/cifar100_multisamplesize_meta_model.pth',
        $project_dir . '/examples/cifar100_multisamplesize_trained.pth'
    ];
    
    $model_file_found = false;
    foreach ($possible_model_files as $potential_model) {
        if (file_exists($potential_model)) {
            // Use the results file directly with the visualization script
            $log_file = $potential_model;
            $log_file_found = true;
            $model_file_found = true;
            break;
        }
    }
    
    if (!$model_file_found) {
        // Generate a placeholder report
        echo "<div class='info-message'>
            <h3>CIFAR-100 Training Progress</h3>
            <p>No CIFAR-100 training log or model files found. This report will display CIFAR-100 training progress once training has begun.</p>
            <p>The training progress report shows loss and accuracy metrics over training epochs.</p>
        </div>";
        exit;
    }
}

// Get the modification time of the log file and the output file
$log_modified = filemtime($log_file);
$output_modified = file_exists($output_file) ? filemtime($output_file) : 0;

// Check if we need to regenerate the report
// Regenerate if output doesn't exist or log file is newer than the output file
$regenerate = !file_exists($output_file) || $log_modified > $output_modified;

if ($regenerate) {
    // Execute the Python script to generate the report
    $command = "python3 " . escapeshellarg($python_script) . 
               " --log " . escapeshellarg($log_file) . 
               " --output " . escapeshellarg($output_file);
    
    // Execute command
    $output = [];
    $return_var = 0;
    exec($command, $output, $return_var);
    
    if ($return_var !== 0) {
        echo "<div class='error-message'>Error generating CIFAR-100 progress report. Check logs for details.</div>";
        echo "<pre>" . implode("\n", $output) . "</pre>";
        exit;
    }
}

// If we've reached here, the report exists and is up-to-date
// Redirect to the report file
header('Location: ' . $output_file);
exit;
?>
