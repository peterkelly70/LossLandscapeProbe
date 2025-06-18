<?php
// PHP script to generate Meta-Model training progress report on-demand

// Set paths
$project_dir = dirname(dirname(__DIR__));
$output_file = 'meta_model_progress_report.html';

// Function to find the most recent meta-model log file
function findMetaModelLog($project_dir, $dataset = null, $sample_size = null) {
    $logs_dir = $project_dir . '/logs';
    $logs = [];
    
    // If logs directory doesn't exist, check alternative locations
    if (!is_dir($logs_dir)) {
        $possible_dirs = [
            $project_dir . '/logs',
            $project_dir,
            $project_dir . '/examples'
        ];
        
        foreach ($possible_dirs as $dir) {
            if (is_dir($dir)) {
                $logs_dir = $dir;
                break;
            }
        }
    }
    
    // If specific dataset and sample size are provided, look for that log
    if ($dataset && $sample_size) {
        $specific_log = $logs_dir . '/' . $dataset . '_meta_model_' . $sample_size . 'pct.log';
        if (file_exists($specific_log)) {
            return $specific_log;
        }
    }
    
    // Otherwise, scan for all meta-model logs and find the most recent
    if (is_dir($logs_dir)) {
        $files = scandir($logs_dir);
        foreach ($files as $file) {
            if (preg_match('/(.+)_meta_model_(\d+)pct\.log$/', $file, $matches)) {
                $log_path = $logs_dir . '/' . $file;
                $logs[] = [
                    'path' => $log_path,
                    'dataset' => $matches[1],
                    'sample_size' => $matches[2],
                    'modified' => filemtime($log_path)
                ];
            }
        }
    }
    
    // Sort by modification time (newest first)
    usort($logs, function($a, $b) {
        return $b['modified'] - $a['modified'];
    });
    
    // Return the most recent log file, or null if none found
    return !empty($logs) ? $logs[0]['path'] : null;
}

// Get parameters from query string
$dataset = isset($_GET['dataset']) ? $_GET['dataset'] : null;
$sample_size = isset($_GET['sample_size']) ? $_GET['sample_size'] : null;

// Find the appropriate log file
$log_file = findMetaModelLog($project_dir, $dataset, $sample_size);

if (!$log_file) {
    // Generate a placeholder report
    echo "<div class='info-message'>
        <h3>Meta-Model Training Progress</h3>
        <p>No meta-model training logs found. This report will display meta-model training progress once training has begun.</p>
        <p>The meta-model progress report shows hyperparameter configurations, their performance, and convergence over iterations.</p>
    </div>";
    exit;
}

// Extract dataset and sample size from log filename
preg_match('/(.+)_meta_model_(\d+)pct\.log$/', basename($log_file), $matches);
$dataset = $matches[1] ?? 'unknown';
$sample_size = $matches[2] ?? 'unknown';

// Get the modification time of the log file and the output file
$log_modified = filemtime($log_file);
$output_file = $dataset . '_' . $sample_size . 'pct_' . $output_file;
$output_modified = file_exists($output_file) ? filemtime($output_file) : 0;

// Check if we need to regenerate the report
// Regenerate if output doesn't exist or log file is newer than the output file
$regenerate = !file_exists($output_file) || $log_modified > $output_modified;

if ($regenerate) {
    // Execute the Python script to generate the report
    $python_script = $project_dir . '/generate_meta_model_report.py';
    
    $command = "python3 " . escapeshellarg($python_script) . 
               " --dataset " . escapeshellarg($dataset) . 
               " --sample_size " . escapeshellarg($sample_size / 100) . 
               " --output_path " . escapeshellarg($output_file) .
               " --real_time";
    
    // Execute command
    $output = [];
    $return_var = 0;
    exec($command, $output, $return_var);
    
    if ($return_var !== 0) {
        echo "<div class='error-message'>Error generating meta-model progress report. Check logs for details.</div>";
        echo "<pre>" . implode("\n", $output) . "</pre>";
        exit;
    }
}

// If we've reached here, the report exists and is up-to-date
// Redirect to the report file
header('Location: ' . $output_file);
exit;
?>
