<?php
/**
 * Read training log file for real-time visualization
 * 
 * This script reads a training log file from the logs directory
 * and returns its contents for real-time visualization.
 */

// Security: Validate and sanitize input
$model = isset($_GET['model']) ? $_GET['model'] : '';
$log_type = isset($_GET['type']) ? $_GET['type'] : 'training';

// Validate model parameter
$allowed_models = array(
    'cifar10', 'cifar100', 
    'cifar10_10', 'cifar10_20', 'cifar10_30', 'cifar10_40',
    'cifar100_10', 'cifar100_20', 'cifar100_30', 'cifar100_40',
    'cifar100_transfer'
);

if (!in_array($model, $allowed_models)) {
    header('HTTP/1.1 400 Bad Request');
    echo json_encode(array('error' => 'Invalid model parameter'));
    exit;
}

// Validate log type
$allowed_types = array('training', 'test');
if (!in_array($log_type, $allowed_types)) {
    header('HTTP/1.1 400 Bad Request');
    echo json_encode(array('error' => 'Invalid log type'));
    exit;
}

// Determine log file path
$log_file = '';
$meta_model_log = false;
if ($log_type === 'training') {
    if ($model === 'cifar10') {
        $log_file = 'cifar10_training.log';
    } elseif ($model === 'cifar100') {
        $log_file = 'cifar100_training.log';
    } elseif ($model === 'cifar100_transfer') {
        $log_file = 'cifar100_transfer.log';
    } else {
        // For sample percentage models, use the base model log with sample percentage
        if (strpos($model, 'cifar10') === 0) {
            $base = 'cifar10';
            
            // Check if this is a sample percentage model
            if (preg_match('/cifar10_sample(\d+)pct/', $model, $matches)) {
                $sample_pct = $matches[1]; // Extract the sample percentage (e.g., 10, 20)
                $log_file = $base . '_training_sample' . $sample_pct . 'pct.log';
                
                // Also check for meta-model logs
                if (isset($_GET['meta_model']) && $_GET['meta_model'] === 'true') {
                    $log_file = $base . '_meta_model_sample' . $sample_pct . 'pct.log';
                    $meta_model_log = true;
                }
            } else {
                // For backward compatibility with old naming convention
                $resource_level = substr($model, 6); // Extract the resource level (e.g., 10, 20)
                $log_file = $base . '_training_' . $resource_level . 'pct.log';
                
                // Also check for meta-model logs
                if (isset($_GET['meta_model']) && $_GET['meta_model'] === 'true') {
                    $log_file = $base . '_meta_model_' . $resource_level . 'pct.log';
                    $meta_model_log = true;
                }
            }
        } else {
            $base = 'cifar100';
            
            // Check if this is a sample percentage model
            if (preg_match('/cifar100_sample(\d+)pct/', $model, $matches)) {
                $sample_pct = $matches[1]; // Extract the sample percentage (e.g., 10, 20)
                $log_file = $base . '_training_sample' . $sample_pct . 'pct.log';
                
                // Also check for meta-model logs
                if (isset($_GET['meta_model']) && $_GET['meta_model'] === 'true') {
                    $log_file = $base . '_meta_model_sample' . $sample_pct . 'pct.log';
                    $meta_model_log = true;
                }
            } else {
                // For backward compatibility with old naming convention
                $resource_level = substr($model, 7); // Extract the resource level (e.g., 10, 20)
                $log_file = $base . '_training_' . $resource_level . 'pct.log';
                
                // Also check for meta-model logs
                if (isset($_GET['meta_model']) && $_GET['meta_model'] === 'true') {
                    $log_file = $base . '_meta_model_' . $resource_level . 'pct.log';
                    $meta_model_log = true;
                }
            }
        }
    }
} else {
    // Test logs
    if (strpos($model, 'cifar10') === 0) {
        $log_file = 'cifar10_test.log';
    } else {
        $log_file = 'cifar100_test.log';
    }
}

// Define possible locations for log files
$project_root = realpath(__DIR__ . '/../../..');
$logs_dir = $project_root . '/logs';
$reports_dir = $project_root . '/reports';

// Paths to check for log files
$paths_to_check = array();

// For meta-model logs, check in logs directory first, then in reports
if ($meta_model_log) {
    $paths_to_check[] = $logs_dir . '/' . $log_file;
    $paths_to_check[] = $reports_dir . '/' . $model . '/' . $log_file;
} else {
    // For regular training logs
    // Check model-specific training log in reports directory
    $paths_to_check[] = $reports_dir . '/' . $model . '/' . $model . '_training_log.txt';
    
    // Check unified training log in reports directory
    $paths_to_check[] = $reports_dir . '/' . $model . '/' . $log_file;
    
    // Check in logs directory
    $paths_to_check[] = $logs_dir . '/' . $log_file;
    
    // For generic log files in the base dataset directory
    $base_model = strpos($model, '_') !== false ? substr($model, 0, strpos($model, '_')) : $model;
    $paths_to_check[] = $reports_dir . '/' . $base_model . '/' . $base_model . '_training_log.txt';
}

// Set content type to JSON
header('Content-Type: application/json');

// Try each path until we find a log file
$log_content = '';
$found_path = '';

foreach ($paths_to_check as $path) {
    if (file_exists($path)) {
        $log_content = file_get_contents($path);
        $found_path = $path;
        break;
    }
}

if (!empty($log_content)) {
    echo json_encode(array(
        'model' => $model,
        'type' => $log_type,
        'content' => $log_content,
        'source' => $found_path,
        'meta_model' => $meta_model_log
    ));
} else {
    echo json_encode(array(
        'model' => $model,
        'type' => $log_type,
        'content' => '',
        'error' => 'Log file not found',
        'paths_checked' => $paths_to_check,
        'meta_model' => $meta_model_log
    ));
}
?>
