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
    'cifa10', 'cifa100', 
    'cifa10_10', 'cifa10_20', 'cifa10_30', 'cifa10_40',
    'cifa100_10', 'cifa100_20', 'cifa100_30', 'cifa100_40',
    'cifa100_transfer'
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
if ($log_type === 'training') {
    if ($model === 'cifa10') {
        $log_file = 'cifar10_training.log';
    } elseif ($model === 'cifa100') {
        $log_file = 'cifar100_training.log';
    } elseif ($model === 'cifa100_transfer') {
        $log_file = 'cifa100_transfer_cifar100_transfer.log';
    } else {
        // For resource level models, use the base model log
        if (strpos($model, 'cifa10') === 0) {
            $log_file = 'cifar10_training.log';
        } else {
            $log_file = 'cifar100_training.log';
        }
    }
} else {
    // Test logs
    if (strpos($model, 'cifa10') === 0) {
        $log_file = 'cifar10_test.log';
    } else {
        $log_file = 'cifar100_test.log';
    }
}

// Check if model-specific log exists first
$logs_dir = '../../../reports';  // Look in the reports directory structure

// For model-specific directory structure (e.g., reports/cifa10_10/cifa10_10_training_log.txt)
$model_log_path = $logs_dir . '/' . $model . '/' . $model . '_training_log.txt';

// For generic log files in the base dataset directory (e.g., reports/cifa10/cifa10_training_log.txt)
$base_model = strpos($model, '_') !== false ? substr($model, 0, strpos($model, '_')) : $model;
$generic_log_path = $logs_dir . '/' . $base_model . '/' . $base_model . '_training_log.txt';

// Set content type to JSON
header('Content-Type: application/json');

// Try to read model-specific log first, fall back to generic log
if (file_exists($model_log_path)) {
    $log_content = file_get_contents($model_log_path);
    echo json_encode(array(
        'model' => $model,
        'type' => $log_type,
        'content' => $log_content,
        'source' => 'model_specific'
    ));
} elseif (file_exists($generic_log_path)) {
    $log_content = file_get_contents($generic_log_path);
    echo json_encode(array(
        'model' => $model,
        'type' => $log_type,
        'content' => $log_content,
        'source' => 'generic'
    ));
} else {
    echo json_encode(array(
        'model' => $model,
        'type' => $log_type,
        'content' => '',
        'error' => 'Log file not found',
        'paths_checked' => array($model_log_path, $generic_log_path)
    ));
}
?>
