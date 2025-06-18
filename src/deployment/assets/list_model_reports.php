<?php
// This script lists all files in a specified model directory
header('Content-Type: application/json');

// Sanitize the directory parameter to prevent directory traversal attacks
$dir = isset($_GET['dir']) ? preg_replace('/[^a-zA-Z0-9_]/', '', $_GET['dir']) : '';

if (empty($dir)) {
    echo json_encode([]);
    exit;
}

// Path to the reports directory
$reportsDir = __DIR__ . '/../../reports/' . $dir;

// Check if the directory exists
if (!is_dir($reportsDir)) {
    echo json_encode([]);
    exit;
}

// Get all files in the directory
$files = scandir($reportsDir);
$reportFiles = [];

foreach ($files as $file) {
    // Skip . and .. directories and hidden files
    if ($file[0] === '.') {
        continue;
    }
    
    // Add the file to the list
    $reportFiles[] = $file;
}

// Return the list of files as JSON
echo json_encode($reportFiles);
