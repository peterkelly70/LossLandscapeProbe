<?php
header('Content-Type: application/json');

// Directory containing training report files
$dir = __DIR__ . '/training_reports';

// Create directory if it doesn't exist
if (!is_dir($dir)) {
    echo json_encode([]);
    exit;
}

// Get all HTML files
$files = glob($dir . '/*.html');
$reports = [];

foreach ($files as $file) {
    $filename = basename($file);
    $reports[] = 'training_reports/' . $filename;
}

// Sort by filename (which should put newest first if using timestamp naming)
rsort($reports);

echo json_encode($reports);
?>
