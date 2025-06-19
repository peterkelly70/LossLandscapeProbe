<?php
// Meta-model report generator with real-time refresh functionality

// Set error reporting for debugging
error_reporting(E_ALL);
ini_set('display_errors', 1);

// Define paths
$project_dir = realpath(__DIR__ . '/..');

// Check multiple possible locations for the log file with correct directory naming
$possible_log_paths = [
    '/var/www/html/loss.computer-wizard.com.au/reports/cifar10_10/cifar10_meta_model_10pct.log',
    $project_dir . '/reports/cifar10_10/cifar10_meta_model_10pct.log',
    '/var/www/html/loss.computer-wizard.com.au/reports/cifar10/cifar10_meta_model_10pct.log',
    $project_dir . '/reports/cifar10/cifar10_meta_model_10pct.log',
    $project_dir . '/logs/cifar10_meta_model_10pct.log',
    $project_dir . '/cifar10_meta_model_10pct.log'
];

// Also check the old incorrect directory name for backward compatibility
$possible_log_paths[] = '/var/www/html/loss.computer-wizard.com.au/reports/cifar10/cifar10_meta_model_10pct.log';
$possible_log_paths[] = $project_dir . '/reports/cifar10/cifar10_meta_model_10pct.log';

// Find the most recently updated log file
$log_file = null;
$latest_mtime = 0;
foreach ($possible_log_paths as $path) {
    if (file_exists($path)) {
        $mtime = filemtime($path);
        if ($mtime > $latest_mtime) {
            $latest_mtime = $mtime;
            $log_file = $path;
        }
    }
}

// If no log file found, default to the first path
if ($log_file === null) {
    $log_file = $possible_log_paths[0];
}

// Determine the output paths based on the log file path
if (strpos($log_file, 'cifar10_10') !== false) {
    $output_path = $project_dir . '/reports/cifar10_10/meta_model_report.html';
    $web_output_path = '/var/www/html/loss.computer-wizard.com.au/reports/cifar10_10/meta_model_report.html';
} else {
    $output_path = $project_dir . '/reports/cifar10/meta_model_report.html';
    $web_output_path = '/var/www/html/loss.computer-wizard.com.au/reports/cifar10/meta_model_report.html';
}

// Check if refresh was requested
$refresh = isset($_GET['refresh']) && $_GET['refresh'] === 'true';

// Function to parse the meta-model log file
function parse_meta_model_log($log_file) {
    if (!file_exists($log_file)) {
        return null;
    }
    
    $log_content = file_get_contents($log_file);
    $info = [
        'start_time' => null,
        'current_iteration' => 0,
        'total_iterations' => 3,  // Default from log
        'configurations_evaluated' => 0,
        'total_configurations' => 10,  // Default from log
        'current_resource_level' => 0.1,
        'configurations' => []
    ];
    
    // Extract start time
    preg_match('/([\d-]+ [\d:]+,[\d]+)/', $log_content, $start_time_match);
    if (!empty($start_time_match)) {
        $info['start_time'] = $start_time_match[1];
    }
    
    // Extract iteration information
    preg_match('/Meta-optimization iteration (\d+)\/(\d+)/', $log_content, $iteration_match);
    if (!empty($iteration_match)) {
        $info['current_iteration'] = (int)$iteration_match[1];
        $info['total_iterations'] = (int)$iteration_match[2];
    }
    
    // Extract resource level
    preg_match('/at resource level ([\d.]+)/', $log_content, $resource_match);
    if (!empty($resource_match)) {
        $info['current_resource_level'] = (float)$resource_match[1];
    }
    
    // Extract configurations evaluated
    preg_match_all('/Evaluating configuration (\d+)\/(\d+)/', $log_content, $config_matches, PREG_SET_ORDER);
    if (!empty($config_matches)) {
        $last_match = end($config_matches);
        $info['configurations_evaluated'] = (int)$last_match[1];
        $info['total_configurations'] = (int)$last_match[2];
    }
    
    // Extract sharpness and robustness measurements
    preg_match_all('/Sharpness: ([\d.]+), Perturbation Robustness: ([\d.]+)/', $log_content, $sharpness_matches, PREG_SET_ORDER);
    foreach ($sharpness_matches as $i => $match) {
        if (!isset($info['configurations'][$i])) {
            $info['configurations'][$i] = [];
        }
        $info['configurations'][$i]['sharpness'] = (float)$match[1];
        $info['configurations'][$i]['robustness'] = (float)$match[2];
    }
    
    // Extract performance metrics
    preg_match_all('/Added training example with performance ([\d.]+)/', $log_content, $performance_matches, PREG_SET_ORDER);
    foreach ($performance_matches as $i => $match) {
        if (!isset($info['configurations'][$i])) {
            $info['configurations'][$i] = [];
        }
        $info['configurations'][$i]['performance'] = (float)$match[1];
    }
    
    // Calculate overall progress percentage
    $total_steps = $info['total_iterations'] * $info['total_configurations'];
    $current_steps = ($info['current_iteration'] - 1) * $info['total_configurations'] + $info['configurations_evaluated'];
    $info['progress_percent'] = min(100, round(($current_steps / $total_steps) * 100));
    
    return $info;
}

// Function to generate the HTML report
function generate_html_report($info) {
    // Format the timestamp
    $timestamp = date('Y-m-d H:i:s');
    
    // Create configuration table rows
    $config_rows = '';
    foreach ($info['configurations'] as $i => $config) {
        if (isset($config['sharpness']) && isset($config['robustness']) && isset($config['performance'])) {
            $config_rows .= "
            <tr>
                <td>" . ($i+1) . "</td>
                <td>" . number_format($config['sharpness'], 6) . "</td>
                <td>" . number_format($config['robustness'], 4) . "</td>
                <td>" . number_format($config['performance'], 4) . "</td>
            </tr>
            ";
        }
    }
    
    return "<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"UTF-8\">
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
    <title>Meta-Model Training Progress</title>
    <style>
        body {
            padding: 20px;
            font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background-color: #111;
            color: #eee;
            line-height: 1.6;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #4cf;
        }
        .header {
            margin-bottom: 30px;
            border-bottom: 1px solid #333;
            padding-bottom: 10px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .status {
            display: inline-block;
            padding: 5px 10px;
            border-radius: 4px;
            font-weight: bold;
            background-color: #264c73;
            color: #4cf;
            margin-left: 10px;
        }
        .card {
            background-color: #222;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.3);
            border: 1px solid #333;
        }
        .progress-container {
            background-color: #333;
            border-radius: 4px;
            height: 25px;
            margin-bottom: 20px;
            position: relative;
        }
        .progress-bar {
            background-color: #4cf;
            height: 100%;
            border-radius: 4px;
            width: {$info['progress_percent']}%;
            transition: width 0.5s ease;
        }
        .progress-text {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            text-align: center;
            line-height: 25px;
            color: white;
            font-weight: bold;
            text-shadow: 1px 1px 1px rgba(0,0,0,0.5);
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        .metric-card {
            background-color: #1a1a1a;
            border-radius: 8px;
            padding: 15px;
            text-align: center;
            border: 1px solid #333;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #4cf;
            margin-bottom: 5px;
        }
        .metric-label {
            font-size: 14px;
            color: #aaa;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }
        th, td {
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #333;
        }
        th {
            background-color: #1a1a1a;
            color: #4cf;
        }
        tr:nth-child(even) {
            background-color: #1a1a1a;
        }
        .footer {
            margin-top: 30px;
            padding-top: 10px;
            border-top: 1px solid #333;
            font-size: 12px;
            color: #777;
        }
        .refresh-button {
            background-color: #264c73;
            color: #4cf;
            border: none;
            padding: 8px 15px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            border: 1px solid #4cf;
            text-decoration: none;
            display: inline-block;
        }
        .refresh-button:hover {
            background-color: #1a3a5a;
        }
        .navigation {
            background-color: #222;
            padding: 10px 0;
            margin-bottom: 20px;
            border-radius: 8px;
        }
        .navigation ul {
            list-style: none;
            padding: 0;
            margin: 0;
            display: flex;
            justify-content: center;
        }
        .navigation li {
            margin: 0 10px;
        }
        .navigation a {
            color: #eee;
            text-decoration: none;
            padding: 5px 10px;
            border-radius: 4px;
        }
        .navigation a:hover {
            background-color: #333;
        }
        .navigation a.active {
            background-color: #264c73;
            color: #4cf;
        }
    </style>
</head>
<body>
    <div class=\"container\">
        <div class=\"navigation\">
            <ul>
                <li><a href=\"index.html\">Home</a></li>
                <li><a href=\"cifar10_progress.html\">CIFAR-10 Progress</a></li>
                <li><a href=\"cifar100_progress.html\">CIFAR-100 Progress</a></li>
                <li><a href=\"meta_model_progress.php\" class=\"active\">Meta-Model Progress</a></li>
                <li><a href=\"about.html\">About</a></li>
            </ul>
        </div>
        
        <div class=\"header\">
            <h1>CIFAR-10 Meta-Model Training Progress <span class=\"status\">IN PROGRESS</span></h1>
            <div>
                <a href=\"?refresh=true\" class=\"refresh-button\">Refresh Data</a>
                <p>Last updated: {$timestamp}</p>
            </div>
        </div>
        
        <div class=\"card\">
            <h2>Training Progress</h2>
            <div class=\"progress-container\">
                <div class=\"progress-bar\"></div>
                <div class=\"progress-text\">{$info['progress_percent']}% Complete</div>
            </div>
            
            <div class=\"metrics-grid\">
                <div class=\"metric-card\">
                    <div class=\"metric-value\">{$info['current_iteration']}/{$info['total_iterations']}</div>
                    <div class=\"metric-label\">Iteration</div>
                </div>
                <div class=\"metric-card\">
                    <div class=\"metric-value\">{$info['configurations_evaluated']}/{$info['total_configurations']}</div>
                    <div class=\"metric-label\">Configurations</div>
                </div>
                <div class=\"metric-card\">
                    <div class=\"metric-value\">{$info['current_resource_level']}</div>
                    <div class=\"metric-label\">Resource Level</div>
                </div>
                <div class=\"metric-card\">
                    <div class=\"metric-value\">" . count($info['configurations']) . "</div>
                    <div class=\"metric-label\">Models Evaluated</div>
                </div>
            </div>
        </div>
        
        <div class=\"card\">
            <h2>Configuration Evaluation Results</h2>
            <table>
                <thead>
                    <tr>
                        <th>Config #</th>
                        <th>Sharpness</th>
                        <th>Perturbation Robustness</th>
                        <th>Performance</th>
                    </tr>
                </thead>
                <tbody>
                    {$config_rows}
                </tbody>
            </table>
        </div>
        
        <div class=\"card\">
            <h2>About Meta-Model Training</h2>
            <p>The meta-model is a surrogate model that predicts the performance of neural network configurations based on their loss landscape characteristics. This approach allows for efficient hyperparameter optimization without training each configuration to completion.</p>
            <p>The training process involves:</p>
            <ol>
                <li>Evaluating multiple configurations at low resource levels</li>
                <li>Measuring loss landscape sharpness and perturbation robustness</li>
                <li>Training a meta-model to predict full-resource performance</li>
                <li>Progressively eliminating poor-performing configurations</li>
            </ol>
        </div>
        
        <div class=\"footer\">
            <p>Generated by LossLandscapeProbe Framework v1.0.0</p>
            <p><a href=\"https://loss.computer-wizard.com.au\" style=\"color: #4cf;\">https://loss.computer-wizard.com.au</a></p>
        </div>
    </div>
</body>
</html>";
}

// If refresh is requested or the report doesn't exist, regenerate it
if ($refresh || !file_exists($output_path)) {
    // Parse the log file
    $info = parse_meta_model_log($log_file);
    
    if ($info) {
        // Generate the HTML report
        $html_content = generate_html_report($info);
        
        // Save the report to both locations
        file_put_contents($output_path, $html_content);
        file_put_contents($web_output_path, $html_content);
    }
}

// Check if the report exists now
if (file_exists($output_path)) {
    // Read the content of the file
    $content = file_get_contents($output_path);
    
    // Output the content
    echo $content;
} else {
    // If the file doesn't exist, display an error message
    echo '<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Meta-Model Report Error</title>
    <style>
        body {
            padding: 20px;
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background-color: #111;
            color: #eee;
            line-height: 1.6;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        h1, h2 {
            color: #4cf;
        }
        .error-card {
            background-color: #2a1a1a;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            border: 1px solid #f88;
        }
        .error-title {
            color: #f88;
            margin-top: 0;
        }
        .refresh-button {
            background-color: #264c73;
            color: #4cf;
            border: none;
            padding: 8px 15px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            border: 1px solid #4cf;
            text-decoration: none;
            display: inline-block;
            margin-top: 20px;
        }
        .refresh-button:hover {
            background-color: #1a3a5a;
        }
        .navigation {
            background-color: #222;
            padding: 10px 0;
            margin-bottom: 20px;
            border-radius: 8px;
        }
        .navigation ul {
            list-style: none;
            padding: 0;
            margin: 0;
            display: flex;
            justify-content: center;
        }
        .navigation li {
            margin: 0 10px;
        }
        .navigation a {
            color: #eee;
            text-decoration: none;
            padding: 5px 10px;
            border-radius: 4px;
        }
        .navigation a:hover {
            background-color: #333;
        }
        .navigation a.active {
            background-color: #264c73;
            color: #4cf;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="navigation">
            <ul>
                <li><a href="index.html">Home</a></li>
                <li><a href="cifar10_progress.html">CIFAR-10 Progress</a></li>
                <li><a href="cifar100_progress.html">CIFAR-100 Progress</a></li>
                <li><a href="meta_model_progress.php" class="active">Meta-Model Progress</a></li>
                <li><a href="about.html">About</a></li>
            </ul>
        </div>
        
        <h1>Meta-Model Report</h1>
        
        <div class="error-card">
            <h2 class="error-title">Report Not Found</h2>
            <p>The meta-model report could not be found at the expected location:</p>
            <code>' . $output_path . '</code>
            <p>Please ensure that meta-model training has been started and that the report has been generated.</p>
        </div>
        
        <p>To generate a meta-model report:</p>
        <ol>
            <li>Run the meta-model training script</li>
            <li>Wait for the training to progress</li>
            <li>The report will be automatically generated</li>
        </ol>
        
        <a href="?refresh=true" class="refresh-button">Try to Generate Report</a>
    </div>
</body>
</html>';
}
?>
