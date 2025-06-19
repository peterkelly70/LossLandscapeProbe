#!/usr/bin/env python3
"""
Visualization Script for Sample Percentage Comparison Results

This script generates visualizations from the sample percentage comparison experiments
to help analyze the trade-offs between different sample percentages.
"""

import os
import json
import glob
import pandas as pd
import numpy as np
import argparse
from datetime import datetime

# Parse command line arguments
parser = argparse.ArgumentParser(description='Visualize sample percentage comparison results')
parser.add_argument('--headless', type=lambda x: x.lower() == 'true', default=True,
                    help='Run in headless mode without GUI (default: True)')
parser.add_argument('--show-plots', type=lambda x: x.lower() == 'true', default=False,
                    help='Show plots on screen (not for headless systems) (default: False)')
parser.add_argument('--save-plots', type=lambda x: x.lower() == 'true', default=True,
                    help='Save plots to disk (default: True)')
parser.add_argument('--output-dir', type=str, default=None,
                    help='Directory to save visualization outputs')
args = parser.parse_args()

# Configure matplotlib based on headless mode
if args.headless:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend for headless mode

import matplotlib.pyplot as plt
import seaborn as sns

# Set up paths
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
PLOTS_DIR = args.output_dir if args.output_dir else os.path.join(RESULTS_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

# Date-time string for unique filenames
DATE_STR = datetime.now().strftime("%Y%m%d_%H%M%S")

def load_results(dataset='cifar10'):
    """Load results from CSV files."""
    # Find the most recent CSV file for the dataset
    pattern = os.path.join(RESULTS_DIR, f'{dataset}_sample_size_comparison_*.csv')
    files = sorted(glob.glob(pattern))
    
    if not files:
        raise FileNotFoundError(f"No result files found for {dataset}")
    
    # Load the most recent file
    latest_file = files[-1]
    print(f"Loading results from {latest_file}")
    
    df = pd.read_csv(latest_file)
    return df, latest_file

def load_history(dataset='cifar10'):
    """Load training history from JSON files."""
    histories = {}
    
    # Find all JSON files for each sample percentage
    for sample_size in [0.1, 0.2, 0.3, 0.4]:
        pattern = os.path.join(RESULTS_DIR, f'{dataset}_sample_size_{sample_size}_*.json')
        files = sorted(glob.glob(pattern))
        
        if files:
            # Load the most recent file
            latest_file = files[-1]
            with open(latest_file, 'r') as f:
                data = json.load(f)
                histories[sample_size] = data['history']
    
    return histories

def plot_accuracy_vs_time(df, dataset='cifar10'):
    """Plot test accuracy vs total time for different dataset percentages."""
    plt.figure(figsize=(12, 7))
    
    # Convert sample_size to percentage for better readability
    df['dataset_percentage'] = df['sample_size'] * 100
    
    # Create scatter plot
    scatter = sns.scatterplot(
        x='total_time', 
        y='final_test_acc', 
        hue='dataset_percentage',
        size='dataset_percentage',
        sizes=(100, 250),
        palette='viridis',
        data=df,
        legend='full'
    )
    
    # Add labels for each point
    for i, row in df.iterrows():
        percentage = int(row['sample_size'] * 100)
        plt.annotate(
            f"{percentage}%",
            (row['total_time'], row['final_test_acc']),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center',
            fontweight='bold',
            fontsize=11
        )
    
    # Update legend to show percentages
    handles, labels = scatter.get_legend_handles_labels()
    new_labels = [f"{float(label):.0f}% Dataset" for label in labels]
    plt.legend(handles, new_labels, title="Dataset Size", fontsize=10)
    
    plt.title(f'{dataset.upper()} Test Accuracy vs Training Time by Dataset Percentage', fontsize=14)
    plt.xlabel('Total Training Time (seconds)', fontsize=12)
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add explanatory text
    plt.figtext(0.5, 0.01, 
               f"This plot shows how model accuracy relates to training time when using different percentages of the {dataset.upper()} dataset.\n"
               f"For each percentage, multiple training runs were performed with different hyperparameter perturbations.", 
               ha='center', fontsize=10, bbox=dict(boxstyle='round,pad=1', facecolor='aliceblue', alpha=0.5))
    
    # Save plot if requested
    if args.save_plots:
        plot_path = os.path.join(PLOTS_DIR, f'{dataset}_accuracy_vs_time_{DATE_STR}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {plot_path}")
    
    # Show plot if requested and not in headless mode
    if args.show_plots and not args.headless:
        plt.show()
    elif args.show_plots and args.headless:
        print("Warning: Cannot show plots in headless mode")
    
    return plot_path if args.save_plots else None

def plot_meta_vs_training_time(df, dataset='cifar10'):
    """Plot meta-model time vs training time for different sample percentages."""
    plt.figure(figsize=(10, 6))
    
    # Create bar chart
    df_sorted = df.sort_values('sample_size')
    x = np.arange(len(df_sorted))
    width = 0.35
    
    plt.bar(x - width/2, df_sorted['meta_time'], width, label='Meta-Model Time')
    plt.bar(x + width/2, df_sorted['training_time'], width, label='Training Time')
    
    plt.xlabel('Sample Percentage')
    plt.ylabel('Time (seconds)')
    plt.title(f'{dataset.upper()} Meta-Model Time vs Training Time')
    plt.xticks(x, [str(level) for level in df_sorted['sample_size']])
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add total time as text on top of bars
    for i, row in enumerate(df_sorted.itertuples()):
        plt.text(i, row.meta_time + row.training_time + 10, 
                 f"Total: {row.total_time:.0f}s", 
                 ha='center')
    
    # Save plot
    plot_path = os.path.join(PLOTS_DIR, f'{dataset}_time_breakdown_{DATE_STR}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {plot_path}")
    
    return plot_path

def plot_hyperparameter_comparison(df, dataset='cifar10'):
    """Plot hyperparameter values for different dataset percentages."""
    # Select only hyperparameter columns
    hyper_cols = [col for col in df.columns if col.startswith('best_') and col != 'best_epoch']
    
    if not hyper_cols:
        print(f"No hyperparameter columns found for {dataset}")
        return None
        
    # Convert sample_size to percentage for better readability
    df['dataset_percentage'] = df['sample_size'] * 100
    
    # Separate numeric and string hyperparameters
    numeric_cols = []
    string_cols = []
    
    for col in hyper_cols:
        if df[col].dtype == 'object':
            string_cols.append(col)
        else:
            numeric_cols.append(col)
    
    # Create a long-format dataframe for numeric parameters
    numeric_data = []
    for i, row in df.iterrows():
        percentage = int(row['sample_size'] * 100)
        for col in numeric_cols:
            param_name = col.replace('best_', '')
            numeric_data.append({
                'Dataset Percentage': f"{percentage}%",
                'Hyperparameter': param_name.replace('_', ' ').title(),
                'Value': row[col]
            })
    
    # Create a separate dataframe for string parameters
    string_data = []
    for i, row in df.iterrows():
        percentage = int(row['sample_size'] * 100)
        for col in string_cols:
            param_name = col.replace('best_', '')
            string_data.append({
                'Dataset Percentage': f"{percentage}%",
                'Hyperparameter': param_name.replace('_', ' ').title(),
                'Value': row[col].upper()  # Store string value
            })
    
    # Create dataframes
    numeric_df = pd.DataFrame(numeric_data) if numeric_data else pd.DataFrame()
    string_df = pd.DataFrame(string_data) if string_data else pd.DataFrame()
    
    # Create plot for numeric hyperparameters
    if not numeric_df.empty:
        plt.figure(figsize=(14, 9))
        
        # Use catplot for grouped bar chart
        g = sns.catplot(
            data=numeric_df,
            x='Hyperparameter',
            y='Value',
            hue='Dataset Percentage',
            kind='bar',
            height=7,
            aspect=1.6,
            palette='viridis',
            legend_out=False
        )
        
        g.set_xticklabels(rotation=45, ha='right')
        g.fig.suptitle(f'{dataset.upper()} Optimal Numeric Hyperparameters by Dataset Percentage', y=1.05, fontsize=16)
        g.set_axis_labels("Hyperparameter", "Value", fontsize=12)
    
    # Add explanatory text for numeric hyperparameters
    if not numeric_df.empty:
        plt.figtext(0.5, 0.01, 
                  "This plot shows the optimal numeric hyperparameter values determined by the meta-model for different dataset percentages. "
                  "Each group represents a hyperparameter, with bars showing the optimal value for each dataset percentage.",
                  ha='center', fontsize=10, wrap=True)
        
        # Save numeric hyperparameter plot
        numeric_plot_path = os.path.join(PLOTS_DIR, f'{dataset}_numeric_hyperparameter_comparison_{DATE_STR}.png')
        plt.savefig(numeric_plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved numeric hyperparameter plot to {numeric_plot_path}")
    
    # Create plot for categorical hyperparameters (like optimizer)
    if not string_df.empty:
        # Get unique hyperparameters
        categorical_params = string_df['Hyperparameter'].unique()
        
        for param in categorical_params:
            param_df = string_df[string_df['Hyperparameter'] == param].copy()
            
            # Create a table for categorical values
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.axis('tight')
            ax.axis('off')
            
            # Extract data for the table
            percentages = param_df['Dataset Percentage'].tolist()
            values = param_df['Value'].tolist()
            
            # Create table data
            table_data = [[p, v] for p, v in zip(percentages, values)]
            
            # Create the table
            table = ax.table(
                cellText=table_data,
                colLabels=['Dataset Percentage', param],
                cellLoc='center',
                loc='center',
                colColours=['#f0f0f0', '#f0f0f0']
            )
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1.2, 1.5)
            
            plt.title(f'{dataset.upper()} Optimal {param} by Dataset Percentage', fontsize=14)
            
            # Add explanatory text
            plt.figtext(0.5, 0.01,
                      f"This table shows the optimal {param} determined by the meta-model for different dataset percentages.",
                      ha='center', fontsize=10, wrap=True)
            
            # Save categorical plot
            cat_plot_path = os.path.join(PLOTS_DIR, f'{dataset}_{param.lower().replace(" ", "_")}_comparison_{DATE_STR}.png')
            plt.savefig(cat_plot_path, dpi=300, bbox_inches='tight')
            print(f"Saved {param} comparison plot to {cat_plot_path}")
    
    # Return the path to one of the plots
    if not numeric_df.empty:
        return numeric_plot_path
    elif not string_df.empty:
        return cat_plot_path
    else:
        return None

def plot_training_curves(histories, dataset='cifar10'):
    """Plot training and test curves for different sample percentages."""
    plt.figure(figsize=(15, 10))
    
    # Plot training loss
    plt.subplot(2, 1, 1)
    for sample_size, history in histories.items():
        percentage = int(sample_size * 100)
        plt.plot(history['train_loss'], label=f'{percentage}% Dataset')
    plt.title(f'{dataset.upper()} Training Loss by Dataset Percentage', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Dataset Size', fontsize=11)
    
    # Plot test accuracy
    plt.subplot(2, 1, 2)
    for sample_size, history in histories.items():
        percentage = int(sample_size * 100)
        plt.plot(history['test_acc'], label=f'{percentage}% Dataset', linewidth=2)
    plt.title(f'{dataset.upper()} Test Accuracy by Dataset Percentage', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Dataset Size', fontsize=11)
    
    plt.tight_layout()
    plt.suptitle(f'{dataset.upper()} Training Performance with Varying Dataset Sizes', y=1.02, fontsize=16)
    
    # Add explanatory text
    plt.figtext(0.5, 0.01, 
               f"Each line represents training with a different percentage of the {dataset.upper()} dataset.\n"
               f"For each percentage, multiple random subsets were used with various hyperparameter perturbations.", 
               ha='center', fontsize=10, bbox=dict(boxstyle='round,pad=1', facecolor='aliceblue', alpha=0.5))
    
    # Save plot
    plot_path = os.path.join(PLOTS_DIR, f'{dataset}_training_curves_{DATE_STR}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {plot_path}")
    
    return plot_path

def create_summary_table(df, dataset='cifar10'):
    """Create a summary table with key metrics for different dataset percentages."""
    # Convert sample_size to percentage for better readability
    df['dataset_percentage'] = df['sample_size'] * 100
    
    # Calculate efficiency metrics
    df['accuracy_per_second'] = df['final_test_acc'] / df['total_time']
    df['perturbations_count'] = df['num_perturbations']
    df['random_subsets'] = df['num_subsets']
    
    # Format the table
    table = df[['dataset_percentage', 'random_subsets', 'perturbations_count', 
                'meta_time', 'training_time', 'total_time', 
                'final_test_acc', 'accuracy_per_second']].copy()
    
    # Round numeric columns
    table['dataset_percentage'] = table['dataset_percentage'].astype(int)
    table['meta_time'] = table['meta_time'].round(1)
    table['training_time'] = table['training_time'].round(1)
    table['total_time'] = table['total_time'].round(1)
    table['final_test_acc'] = table['final_test_acc'].round(2)
    table['accuracy_per_second'] = (table['accuracy_per_second'] * 1000).round(2)
    
    # Rename columns for better readability
    table = table.rename(columns={
        'dataset_percentage': 'Dataset Size (%)',
        'random_subsets': 'Random Subsets',
        'perturbations_count': 'Hyperparameter Perturbations',
        'meta_time': 'Meta-Model Time (s)',
        'training_time': 'Training Time (s)',
        'total_time': 'Total Time (s)',
        'final_test_acc': 'Test Accuracy (%)',
        'accuracy_per_second': 'Efficiency (Acc/Time × 1000)'
    })
    
    # Sort by dataset percentage
    table = table.sort_values('Dataset Size (%)')
    
    # Add a description to the CSV
    description = f"""
# {dataset.upper()} Resource Comparison Summary
# Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
#
# This table shows how model performance varies with different dataset percentages.
# For each percentage, multiple random subsets were used with various hyperparameter perturbations.
# - Dataset Size (%): Percentage of the full dataset used for training
# - Random Subsets: Number of different random subsets used at each percentage level
# - Hyperparameter Perturbations: Number of hyperparameter variations tested per subset
# - Meta-Model Time: Time spent on meta-model training and prediction
# - Training Time: Time spent on actual model training
# - Total Time: Combined time for the entire process
# - Test Accuracy: Final accuracy achieved on the test set
# - Efficiency: Accuracy per unit time (higher is better)
"""
    
    # Save table to CSV with description
    table_path = os.path.join(PLOTS_DIR, f'{dataset}_summary_table_{DATE_STR}.csv')
    with open(table_path, 'w') as f:
        f.write(description)
        f.write(table.to_csv(index=False))
    print(f"Saved summary table to {table_path}")
    
    return table

def visualize_results(dataset='cifar10'):
    """Generate all visualizations for a dataset."""
    print(f"\nVisualizing results for {dataset.upper()}...")
    
    # Load results
    try:
        df, csv_file = load_results(dataset)
        histories = load_history(dataset)
        
        # Generate plots
        plot_accuracy_vs_time(df, dataset)
        plot_meta_vs_training_time(df, dataset)
        plot_hyperparameter_comparison(df, dataset)
        
        if histories:
            plot_training_curves(histories, dataset)
        
        # Create summary table
        summary_table = create_summary_table(df, dataset)
        print(f"\nSummary for {dataset.upper()}:")
        print(summary_table.to_string(index=False))
        
        print(f"\nAll visualizations for {dataset.upper()} completed successfully.")
        
    except FileNotFoundError as e:
        print(f"Warning: {e}")
    except Exception as e:
        print(f"Error visualizing {dataset} results: {e}")

def main():
    """Main function to generate all visualizations."""
    print("Sample Percentage Comparison Visualization")
    print("======================================")
    
    # Print configuration
    print(f"\nConfiguration:")
    print(f"  Headless mode: {args.headless}")
    print(f"  Show plots: {args.show_plots}")
    print(f"  Save plots: {args.save_plots}")
    print(f"  Output directory: {PLOTS_DIR}")
    
    # Visualize CIFAR-10 results
    visualize_results('cifar10')
    
    # Visualize CIFAR-100 results
    visualize_results('cifar100')
    
    if args.save_plots:
        print("\nVisualization complete. All plots saved to:", PLOTS_DIR)
    else:
        print("\nVisualization complete. No plots were saved (save_plots=False).")
        
    if args.show_plots and args.headless:
        print("\nNote: Plots were not displayed because headless mode is enabled.")
        print("To view plots, either disable headless mode or check the saved files.")
    elif not args.show_plots:
        print("\nNote: Plots were not displayed because show_plots is disabled.")

if __name__ == "__main__":
    main()
