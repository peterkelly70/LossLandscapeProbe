#!/usr/bin/env python3
"""
LossLandscapeProbe Example Runner

This script provides an interactive menu to run different examples and experiments
in the LossLandscapeProbe framework.
"""

import os
import sys
import subprocess
import logging
import argparse
from datetime import datetime

try:
    import curses
    from curses import wrapper
    CURSES_AVAILABLE = True
except ImportError:
    CURSES_AVAILABLE = False
    print("Warning: curses module not available. Falling back to simple menu.")

try:
    from rich.markdown import Markdown
    from rich.console import Console
    from rich.pager import Pager
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Warning: rich module not available. Install with 'pip install rich' for markdown rendering.")

# Add the parent directory and src/ to the path so we can import packages
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))

# Define a simple logging setup instead of importing from llp.utils
def setup_logger():
    """Set up a basic logger for the script"""
    log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

# Setup logging
setup_logger()
logger = logging.getLogger(__name__)

# Define available examples with their arguments and categories
EXAMPLES = {
    # Documentation category
    "documentation": {
        "name": "Documentation",
        "type": "category",
        "items": {
            "view_readme": {
                "description": "View the project README.md in the terminal",
                "script": "internal",
                "internal_function": "view_markdown",
                "file": "README.md",
                "args": []
            },
            "view_whitepaper": {
                "description": "View the project whitepaper.md in the terminal",
                "script": "internal",
                "internal_function": "view_markdown",
                "file": "whitepaper.md",
                "args": []
            }
        }
    },
    
    # Training category
    "training": {
        "name": "Train Models",
        "type": "category",
        "items": {
            "cifar10": {
                "name": "CIFAR-10",
                "description": "Train models on CIFAR-10 dataset",
                "type": "category",
                "items": {
                    "multisize": {
                        "description": "Multiple Passes with 10/20/30/40% Dataset Slices to Train Meta-Model",
                        "script": "src/training/unified_cifar_training.py",
                        "args": [
                            {"name": "--dataset", "help": "Dataset to use", "default": "cifar10", "type": "str"},
                            {"name": "--mode", "help": "Training mode", "default": "multisize", "type": "str"},
                            {"name": "--epochs", "help": "Number of epochs", "default": "100", "type": "int"},
                            {"name": "--batch-size", "help": "Batch size", "default": "128", "type": "int"},
                            {"name": "--num-iterations", "help": "Number of meta-model iterations", "default": "3", "type": "int"},
                            {"name": "--num-configs", "help": "Number of configurations per iteration", "default": "10", "type": "int"}
                        ]
                    },
                    "sample_size_10": {
                        "description": "Train Meta-Model Using 10% Dataset Slices",
                        "script": "src/training/unified_cifar_training.py",
                        "args": [
                            {"name": "--dataset", "help": "Dataset to use", "default": "cifar10", "type": "str", "preselected": True},
                            {"name": "--mode", "help": "Training mode", "default": "single", "type": "str", "preselected": True},
                            {"name": "--sample-size", "help": "Sample size", "default": "0.1", "type": "float", "preselected": True}
                        ]
                    },
                    "sample_size_20": {
                        "description": "Train Meta-Model Using 20% Dataset Slices",
                        "script": "src/training/unified_cifar_training.py",
                        "args": [
                            {"name": "--dataset", "help": "Dataset to use", "default": "cifar10", "type": "str", "preselected": True},
                            {"name": "--mode", "help": "Training mode", "default": "single", "type": "str", "preselected": True},
                            {"name": "--sample-size", "help": "Sample size", "default": "0.2", "type": "float", "preselected": True}
                        ]
                    },
                    "sample_size_30": {
                        "description": "Train Meta-Model Using 30% Dataset Slice",
                        "script": "src/training/unified_cifar_training.py",
                        "args": [
                            {"name": "--dataset", "help": "Dataset to use", "default": "cifar10", "type": "str", "preselected": True},
                            {"name": "--mode", "help": "Training mode", "default": "single", "type": "str", "preselected": True},
                            {"name": "--sample-size", "help": "Sample size", "default": "0.3", "type": "float", "preselected": True},
                            {"name": "--epochs", "help": "Number of epochs", "default": "100", "type": "int", "preselected": True},
                            {"name": "--batch-size", "help": "Batch size", "default": "128", "type": "int", "preselected": True}
                        ]
                    },
                    "sample_size_40": {
                        "description": "Train Meta-Model Using 40% Dataset Slices",
                        "script": "src/training/unified_cifar_training.py",
                        "args": [
                            {"name": "--dataset", "help": "Dataset to use", "default": "cifar10", "type": "str", "preselected": True},
                            {"name": "--mode", "help": "Training mode", "default": "single", "type": "str", "preselected": True},
                            {"name": "--sample-size", "help": "Sample size", "default": "0.4", "type": "float", "preselected": True},
                            {"name": "--epochs", "help": "Number of epochs", "default": "100", "type": "int", "preselected": True},
                            {"name": "--batch-size", "help": "Batch size", "default": "128", "type": "int", "preselected": True}
                        ]
                    }
                }
            },
            "cifar100": {
                "name": "CIFAR-100",
                "description": "Train models on CIFAR-100 dataset",
                "type": "category",
                "items": {
                    "multisize": {
                        "description": "Multiple Passes with 10/20/30/40% Dataset Slices to Train Meta-Model",
                        "script": "src/training/unified_cifar_training.py",
                        "args": [
                            {"name": "--dataset", "help": "Dataset to use", "default": "cifar100", "type": "str"},
                            {"name": "--mode", "help": "Training mode", "default": "multisize", "type": "str"},
                            {"name": "--epochs", "help": "Number of epochs", "default": "100", "type": "int"},
                            {"name": "--batch-size", "help": "Batch size", "default": "128", "type": "int"},
                            {"name": "--num-iterations", "help": "Number of meta-model iterations", "default": "3", "type": "int"},
                            {"name": "--num-configs", "help": "Number of configurations per iteration", "default": "10", "type": "int"}
                        ]
                    },
                    "sample_size_10": {
                        "description": "Train Meta-Model Using 10% Dataset Slice",
                        "script": "src/training/unified_cifar_training.py",
                        "args": [
                            {"name": "--dataset", "help": "Dataset to use", "default": "cifar100", "type": "str"},
                            {"name": "--mode", "help": "Training mode", "default": "single", "type": "str"},
                            {"name": "--sample-size", "help": "Sample size", "default": "0.1", "type": "float"},
                            {"name": "--epochs", "help": "Number of epochs", "default": "100", "type": "int"},
                            {"name": "--batch-size", "help": "Batch size", "default": "128", "type": "int"}
                        ]
                    },
                    "sample_size_20": {
                        "description": "Train Meta-Model Using 20% Dataset Slice",
                        "script": "src/training/unified_cifar_training.py",
                        "args": [
                            {"name": "--dataset", "help": "Dataset to use", "default": "cifar100", "type": "str"},
                            {"name": "--mode", "help": "Training mode", "default": "single", "type": "str"},
                            {"name": "--sample-size", "help": "Sample size", "default": "0.2", "type": "float"},
                            {"name": "--epochs", "help": "Number of epochs", "default": "100", "type": "int"},
                            {"name": "--batch-size", "help": "Batch size", "default": "128", "type": "int"}
                        ]
                    },
                    "sample_size_30": {
                        "description": "Train Meta-Model Using 30% Dataset Slices",
                        "script": "src/training/unified_cifar_training.py",
                        "args": [
                            {"name": "--dataset", "help": "Dataset to use", "default": "cifar100", "type": "str", "preselected": True},
                            {"name": "--mode", "help": "Training mode", "default": "single", "type": "str", "preselected": True},
                            {"name": "--sample-size", "help": "Sample size", "default": "0.3", "type": "float", "preselected": True},
                            {"name": "--epochs", "help": "Number of epochs", "default": "100", "type": "int", "preselected": True},
                            {"name": "--batch-size", "help": "Batch size", "default": "128", "type": "int", "preselected": True}
                        ]
                    },
                    "sample_size_40": {
                        "description": "Train Meta-Model Using 40% Dataset Slices",
                        "script": "src/training/unified_cifar_training.py",
                        "args": [
                            {"name": "--dataset", "help": "Dataset to use", "default": "cifar100", "type": "str", "preselected": True},
                            {"name": "--mode", "help": "Training mode", "default": "single", "type": "str", "preselected": True},
                            {"name": "--sample-size", "help": "Sample size", "default": "0.4", "type": "float", "preselected": True},
                            {"name": "--epochs", "help": "Number of epochs", "default": "100", "type": "int", "preselected": True},
                            {"name": "--batch-size", "help": "Batch size", "default": "128", "type": "int", "preselected": True}
                        ]
                    }
                }
            }
        }
    },
    
    # Visualization category
    "visualization": {
        "name": "Visualization",
        "type": "category",
        "items": {
            "visualize": {
                "description": "Visualize the loss landscape",
                "script": "src/visualization/visualize_resource_comparison.py",
                "args": [
                    {"name": "--save-plots", "help": "Save plots to disk", "default": "True", "type": "bool"},
                    {"name": "--show-plots", "help": "Show plots (not for headless systems)", "default": "False", "type": "bool"},
                    {"name": "--headless", "help": "Run in headless mode (no GUI)", "default": "True", "type": "bool"},
                    {"name": "--output-dir", "help": "Directory to save visualization outputs", "default": "outputs/visualizations", "type": "str"}
                ]
            },
            "visualize_progress": {
                "description": "Visualize training progress for any dataset and resource level",
                "script": "src/visualization/visualize_progress.py",
                "args": [
                    {"name": "--log", "help": "Path to training log file", "type": "str"},
                    {"name": "--output", "help": "Path to output HTML report", "type": "str"},
                    {"name": "--dataset", "help": "Dataset type (cifar10 or cifar100)", "type": "str", "choices": ["cifar10", "cifar100"]},
                    {"name": "--sample-size", "help": "Resource level (10, 20, 30, 40, or 100 for full dataset)", "type": "str", "choices": ["10", "20", "30", "40", "100"]},
                    {"name": "--title", "help": "Custom title for the report", "type": "str"}
                ]
            },
            "generate_reports": {
                "description": "Generate all reports for models (test, training progress, transfer learning)",
                "script": "internal",
                "internal_function": "generate_all_reports",
                "args": []
            }
        }
    },
    
    # Deployment category
    "deployment": {
        "name": "Deployment",
        "type": "category",
        "items": {
            "setup_website": {
                "description": "Set up the website to display results (requires sudo)",
                "script": "src/deployment/setup_website.py",
                "args": [],
                "sudo": True,
                "use_mksite": True
            }
        }
    }
}

def interactive_menu():
    """Display an interactive menu for selecting examples."""
    while True:
        if CURSES_AVAILABLE:
            # Use curses-based menu
            example_name, args = curses.wrapper(_curses_menu)
            if example_name:
                run_example(example_name, args)
                # Pause to let user see the results
                input("\nPress Enter to return to the menu...")
            else:
                # User chose to quit
                break
        else:
            # Fallback to simple text menu
            example_name, args = _simple_menu()
            if example_name:
                run_example(example_name, args)
                # Pause to let user see the results
                input("\nPress Enter to return to the menu...")
            else:
                # User chose to quit
                break

def display_interactive_menu():
    """Display an interactive menu with up/down arrow key selection.
    This is a legacy function kept for compatibility.
    """
    if not CURSES_AVAILABLE:
        print("Curses library not available, using simple menu.")
        return _simple_menu()
    else:
        return curses.wrapper(_curses_menu)

def _curses_menu(stdscr):
    """Internal function to handle curses-based menu."""
    # Clear screen
    stdscr.clear()
    curses.curs_set(0)  # Hide cursor
    curses.start_color()
    curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)  # Selected item
    curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)  # Category
    curses.init_pair(3, curses.COLOR_CYAN, curses.COLOR_BLACK)  # Argument
    
    # Get screen dimensions
    max_y, max_x = stdscr.getmaxyx()
    
    # Initialize variables
    current_selection = 0
    arg_selection = -1  # -1 means no argument selected
    selected_args = {}
    selected_example = None
    selected_example_category = None
    menu_level = "category"  # Start at category level (top level)
    current_category = None
    current_subcategory = None
    
    # Get available models for report generation
    available_models = find_available_models()
    
    # Main loop
    while True:
        stdscr.clear()
        
        # Display header
        header = "LossLandscapeProbe Interactive Menu"
        stdscr.addstr(0, 0, header, curses.A_BOLD)
        stdscr.addstr(1, 0, "=" * len(header))
        
        # Display navigation instructions
        nav_instructions = "Use arrow keys to navigate, Enter to select, Backspace to go back, 'q' to quit"
        if max_y > 3:
            stdscr.addstr(2, 0, nav_instructions)
        
        # Get category names
        category_names = sorted(list(EXAMPLES.keys()))
        
        # Handle hierarchical menu levels
        if menu_level == "category":
            # Display top-level categories
            stdscr.addstr(4, 0, "Categories:", curses.A_BOLD | curses.color_pair(2))
            
            for i, category_id in enumerate(category_names):
                category = EXAMPLES[category_id]
                y_pos = 6 + i
                
                if y_pos < max_y - 2:
                    if i == current_selection:
                        stdscr.addstr(y_pos, 0, f"> {category['name']}", curses.color_pair(1))
                    else:
                        stdscr.addstr(y_pos, 0, f"  {category['name']}")
        
        elif menu_level == "subcategory":
            # Display items in selected category
            category = EXAMPLES[current_category]
            stdscr.addstr(4, 0, f"Category: {category['name']}", curses.A_BOLD | curses.color_pair(2))
            
            # Get item names
            item_names = sorted(list(category['items'].keys()))
            
            for i, item_id in enumerate(item_names):
                item = category['items'][item_id]
                y_pos = 6 + i
                
                if y_pos < max_y - 2:
                    # Check if this is a subcategory or a direct item
                    if 'type' in item and item['type'] == 'category':
                        # This is a subcategory
                        if i == current_selection:
                            stdscr.addstr(y_pos, 0, f"> {item['name']} >", curses.color_pair(1))
                        else:
                            stdscr.addstr(y_pos, 0, f"  {item['name']} >")
                    else:
                        # This is a direct item
                        if i == current_selection and arg_selection == -1:
                            stdscr.addstr(y_pos, 0, f"> {item_id:20} - {item['description']}", curses.color_pair(1))
                        else:
                            stdscr.addstr(y_pos, 0, f"  {item_id:20} - {item['description']}")
        
        elif menu_level == "item":
            # Display items in selected subcategory
            category = EXAMPLES[current_category]
            subcategory = category['items'][current_subcategory]
            stdscr.addstr(4, 0, f"Category: {category['name']} > {subcategory['name']}", curses.A_BOLD | curses.color_pair(2))
            
            # Get item names
            item_names = sorted(list(subcategory['items'].keys()))
            
            for i, item_id in enumerate(item_names):
                item = subcategory['items'][item_id]
                y_pos = 6 + i
                
                if y_pos < max_y - 2:
                    if i == current_selection and arg_selection == -1:
                        stdscr.addstr(y_pos, 0, f"> {item_id:20} - {item['description']}", curses.color_pair(1))
                    else:
                        stdscr.addstr(y_pos, 0, f"  {item_id:20} - {item['description']}")
        
        # If an example is selected, show its arguments
        if selected_example is not None and selected_example_category is not None:
            if selected_example_category == 'training':
                # For training category, we need to handle the subcategory structure
                category = EXAMPLES[selected_example_category]
                subcategory = category['items'][current_subcategory]
                example = subcategory['items'][selected_example]
            else:
                # For other categories, use the normal structure
                category = EXAMPLES[selected_example_category]
                example = category['items'][selected_example]
                
            args = example.get('args', [])
            
            # Special handling for report generation examples
            if selected_example in ['generate_reports', 'generate_cifar100_report']:
                # Add model selection options based on available models
                model_args = []
                
                if 'cifar10' in available_models and available_models['cifar10']:
                    for model in available_models['cifar10']:
                        model_name = f"--cifar10-model={model}"
                        model_args.append({"name": model_name, "help": "CIFAR-10 model", "default": model, "type": "str"})
                
                if 'cifar100' in available_models and available_models['cifar100'] and selected_example == 'generate_cifar100_report':
                    for model in available_models['cifar100']:
                        model_name = f"--cifar100-model={model}"
                        model_args.append({"name": model_name, "help": "CIFAR-100 model", "default": model, "type": "str"})
                
                # Add max samples option
                if selected_example == 'generate_reports':
                    model_args.append({"name": "--max-samples", "help": "Max samples in report", "default": "200", "type": "int"})
                
                args = model_args
            
            # Display argument header
            arg_header = f"Arguments for {selected_example}:"
            stdscr.addstr(max_y - len(args) - 4, 0, arg_header, curses.A_BOLD)
            
            # Pre-select arguments that have the preselected flag
            for arg in args:
                arg_name = arg['name']
                # If this is the first time seeing this argument and it has preselected=True
                if arg_name not in selected_args and arg.get('preselected', False):
                    selected_args[arg_name] = True
            
            # Display arguments as checkboxes
            for i, arg in enumerate(args):
                y_pos = max_y - len(args) - 2 + i
                
                if y_pos >= max_y - 2:
                    continue
                
                arg_name = arg['name']
                arg_help = arg.get('help', '')
                is_selected = selected_args.get(arg_name, False)
                
                # Show preselected arguments differently
                is_preselected = arg.get('preselected', False)
                checkbox = "[X]" if is_selected else "[ ]"
                # Add a visual indicator for preselected arguments
                if is_preselected:
                    arg_help = f"{arg_help} (preselected)"
                
                if i == arg_selection:
                    stdscr.addstr(y_pos, 0, f"> {checkbox} {arg_name} - {arg_help}", curses.color_pair(3))
                else:
                    stdscr.addstr(y_pos, 0, f"  {checkbox} {arg_name} - {arg_help}")
            
            # Display run button
            run_y = max_y - 2
            if arg_selection == len(args):
                stdscr.addstr(run_y, 0, "> [RUN EXAMPLE]", curses.color_pair(1))
            else:
                stdscr.addstr(run_y, 0, "  [RUN EXAMPLE]")
        
        # Refresh the screen
        stdscr.refresh()
        
        # Get user input
        key = stdscr.getch()
        
        # Handle navigation
        if key == curses.KEY_UP:
            if arg_selection == -1:
                # Navigate categories, subcategories, or items
                if menu_level == "category":
                    current_selection = (current_selection - 1) % len(category_names)
                elif menu_level == "subcategory":
                    current_selection = (current_selection - 1) % len(category['items'])
                elif menu_level == "item":
                    current_selection = (current_selection - 1) % len(subcategory['items'])
            else:
                # Navigate arguments
                arg_selection = (arg_selection - 1) % (len(args) + 1)
        elif key == curses.KEY_DOWN:
            if arg_selection == -1:
                # Navigate categories, subcategories, or items
                if menu_level == "category":
                    current_selection = (current_selection + 1) % len(category_names)
                elif menu_level == "subcategory":
                    current_selection = (current_selection + 1) % len(category['items'])
                elif menu_level == "item":
                    current_selection = (current_selection + 1) % len(subcategory['items'])
            else:
                # Navigate arguments
                arg_selection = (arg_selection + 1) % (len(args) + 1)
        elif key == curses.KEY_RIGHT or key == ord('\n') or key == ord(' '):
            if menu_level == "category":
                # Enter a category
                current_category = category_names[current_selection]
                menu_level = "subcategory"
                current_selection = 0  # Reset selection for subcategories/items
            elif menu_level == "subcategory":
                # Get the selected item
                category = EXAMPLES[current_category]
                item_names = sorted(list(category['items'].keys()))
                selected_item_id = item_names[current_selection]
                selected_item = category['items'][selected_item_id]
                
                # Check if this is a subcategory or a direct item
                if 'type' in selected_item and selected_item['type'] == 'category':
                    # Enter the subcategory
                    current_subcategory = selected_item_id
                    menu_level = "item"
                    current_selection = 0  # Reset selection for items
                else:
                    # This is a direct item, select it
                    selected_example = selected_item_id
                    selected_example_category = current_category
                    if selected_item.get('args', []) or selected_example in ['generate_reports', 'generate_cifar100_report']:
                        # Check if all arguments are preselected
                        args = selected_item.get('args', [])
                        all_preselected = all(arg.get('preselected', False) for arg in args)
                        
                        if all_preselected:
                            # All arguments are preselected, run immediately
                            args_list = []
                            for arg in args:
                                args_list.append(arg['name'])
                                if arg.get('default'):
                                    args_list.append(arg['default'])
                            return (selected_example_category, selected_example), args_list
                        else:
                            # Some arguments need user selection
                            arg_selection = 0
                    else:
                        # No arguments, run immediately
                        return (selected_example_category, selected_example), []
            elif menu_level == "item" and arg_selection == -1:
                # Select an item from the subcategory
                category = EXAMPLES[current_category]
                subcategory = category['items'][current_subcategory]
                item_names = sorted(list(subcategory['items'].keys()))
                selected_example = item_names[current_selection]
                selected_example_category = 'training'  # Special case for training category
                item = subcategory['items'][selected_example]
                args = item.get('args', [])
                if args:
                    # Check if all arguments are preselected
                    all_preselected = all(arg.get('preselected', False) for arg in args)
                    
                    if all_preselected:
                        # All arguments are preselected, run immediately
                        args_list = []
                        for arg in args:
                            args_list.append(arg['name'])
                            if arg.get('default'):
                                args_list.append(arg['default'])
                        return (current_category, current_subcategory, selected_example), args_list
                    else:
                        # Some arguments need user selection
                        arg_selection = 0
                else:
                    # No arguments, run immediately
                    return (current_category, current_subcategory, selected_example), []
            elif arg_selection >= 0 and arg_selection < len(args):
                # Toggle checkbox
                arg_name = args[arg_selection]['name']
                selected_args[arg_name] = not selected_args.get(arg_name, False)
        elif key == curses.KEY_LEFT or key == ord('\b') or key == 127:  # Left arrow, Backspace, or Delete
            if arg_selection != -1:
                # Move back to item selection
                arg_selection = -1
                selected_example = None
                selected_example_category = None
            elif menu_level == "item":
                # Move back to subcategory selection
                menu_level = "subcategory"
                current_selection = list(category['items'].keys()).index(current_subcategory)
                current_subcategory = None
            elif menu_level == "subcategory":
                # Move back to category selection
                menu_level = "category"
                current_selection = category_names.index(current_category)
                current_category = None
        elif key == ord(' ') and arg_selection >= 0 and arg_selection < len(args):
            # Toggle checkbox
            arg_name = args[arg_selection]['name']
            selected_args[arg_name] = not selected_args.get(arg_name, False)
        elif arg_selection == len(args):
            # Run button selected
            args_list = []
            for arg_name, is_selected in selected_args.items():
                if is_selected:
                    if '=' in arg_name:
                        # For model paths with equals sign, just add the whole argument
                        args_list.append(arg_name)
                    else:
                        # For regular arguments, add the name and default value
                        for arg in args:
                            if arg['name'] == arg_name:
                                args_list.append(arg_name)
                                if arg.get('default'):
                                    args_list.append(arg['default'])
                                break
            
            # Return the appropriate tuple based on the menu structure
            if selected_example_category == 'training' and current_subcategory is not None:
                # For training category with subcategories
                return (selected_example_category, current_subcategory, selected_example), args_list
            else:
                # For regular categories without subcategories
                return (selected_example_category, selected_example), args_list
        elif key == ord('q'):
            # Quit
            return None, None
        elif key == ord('\b') or key == 127:  # Backspace or Delete key
            if menu_level == "item" and arg_selection == -1:
                # Go back to category selection
                menu_level = "category"
                current_selection = category_names.index(current_category)

def simple_menu():
    """Display a simple text-based menu."""
    print("\nLoss Landscape Probe Runner")
    print("================================")
    
    # Display examples alphabetically
    example_names = sorted(list(EXAMPLES.keys()))
    for i, name in enumerate(example_names):
        print(f"{i+1}. {name} - {EXAMPLES[name]['description']}")
    
    print("q. Quit")
    
    # Get user selection
    while True:
        try:
            choice = input("\nSelect an example to run (1-{0} or q): ".format(len(EXAMPLES)))
            
            if choice.lower() == 'q':
                return None, None
            
            idx = int(choice) - 1
            if 0 <= idx < len(EXAMPLES):
                example_name = list(EXAMPLES.keys())[idx]
                
                # Handle arguments
                args_list = []
                example = EXAMPLES[example_name]
                
                if 'args' in example and example['args']:
                    print(f"\nArguments for {example_name}:")
                    
                    for i, arg in enumerate(example['args']):
                        use_arg = input(f"Use {arg['name']} ({arg['help']}, default: {arg['default']})? (y/n): ").lower() == 'y'
                        if use_arg:
                            args_list.append(arg['name'])
                            args_list.append(arg['default'])
                
                return example_name, args_list
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number or 'q'.")

def list_examples():
    """Print a list of available examples in a hierarchical structure."""
    print("\nAvailable Categories:")
    print("====================")
    
    for category_id, category in EXAMPLES.items():
        print(f"\n{category['name']}:")
        print("-" * (len(category['name']) + 1))
        
        if 'items' in category:
            for item_id, item in category['items'].items():
                # Check if this is a subcategory
                if 'type' in item and item['type'] == 'category':
                    # This is a subcategory
                    print(f"  {item['name']}:")
                    
                    # Print items in the subcategory
                    if 'items' in item:
                        for subitem_id, subitem in item['items'].items():
                            print(f"    {subitem_id:18} - {subitem['description']}")
                else:
                    # This is a direct item
                    print(f"  {item_id:20} - {item['description']}")
    
    print("\nUse 'python run.py <example_name> [args]' to run an example.")
    print("Add --help after the example name to see example-specific options.")
    print("Or run without arguments for an interactive menu.")
    print("The interactive menu provides easier navigation through categories and subcategories.")
    print("Use arrow keys to navigate, Enter to select, Backspace to go back, and 'q' to quit.")
    

def view_markdown(file_path):
    """View a markdown file in the terminal with rich formatting."""
    # Check if file exists
    md_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_path)
    if not os.path.exists(md_path):
        print(f"Error: File {file_path} not found.")
        return False
    
    # Read the markdown file
    try:
        with open(md_path, 'r') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return False
    
    # Display with rich if available, otherwise plain text
    print(f"\n{'=' * 80}\n{file_path}\n{'=' * 80}")
    
    if RICH_AVAILABLE:
        console = Console()
        md = Markdown(content)
        
        # Use pager for scrollable display
        try:
            with console.pager():
                console.print(md)
            print(f"\n{'=' * 80}")
            return True
        except Exception as e:
            # Fallback if pager fails
            print(f"\nPager failed: {e}. Using standard output.")
            console.print(md)
            print(f"\n{'=' * 80}")
            return True
    else:
        print("\nRich library not available. Installing it for better markdown rendering:")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "rich"], check=True)
            print("\nRich installed successfully. Please run the command again.")
        except Exception as e:
            print(f"Failed to install rich: {e}")
            
            # Use built-in pager as fallback
            try:
                import pydoc
                print("\nUsing built-in pager:\n")
                pydoc.pager(content)
            except Exception:
                print("\nDisplaying plain text version (no pagination):\n")
                print(content)
        
        return False

def generate_all_reports():
    """Generate all available reports for models without requiring user selection."""
    print("\nGenerating all available reports...\n")
    
    # Define available report types
    report_types = [
        {
            "name": "CIFAR-10 Test Report",
            "script": "src/visualization/generate_test_report.py",
            "args": ["--model-type", "cifa10", "--model", "cifar10_model_trained.pth"]
        },
        {
            "name": "CIFAR-10 Resource Level 10% Test Report",
            "script": "src/visualization/generate_test_report.py",
            "args": ["--model-type", "cifa10_10", "--model", "cifar10_10_model_trained.pth"]
        },
        {
            "name": "CIFAR-10 Resource Level 20% Test Report",
            "script": "src/visualization/generate_test_report.py",
            "args": ["--model-type", "cifa10_20", "--model", "cifar10_20_model_trained.pth"]
        },
        {
            "name": "CIFAR-10 Resource Level 30% Test Report",
            "script": "src/visualization/generate_test_report.py",
            "args": ["--model-type", "cifa10_30", "--model", "cifar10_30_model_trained.pth"]
        },
        {
            "name": "CIFAR-10 Resource Level 40% Test Report",
            "script": "src/visualization/generate_test_report.py",
            "args": ["--model-type", "cifa10_40", "--model", "cifar10_40_model_trained.pth"]
        },
        {
            "name": "CIFAR-100 Test Report",
            "script": "src/visualization/generate_test_report.py",
            "args": ["--model-type", "cifa100", "--model", "cifar100_model_trained.pth"]
        },
        {
            "name": "CIFAR-100 Resource Level 10% Test Report",
            "script": "src/visualization/generate_test_report.py",
            "args": ["--model-type", "cifa100_10", "--model", "cifar100_10_model_trained.pth"]
        },
        {
            "name": "CIFAR-100 Resource Level 20% Test Report",
            "script": "src/visualization/generate_test_report.py",
            "args": ["--model-type", "cifa100_20", "--model", "cifar100_20_model_trained.pth"]
        },
        {
            "name": "CIFAR-100 Resource Level 30% Test Report",
            "script": "src/visualization/generate_test_report.py",
            "args": ["--model-type", "cifa100_30", "--model", "cifar100_30_model_trained.pth"]
        },
        {
            "name": "CIFAR-100 Resource Level 40% Test Report",
            "script": "src/visualization/generate_test_report.py",
            "args": ["--model-type", "cifa100_40", "--model", "cifar100_40_model_trained.pth"]
        },
        {
            "name": "CIFAR-100 Transfer Report",
            "script": "src/visualization/generate_cifar100_report.py",
            "args": ["--cifar10-model", "meta_model_trained.pth", "--cifar100-model", "cifar100_model_trained.pth"]
        },
        {
            "name": "CIFAR-10 Training Progress Report",
            "script": "src/visualization/visualize_progress.py",
            "args": ["--log", "cifar10_training.log", "--output", "reports/cifar10_progress_report.html", "--dataset", "cifar10"]
        },
        {
            "name": "CIFAR-100 Training Progress Report",
            "script": "src/visualization/visualize_progress.py",
            "args": ["--log", "cifar100_transfer.log", "--output", "reports/cifar100_progress_report.html", "--dataset", "cifar100"]
        },
        {
            "name": "CIFAR-10 (10% Resource) Progress Report",
            "script": "src/visualization/visualize_progress.py",
            "args": ["--log", "cifar10_10_training.log", "--output", "reports/cifar10_10_progress_report.html", "--dataset", "cifar10", "--resource-level", "10"]
        },
        {
            "name": "CIFAR-10 (40% Resource) Progress Report",
            "script": "src/visualization/visualize_progress.py",
            "args": ["--log", "cifar10_40_training.log", "--output", "reports/cifar10_40_progress_report.html", "--dataset", "cifar10", "--resource-level", "40"]
        },
        {
            "name": "CIFAR-100 (10% Resource) Progress Report",
            "script": "src/visualization/visualize_progress.py",
            "args": ["--log", "cifar100_10_training.log", "--output", "reports/cifar100_10_progress_report.html", "--dataset", "cifar100", "--resource-level", "10"]
        },
        {
            "name": "CIFAR-100 (40% Resource) Progress Report",
            "script": "src/visualization/visualize_progress.py",
            "args": ["--log", "cifar100_40_training.log", "--output", "reports/cifar100_40_progress_report.html", "--dataset", "cifar100", "--resource-level", "40"]
        }
    ]
    
    # Use all reports
    selected_reports = list(range(len(report_types)))
    
    # Print the reports that will be generated
    print("The following reports will be generated:")
    for i in selected_reports:
        print(f"  - {report_types[i]['name']}")
    print()
    
    # Generate selected reports
    print("\nGenerating reports...")
    success_count = 0
    for i, idx in enumerate(selected_reports):
        report = report_types[idx]
        print(f"\n[{i+1}/{len(selected_reports)}] Generating {report['name']}...")
        
        cmd = [sys.executable, os.path.join(project_root, report['script'])] + report['args']
        try:
            subprocess.run(cmd, check=True)
            success_count += 1
            print(f"{report['name']} generated successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error generating {report['name']}: {e}")
    
    # Automatically set up the website
    print("\nSetting up website with generated reports...")
    try:
        web_dir = "/var/www/html/loss.computer-wizard.com.au"  # Default web directory
        mksite_cmd = ['sudo', 'mksite', web_dir]
        subprocess.run(mksite_cmd, check=True)
        print("Website setup completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error setting up website: {e}")
    
    print(f"\nReport generation complete. {success_count}/{len(selected_reports)} reports generated successfully.")
    return True


def find_available_models():
    """Locate available trained model files (.pth).

    By convention, all checkpoints are stored under the dedicated
    `trained/` directory at the project root.  For backward-compatibility
    we fall back to scanning the whole project if that directory is
    missing.
    """
    models: dict[str, list[str]] = {}
    project_dir = os.path.dirname(os.path.abspath(__file__))

    # Preferred location for checkpoints
    search_root = os.path.join(project_dir, "trained")
    if not os.path.isdir(search_root):
        # Older layouts stored checkpoints alongside code
        search_root = project_dir

    for root, _, files in os.walk(search_root):
        for file in files:
            if not file.endswith(".pth"):
                continue
            rel_path = os.path.relpath(os.path.join(root, file), project_dir)
            lower = file.lower()
            if "cifar10" in lower:
                models.setdefault("cifar10", []).append(rel_path)
            elif "cifar100" in lower:
                models.setdefault("cifar100", []).append(rel_path)
            else:
                models.setdefault("generic", []).append(rel_path)

    return models

def run_example(example_info, args=None):
    """Run the specified example with optional arguments.
    
    Args:
        example_info: Either a string (legacy mode) or a tuple (category_id, item_id) or a tuple (category_id, subcategory_id, item_id)
        args: Optional list of command-line arguments
    """
    # Handle legacy string mode and different tuple modes
    if isinstance(example_info, tuple):
        if len(example_info) == 3:
            # This is a three-level hierarchy (category, subcategory, item)
            category_id, subcategory_id, item_id = example_info
            
            if category_id not in EXAMPLES:
                logger.error(f"Category '{category_id}' not found.")
                print(f"Category '{category_id}' not found. Use 'list' to see available categories.")
                return False
                
            category = EXAMPLES[category_id]
            if subcategory_id not in category['items']:
                logger.error(f"Subcategory '{subcategory_id}' not found in category '{category['name']}'.")
                print(f"Subcategory '{subcategory_id}' not found in category '{category['name']}'. Use 'list' to see available subcategories.")
                return False
                
            subcategory = category['items'][subcategory_id]
            if item_id not in subcategory['items']:
                logger.error(f"Item '{item_id}' not found in subcategory '{subcategory['name']}'.")
                print(f"Item '{item_id}' not found in subcategory '{subcategory['name']}'. Use 'list' to see available items.")
                return False
                
            example = subcategory['items'][item_id]
        else:
            # This is a two-level hierarchy (category, item)
            category_id, item_id = example_info
            
            if category_id not in EXAMPLES:
                logger.error(f"Category '{category_id}' not found.")
                print(f"Category '{category_id}' not found. Use 'list' to see available categories.")
                return False
                
            category = EXAMPLES[category_id]
            if item_id not in category['items']:
                logger.error(f"Item '{item_id}' not found in category '{category['name']}'.")
                print(f"Item '{item_id}' not found in category '{category['name']}'. Use 'list' to see available items.")
                return False
                
            example = category['items'][item_id]
    else:
        # Legacy mode - direct example name
        item_id = example_info
        # Search for the example in all categories
        found = False
        for category_id, category in EXAMPLES.items():
            if 'items' in category and item_id in category['items']:
                example = category['items'][item_id]
                found = True
                break
                
        if not found:
            logger.error(f"Example '{item_id}' not found.")
            print(f"Example '{item_id}' not found. Use 'list' to see available examples.")
            return False
    
    # Handle internal functions
    if example.get('script') == 'internal':
        internal_function = example.get('internal_function')
        if internal_function == 'view_markdown':
            return view_markdown(example.get('file'))
        elif internal_function == 'generate_all_reports':
            return generate_all_reports()
        else:
            logger.error(f"Unknown internal function '{internal_function}'")
            return False
    
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), example['script'])
    
    # Check if the script exists
    if not os.path.exists(script_path):
        logger.error(f"Script '{script_path}' not found.")
        print(f"Script '{script_path}' not found.")
        return False
    
    # For report generation, check for available models
    if item_id in ['generate_reports', 'generate_cifar100_report'] and not args:
        available_models = find_available_models()
        print(f"\nAvailable models for {item_id}:")
        
        if 'cifar10' in available_models and available_models['cifar10']:
            print("\nCIFAR-10 models:")
            for i, model in enumerate(available_models['cifar10']):
                print(f"  {i+1}. {model}")
        else:
            print("  No CIFAR-10 models found.")
            
        if 'cifar100' in available_models and available_models['cifar100']:
            print("\nCIFAR-100 models:")
            for i, model in enumerate(available_models['cifar100']):
                print(f"  {i+1}. {model}")
        else:
            print("  No CIFAR-100 models found.")
            
        if 'generic' in available_models and available_models['generic']:
            print("\nOther models:")
            for i, model in enumerate(available_models['generic']):
                print(f"  {i+1}. {model}")
    
    # Special handling for website setup with mksite
    if item_id == 'setup_website' and example.get('use_mksite', False):
        print("\nUsing mksite command to set up the website...")
        
        # Run mksite command - it will now use the configuration file
        mksite_cmd = ['sudo', 'mksite']
        try:
            print(f"Running: {' '.join(mksite_cmd)}")
            subprocess.run(mksite_cmd, check=True)
            print("mksite command completed successfully.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running mksite: {e}")
    
    # Build the command
    cmd = [sys.executable, script_path]
    
    # Add any additional arguments
    if args:
        cmd.extend(args)
    
    logger.info(f"Running {item_id}: {example['description']}")
    
    # Check if this example requires sudo
    needs_sudo = example.get('sudo', False)
    if needs_sudo:
        print("\nThis example requires superuser privileges.")
        print("You may be prompted for your password.\n")
        sudo_cmd = ['sudo'] + cmd
        logger.info(f"Command (with sudo): {' '.join(sudo_cmd)}")
    else:
        sudo_cmd = cmd
        logger.info(f"Command: {' '.join(cmd)}")
    
    try:
        print(f"\nRunning {item_id}...\n")
        start_time = datetime.now()
        if needs_sudo:
            subprocess.run(sudo_cmd, check=True)
        else:
            subprocess.run(cmd, check=True)
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info(f"Example {item_id} completed successfully in {duration:.2f} seconds")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running example {item_id}: {e}")
        return False

def main():
    """Main function to parse arguments and run examples."""
    parser = argparse.ArgumentParser(
        description="LossLandscapeProbe Example Runner",
        epilog="Use 'list' to see available categories and examples."
    )
    parser.add_argument(
        "example", 
        nargs="?", 
        help="Name of the example to run, or 'list' to see available examples"
    )
    parser.add_argument(
        "args", 
        nargs=argparse.REMAINDER, 
        help="Arguments to pass to the example script"
    )
    
    args = parser.parse_args()
    
    # If no arguments provided, show the interactive menu
    if not args.example:
        interactive_menu()
        return
    
    # Handle list command
    if args.example == "list":
        list_examples()
        return
    
    # Run the specified example with arguments (legacy mode)
    run_example(args.example, args.args)

if __name__ == "__main__":
    main()
