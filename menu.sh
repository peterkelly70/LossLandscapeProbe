#!/bin/bash

# menu.sh - Interactive menu for CIFAR training

# Source the main script for shared functions
source "$(dirname "$0")/train.sh"

# Function to show header
show_header() {
    clear
    echo "╔══════════════════════════════════════════════════════════════════════════════╗"
    echo "║                     CIFAR-10/100 Training Menu                           ║"
    echo "╚══════════════════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "Current Configuration:"
    echo "  Dataset: ${DATASET}"
    echo "  Sample Sizes: ${SAMPLE_SIZES[*]}"
    echo "  Epochs: ${EPOCHS}"
    echo "  Batch Size: ${BATCH_SIZE}"
    echo "  Learning Rate: ${LEARNING_RATE}"
    echo "  Weight Decay: ${WEIGHT_DECAY}"
    echo "  Momentum: ${MOMENTUM}"
    echo "  Workers: ${NUM_WORKERS}"
    echo ""
}

# Function to update parameters
update_parameters() {
    local param=$1
    local prompt=$2
    local current_value=${!param}
    
    read -p "${prompt} [${current_value}]: " new_value
    if [ -n "$new_value" ]; then
        eval "${param}=${new_value}"
    fi
}

# Function to show main menu
show_menu() {
    echo "╔══════════════════════════════════════════════════════════════════════════════╗"
    echo "║                               MAIN MENU                                    ║"
    echo "╠════════════════════════════════════════════════════════════════════════════╣"
    echo "║ 1. Select Dataset (Current: ${DATASET})                                    ║"
    echo "║ 2. Configure Sample Sizes (${SAMPLE_SIZES[*]})                             ║"
    echo "║ 3. Set Training Parameters                                                 ║"
    echo "║ 4. Train with Current Settings                                             ║"
    echo "║ 5. Train All Sample Sizes                                                  ║"
    echo "║ 6. Clear Reports and Models                                                 ║"
    echo "║ 7. Exit                                                                    ║"
    echo "╚════════════════════════════════════════════════════════════════════════════╝"
    echo ""
    read -p "Select an option [1-7]: " choice
    
    case $choice in
        1) select_dataset ;;
        2) configure_sample_sizes ;;
        3) configure_parameters ;;
        4) train_current ;;
        5) train_all_samples ;;
        6) clear_reports ;;
        7) exit 0 ;;
        *) echo "Invalid option. Please try again." ; sleep 1 ;;
    esac
}

# Function to select dataset
select_dataset() {
    echo ""
    echo "Select Dataset:"
    echo "1. CIFAR-10 (10 classes)"
    echo "2. CIFAR-100 (100 classes)"
    read -p "Choose [1-2]: " choice
    
    case $choice in
        1) DATASET="cifar10" ;;
        2) DATASET="cifar100" ;;
        *) echo "Invalid choice. Keeping current dataset." ; sleep 1 ;;
    esac
}

# Function to configure sample sizes
configure_sample_sizes() {
    echo ""
    echo "Current sample sizes: ${SAMPLE_SIZES[*]}"
    echo "Enter new sample sizes separated by spaces (e.g., '10 20 30 40 multi'):"
    read -p "> " new_sizes
    
    if [ -n "$new_sizes" ]; then
        SAMPLE_SIZES=($new_sizes)
    fi
}

# Function to configure training parameters
configure_parameters() {
    while true; do
        clear
        show_header
        echo "╔══════════════════════════════════════════════════════════════════════════════╗"
        echo "║                       TRAINING PARAMETERS                                  ║"
        echo "╠════════════════════════════════════════════════════════════════════════════╣"
        echo "║ 1. Epochs: ${EPOCHS}                                                       ║"
        echo "║ 2. Batch Size: ${BATCH_SIZE}                                              ║"
        echo "║ 3. Learning Rate: ${LEARNING_RATE}                                        ║"
        echo "║ 4. Weight Decay: ${WEIGHT_DECAY}                                          ║"
        echo "║ 5. Momentum: ${MOMENTUM}                                                  ║"
        echo "║ 6. Number of Workers: ${NUM_WORKERS}                                      ║"
        echo "║ 7. Back to Main Menu                                                     ║"
        echo "╚════════════════════════════════════════════════════════════════════════════╝"
        echo ""
        read -p "Select parameter to change [1-7]: " param_choice
        
        case $param_choice in
            1) update_parameters EPOCHS "Enter number of epochs" ;;
            2) update_parameters BATCH_SIZE "Enter batch size" ;;
            3) update_parameters LEARNING_RATE "Enter learning rate" ;;
            4) update_parameters WEIGHT_DECAY "Enter weight decay" ;;
            5) update_parameters MOMENTUM "Enter momentum" ;;
            6) update_parameters NUM_WORKERS "Enter number of workers" ;;
            7) break ;;
            *) echo "Invalid option. Please try again." ; sleep 1 ;;
        esac
    done
}

# Function to train with current settings
train_current() {
    echo ""
    echo "Starting training with current settings..."
    echo "Press Ctrl+C to stop training at any time."
    echo ""
    
    # Use the first sample size by default
    SAMPLE_SIZE=${SAMPLE_SIZES[0]}
    
    # Call the main training function
    train_sample "$SAMPLE_SIZE"
    
    echo ""
    read -p "Press [Enter] to return to the menu..."
}

# Function to train all sample sizes
train_all_samples() {
    echo ""
    echo "This will train models for all sample sizes: ${SAMPLE_SIZES[*]}"
    read -p "Are you sure you want to continue? [y/N] " confirm
    
    if [[ $confirm =~ ^[Yy]$ ]]; then
        for size in "${SAMPLE_SIZES[@]}"; do
            echo ""
            echo "Training with sample size: $size"
            echo "================================"
            train_sample "$size"
        done
    fi
    
    echo ""
    read -p "Press [Enter] to return to the menu..."
}

# Function to clear reports and models
clear_reports() {
    echo ""
    echo "WARNING: This will delete all training reports and models!"
    read -p "Are you sure you want to continue? [y/N] " confirm
    
    if [[ $confirm =~ ^[Yy]$ ]]; then
        echo "Removing reports and models..."
        rm -rf reports/* models/*
        echo "Done."
    fi
    
    read -p "Press [Enter] to continue..."
}

# Main menu loop
while true; do
    show_header
    show_menu
done
