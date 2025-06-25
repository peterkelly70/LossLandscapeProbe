#!/bin/bash

# train.sh - Interactive training script for CIFAR-10/100 with menu

# Exit on error and print each command
set -e

# Check if tqdm is installed for progress bars
if ! command -v python3 -c "import tqdm" &> /dev/null; then
    echo "Installing tqdm for progress bars..."
    pip install tqdm
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
DATASET="cifar10"
SAMPLE_SIZES=("10" "20" "30" "40" "multi")
EPOCHS=100
META_MODEL_ONLY=false
SAVE_MODEL=true
GENERATE_REPORT=true

# Function to display header
show_header() {
    clear
    echo -e "${GREEN}=== CIFAR Training Menu ===${NC}"
    echo -e "Current dataset: ${YELLOW}${DATASET}${NC}"
    echo -e "Current epochs: ${YELLOW}${EPOCHS}${NC}"
    echo "--------------------------------"
}

# Function to clear reports data
clear_reports() {
    local REPORTS_DIR="reports"
    
    if [ ! -d "$REPORTS_DIR" ]; then
        echo -e "${YELLOW}Reports directory does not exist: $REPORTS_DIR${NC}"
        return 0
    fi

    echo -e "${YELLOW}=== Clearing Reports ===${NC}"
    echo -e "This will remove all files in $REPORTS_DIR/"
    echo -e "${RED}WARNING: This action cannot be undone!${NC}"

    # List all subdirectories that will be affected
    echo -e "\nThe following directories will be cleared:"
    find "$REPORTS_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort || echo "(No report directories found)"

    # Ask for confirmation
    read -p "Are you sure you want to continue? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Operation cancelled.${NC}"
        return 0
    fi

    # Clear the reports
    echo -e "\n${YELLOW}Clearing reports...${NC}"
    find "$REPORTS_DIR" -mindepth 2 -type f -exec rm -f {} \; 2>/dev/null

    # Count remaining files
    local REMAINING_FILES=$(find "$REPORTS_DIR" -type f 2>/dev/null | wc -l)

    if [ "$REMAINING_FILES" -eq 0 ]; then
        echo -e "${GREEN}✓ All report files have been removed.${NC}"
    else
        echo -e "${YELLOW}Warning: $REMAINING_FILES files could not be removed.${NC}"
    fi
}

# Function to show menu
show_menu() {
    echo "1. Select Dataset (Current: ${DATASET})"
    echo "2. Set Number of Epochs (Current: ${EPOCHS})"
    echo "3. Train Single Sample Size"
    echo "4. Train All Sample Sizes (10% to 40% + multi)"
    echo "5. Train Meta-Model Only"
    echo "6. Clear Reports Data"
    echo "7. Exit"
    echo -n "Enter your choice [1-7]: "
}

# Function to train a single sample size
train_sample() {
    local sample_size=$1
    local output_dir="reports/${DATASET}_${sample_size}"
    local model_dir="models/${DATASET}/${DATASET}_${sample_size}"
    local checkpoint_dir="${model_dir}/checkpoints"
    local meta_model_dir="${model_dir}/meta_model"
    local log_file="${output_dir}/training.log"
    
    # Create directories
    mkdir -p "${output_dir}"
    mkdir -p "${checkpoint_dir}"
    mkdir -p "${meta_model_dir}"
    
    echo -e "${GREEN}Starting training for ${DATASET} with ${sample_size}% sample size${NC}"
    echo -e "Log file: ${log_file}"
    
    # Clear screen and show initial status
    clear
    echo -e "${GREEN}🚀 Training ${DATASET} (${sample_size}%) - ${EPOCHS} epochs${NC}"
    echo -e "💾 Logs: ${log_file}\n"
    
    # Start training with progress bars
    {
        python -c "
import sys
import re
from tqdm import tqdm

# Initialize progress bars
phase_pbar = tqdm(desc='Phase', bar_format='{desc}: {bar}')
epoch_pbar = tqdm(total=${EPOCHS}, desc='Epochs', 
                 bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

# Update function
def update_progress(line):
    line = line.strip()
    
    # Update phase
    if 'Starting meta-model' in line:
        phase_pbar.set_description('Phase: 🔍 Meta-Model')
    elif 'Starting model training' in line:
        phase_pbar.set_description('Phase: 🏋️ Training')
    elif 'Testing model' in line:
        phase_pbar.set_description('Phase: 🧪 Testing')
    
    # Update epoch progress
    epoch_match = re.search(r'Epoch (\d+)/\d+.*?Loss: ([\d.]+).*?Acc: ([\d.]+)%', line)
    if epoch_match:
        epoch, loss, acc = epoch_match.groups()
        epoch_pbar.update(1)
        epoch_pbar.set_postfix({'loss': f'{float(loss):.3f}', 'acc': f'{float(acc):.1f}%'})
    
    # Clear screen and redraw
    print('\033[2J\033[H', end='')  # Clear screen and move cursor to top
    print(phase_pbar, flush=True)
    print(epoch_pbar, flush=True)

# Main loop
try:
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        update_progress(line)
        
except KeyboardInterrupt:
    print("\nTraining interrupted")
finally:
    phase_pbar.close()
    epoch_pbar.close()
" | python unified_cifar_training.py \
        --dataset "${DATASET}" \
        --sample-size "${sample_size}" \
        --epochs "${EPOCHS}" \
        --outdir "${output_dir}" \
        $( [ "${META_MODEL_ONLY}" = true ] && echo "--meta-model-only" ) \
        $( [ "${SAVE_MODEL}" = true ] && echo "--save-model" ) \
        $( [ "${GENERATE_REPORT}" = true ] && echo "--generate-report" )
    } 2>&1 | tee "${log_file}" >/dev/null  # Suppress duplicate output
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo -e "${GREEN}Training completed successfully!${NC}"
    else
        echo -e "${RED}Training failed! Check the log file: ${log_file}${NC}"
    fi
}

# Function to train all sample sizes
train_all_samples() {
    echo -e "${YELLOW}Starting training for all sample sizes (10% to 40% + multi)...${NC}"
    
    for size in "${SAMPLE_SIZES[@]}"; do
        echo -e "\n${GREEN}=== Processing ${DATASET} ${size}% ===${NC}"
        train_sample "${size}"
    done
    
    echo -e "\n${GREEN}All training jobs completed!${NC}"
}

# Main menu loop
while true; do
    show_header
    show_menu
    
    read -r choice
    case $choice in
        1)
            echo "Select dataset:"
            echo "1. CIFAR-10"
            echo "2. CIFAR-100"
            read -r dataset_choice
            case $dataset_choice in
                1) DATASET="cifar10" ;;
                2) DATASET="cifar100" ;;
                *) echo -e "${RED}Invalid choice!${NC}" ;;
            esac
            ;;
             
        2)
            echo -n "Enter number of epochs: "
            read -r epochs_input
            if [[ "$epochs_input" =~ ^[0-9]+$ ]] && [ "$epochs_input" -gt 0 ]; then
                EPOCHS="$epochs_input"
            else
                echo -e "${RED}Please enter a valid number of epochs!${NC}"
                sleep 2
            fi
            ;;
            
        3)
            echo "Select sample size:"
            echo "1. 10%"
            echo "2. 20%"
            echo "3. 30%"
            echo "4. 40%"
            echo "5. Multi (all sizes)"
            read -r size_choice
            
            case $size_choice in
                1) train_sample "10" ;;
                2) train_sample "20" ;;
                3) train_sample "30" ;;
                4) train_sample "40" ;;
                5) train_sample "multi" ;;
                *) echo -e "${RED}Invalid choice!${NC}" ;;
            esac
            
            echo -n "Press [Enter] to continue..."
            read -r
            ;;
            
        4)
            train_all_samples
            echo -n "Press [Enter] to continue..."
            read -r
            ;;
            
        5)
            META_MODEL_ONLY=true
            echo -e "${YELLOW}Training meta-model only...${NC}"
            train_sample "10"  # Sample size doesn't matter for meta-model only
            META_MODEL_ONLY=false
            echo -n "Press [Enter] to continue..."
            read -r
            ;;
            
        6)
            clear_reports
            read -p "Press any key to continue..." -n1 -s
            ;;
        7)
            echo -e "${GREEN}Exiting...${NC}"
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid option. Please try again.${NC}"
            read -p "Press any key to continue..." -n1 -s
            ;;
    esac
done
