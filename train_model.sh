#!/bin/bash

# train_model.sh - Script to train the meta-model and final model for CIFAR-10/100

# Exit on error and print each command
set -ex

# Parse command line arguments
DATASET="cifar10"  # Default dataset
SAMPLE_SIZE="base"  # Default sample size (base, 10, 20, 30, 40, or transfer for cifar100)
EPOCHS=100
META_MODEL_ONLY=false
SAVE_MODEL=true
GENERATE_REPORT=true
OUTPUT_DIR=""

# Function to display help
show_help() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --dataset DATASET         Dataset to use (cifar10 or cifar100), default: cifar10"
    echo "  --sample-size SIZE       Sample size (base, 10, 20, 30, 40, or transfer for cifar100), default: base"
    echo "  --epochs EPOCHS          Number of epochs for final training, default: 100"
    echo "  --meta-model-only        Only train the meta-model"
    echo "  --no-save-model          Don't save the trained model"
    echo "  --no-report              Don't generate HTML report"
    echo "  --output-dir DIR         Output directory for results"
    echo "  -h, --help               Show this help message"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --sample-size)
            SAMPLE_SIZE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --meta-model-only)
            META_MODEL_ONLY=true
            shift
            ;;
        --no-save-model)
            SAVE_MODEL=false
            shift
            ;;
        --no-report)
            GENERATE_REPORT=false
            shift
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            ;;
    esac
done

# Set default output directory if not specified
if [ -z "$OUTPUT_DIR" ]; then
    if [ "$DATASET" = "cifar100" ] && [ "$SAMPLE_SIZE" = "transfer" ]; then
        OUTPUT_DIR="reports/${DATASET}_transfer"
    else
        OUTPUT_DIR="reports/${DATASET}_${SAMPLE_SIZE}"
    fi
fi

# Create output directories
mkdir -p "$OUTPUT_DIR"
MODEL_DIR="models/${DATASET}/${SAMPLE_SIZE}"
mkdir -p "$MODEL_DIR"

# Set up logging
LOG_FILE="${OUTPUT_DIR}/training_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== Starting training at $(date) ==="
echo "Dataset: $DATASET"
echo "Sample size: $SAMPLE_SIZE"
echo "Epochs: $EPOCHS"
echo "Output directory: $OUTPUT_DIR"
echo "Model directory: $MODEL_DIR"

# Activate Python environment if needed
# Uncomment and modify if using a virtual environment
# source /path/to/venv/bin/activate

# Run the training script with appropriate parameters
python unified_cifar_training.py \
    --dataset "$DATASET" \
    --sample-size "$SAMPLE_SIZE" \
    --epochs "$EPOCHS" \
    --outdir "$OUTPUT_DIR" \
    $( [ "$META_MODEL_ONLY" = true ] && echo "--meta-model-only" ) \
    $( [ "$SAVE_MODEL" = true ] && echo "--save-model" ) \
    $( [ "$GENERATE_REPORT" = true ] && echo "--generate-report" )

# Check if training was successful
if [ $? -eq 0 ]; then
    echo "=== Training completed successfully at $(date) ==="
    echo "Results saved to: $OUTPUT_DIR"
    echo "Model saved to: $MODEL_DIR"
else
    echo "=== Training failed - check the log file: $LOG_FILE ==="
    exit 1
fi

exit 0
