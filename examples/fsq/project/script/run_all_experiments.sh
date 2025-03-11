# Run all training experiments with different settings

# Temperature annealing experiment
echo "Running temperature annealing experiment..."
python run_temperature_annealing.py > temp_annealing.log 2>&1

# Wait for GPU memory to clear
sleep 30

# Regularized training experiment
echo "Running regularized training experiment..."
python run_regularized_training.py > regularized.log 2>&1

# Wait for GPU memory to clear
sleep 30

# Hierarchical quantization experiment
echo "Running hierarchical quantization experiment..."
python run_hierarchical_quantization.py > hierarchical.log 2>&1

# Wait for visualization
sleep 30

# Run visualization
echo "Generating visualizations..."
python visualize_results.py > visualization.log 2>&1

echo "All experiments completed!"