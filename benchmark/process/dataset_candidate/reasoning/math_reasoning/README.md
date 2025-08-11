# Reasoning on MATH-500

Reasoning on MATH-500 dataset with DeepSeek Chat.

```bash
cd /path/to/math_reasoning
python run_infer.py
```

Args:

- `--save_dir`: The directory to save the results.
- `--model`: The model to use.
- `--base_url`: The base URL of the API.
- `--limit`: The number of tasks to process.
- `--stride`: The stride of the tasks to process.
- `--offset`: The offset of the tasks to process.


## Get score

```bash
python get_score.py --save_dir evaluation_results/math500/math500_deepseek-chat
```

Args:

- `--save_dir`: The directory to save the results.