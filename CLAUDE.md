# Chess Reasoner Project Guidelines

## Setup & Commands
- Install dependencies: `pip install -e .`
- Install dev dependencies: `pip install unsloth vllm datasets chess wandb`
- Run training: `python chess_grpo.py`

## Code Style
- Imports: Standard library first, then third-party, then local
- Use type hints for function parameters and return types
- Use PEP8 formatting (4 spaces for indentation)
- Use meaningful variable names and docstrings
- Error handling: Use try/except blocks for chess and dataset operations
- Function organization: Group related functions together

## Project Structure
- Main training script: chess_grpo.py
- Dataset utilities: convert_dset.py
- Models are saved to ./outputs directory