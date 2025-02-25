# Chess Reasoner Project Guidelines

## Setup & Commands
- Install dependencies: `pip install -e .`
- Install dev dependencies: `pip install unsloth vllm datasets chess wandb`
- Run training: `python chess_grpo_clean.py`

## Code Style
- Imports: Standard library first, then third-party, then local
- Use type hints for function parameters and return types
- Use PEP8 formatting (4 spaces for indentation)
- Use meaningful variable names and docstrings
- Error handling: Use try/except blocks for chess and dataset operations
- Function organization: Group related functions together

## Project Structure
- Main training script: chess_grpo_clean.py
- Dataset utilities: convert_dset.py
- Models are saved to ./outputs directory

## Chess Reasoner Implementation Plan

### 1. Core Components
- Use Qwen-2.5-3B as the base model
- Setup chess engine (stockfish) for evaluation
- Use GRPO for training with simple reward functions

### 2. Dataset Approach
- Load chess games from Lichess dataset
- Sample random positions from games (not too early, not too late)
- Store positions as FEN strings

### 3. Reward Functions Design
- **Primary reward**: Evaluate model's move quality using stockfish
  - Simple approach: Compare eval before and after the move
  - Scale from 0.0 (bad move) to 1.0 (best move)
- **Format rewards**: Ensure output follows the XML format
  - <reasoning>...</reasoning>
  - <answer>move</answer>
- **Legality reward**: Check if move is legal in position

### 4. Implementation Simplicity Rules
- Keep the implementation as close to qwen_grpo.py as possible
- Only modify dataset loading and reward function
- Avoid complex data structures and nested lists
- Each dataset item will contain a single chess position
- Stockfish evaluation will be simple and fast