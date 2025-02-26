# Chess Reasoner Project Guidelines

This file contains essential information and commands for the Chess Reasoner project.

## Quick Reference

- Project purpose: Train LLMs to reason about chess positions using GRPO with stockfish evaluation
- Main file: `chess_grpo.py` - Contains training script and configuration parameters

## Setup & Commands

- Install dependencies: `uv sync`
- Install dev dependencies: `uv pip install -r requirements.txt`
- Run training: `python chess_grpo.py`

## Requirements

- Stockfish chess engine must be installed and accessible in PATH
- For best performance, use a CUDA-capable GPU with >= 12GB VRAM
- Wandb account for tracking training metrics

## Key Project Info

- Training uses Lichess games dataset for positions
- Engine analysis time is configurable (default: 4 seconds per position)
- Model formats responses with reasoning in `<think>` tags followed by UCI move
- Reward is weighted combination of formatting, move validity, and centipawn loss

## Code Style Guidelines

- Follow PEP 8 conventions
- Use type hints where possible
- Document complex functions with docstrings
- Use logging for tracking training progress