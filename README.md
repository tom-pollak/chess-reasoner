# Chess Reasoner

Train LLMs to reason on chess positions with GRPO.

Anything that we can verify the output of can now be trained on with GRPO. We can use Stockfish evaluation of the position before and after an LLM has made a move as a reward signal.

- I use [Unsloth](https://github.com/unsloth/unsloth) for efficient fine-tuning and inspired by their GRPO notebook


## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/chess-reasoner.git
cd chess-reasoner

# Install dependencies
uv sync
uv pip install -r requirements.txt
```

## Training

```bash
python chess_grpo.py
```

1. Samples random positions from chess games
2. Model analyzes position and suggests a move
3. Evaluates move quality using Stockfish
4. Computes rewards based on move quality and formatting
5. Updates the model using GRPO
