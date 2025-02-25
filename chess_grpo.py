"""
Chess Reasoner GRPO Training

This script trains a Qwen 2.5 3B model using GRPO to reason about chess positions
and provide high-quality moves. The model is trained to:
1. Generate reasoning about the chess position in <reasoning> tags
2. Provide the best move in <answer> tags
3. Get verified by a chess engine to determine the quality of the move

The training uses Unsloth for efficient fine-tuning and stockfish for evaluation.
"""

import io
import os
import random
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import chess
import chess.engine
import chess.pgn
import torch
import wandb
from datasets import Dataset, load_dataset

# Initialize wandb
wandb.init(project="chess-reasoner", name="qwen-2.5-3b-chess-grpo")

# Setup Unsloth with GRPO
from unsloth import FastLanguageModel, PatchFastRL

PatchFastRL("GRPO", FastLanguageModel)

# Check if we have a GPU with bfloat16 support
from unsloth import is_bfloat16_supported


# Setup Stockfish engine (install stockfish first!)
# For macOS: brew install stockfish
# For Ubuntu: apt-get install stockfish
def setup_engine() -> chess.engine.SimpleEngine:
    try:
        # Try to find stockfish in PATH
        engine = chess.engine.SimpleEngine.popen_uci("stockfish")
        return engine
    except FileNotFoundError:
        # Try common paths if not in PATH
        common_paths = [
            "/usr/local/bin/stockfish",
            "/usr/bin/stockfish",
            "/opt/homebrew/bin/stockfish",
        ]
        for path in common_paths:
            if os.path.exists(path):
                return chess.engine.SimpleEngine.popen_uci(path)

        raise FileNotFoundError(
            "Stockfish not found. Please install stockfish and make sure it's in your PATH."
        )


# Initialize chess engine
try:
    engine = setup_engine()
    print("Chess engine initialized successfully!")
except Exception as e:
    print(f"Warning: Could not initialize chess engine: {e}")
    print("Training will continue, but move evaluation will be random.")
    engine = None

# Constants and system prompt
SYSTEM_PROMPT = """
You are a chess expert. Given a chess position in FEN notation, you will analyze the position and suggest the best move.

Respond in the following format:
<reasoning>
[Your step-by-step analysis of the position, considering tactics, strategy, and possible continuations]
</reasoning>
<answer>
[Your chosen move in algebraic notation, e.g., e2e4, g8f6, etc.]
</answer>
"""

XML_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""


# Extract answers from model responses
def extract_xml_answer(text: str) -> str:
    if "<answer>" not in text or "</answer>" not in text:
        return ""
    try:
        answer = text.split("<answer>")[-1]
        answer = answer.split("</answer>")[0]
        return answer.strip()
    except:
        return ""


# Function to get random positions from chess games
def get_random_position(row) -> Tuple[str, chess.Board]:
    try:
        pgn = io.StringIO(row["text"])
        game = chess.pgn.read_game(pgn)
        if not game:
            # Default to starting position if game parsing fails
            return chess.STARTING_FEN, chess.Board()

        board = game.board()
        mainline_moves = list(game.mainline_moves())
        if not mainline_moves:
            return chess.STARTING_FEN, board

        # Choose a random point in the game (not too early, not too late)
        min_move = min(5, len(mainline_moves) // 5)
        max_move = max(min_move + 1, len(mainline_moves) - 5)
        if max_move <= min_move:
            max_move = min(len(mainline_moves), min_move + 10)

        # Apply moves up to the random point
        move_count = random.randint(min_move, max_move)
        for move in mainline_moves[:move_count]:
            board.push(move)

        return board.fen(), board
    except Exception as e:
        print(f"Error processing game: {e}")
        return chess.STARTING_FEN, chess.Board()


# Prepare the Lichess dataset with random positions
def prepare_chess_dataset(num_samples: int = 1000) -> Dataset:
    dataset = load_dataset("Icannos/lichess_games", streaming=True)

    positions = []
    boards = []

    # Get random positions from games
    count = 0
    for row in dataset["train"]:
        if count >= num_samples:
            break
        try:
            fen, board = get_random_position(row)
            positions.append(fen)
            boards.append(board)
            count += 1
            if count % 100 == 0:
                print(f"Processed {count} games...")
        except Exception as e:
            print(f"Error in game {count}: {e}")

    # Create Dataset with FEN positions
    data_dict = {
        "fen": positions,
        "board_state": boards,
    }

    return Dataset.from_dict(data_dict)


# Function to evaluate a chess move using the engine
def evaluate_move(board: chess.Board, move_str: str, time_limit: float = 0.1) -> float:
    """
    Evaluate a chess move using Stockfish engine
    Returns a score between 0 and 1, where 1 is the best move
    """
    if engine is None:
        # If no engine, return random score as fallback
        return random.random()

    try:
        # Parse the move string (e.g., "e2e4")
        move = chess.Move.from_uci(move_str)

        # Check if move is legal
        if move not in board.legal_moves:
            return 0.0

        # Get top 3 moves from engine
        result = engine.analyse(board, chess.engine.Limit(time=time_limit), multipv=3)
        best_moves = [entry["pv"][0] for entry in result]

        # If the move matches the best engine move
        if move == best_moves[0]:
            return 1.0
        # If it's the second best move
        elif len(best_moves) > 1 and move == best_moves[1]:
            return 0.7
        # If it's the third best move
        elif len(best_moves) > 2 and move == best_moves[2]:
            return 0.4

        # If move is legal but not in top 3
        return 0.1
    except Exception as e:
        print(f"Error evaluating move: {e}")
        return 0.0


# Reward functions for GRPO
def move_correctness_reward(prompts, completions, board_state, **kwargs) -> List[float]:
    """Reward based on how good the suggested move is according to the engine"""
    responses = [completion[0]["content"] for completion in completions]
    extracted_moves = [extract_xml_answer(r) for r in responses]

    rewards = []
    for move, board in zip(extracted_moves, board_state):
        # Clean up the move string (remove extra spaces, etc.)
        move = move.strip()

        # Evaluate the move
        reward = evaluate_move(board, move)
        rewards.append(reward)

    # Log a sample for debugging
    if len(rewards) > 0:
        sample_idx = random.randint(0, len(rewards) - 1)
        fen = prompts[sample_idx][-1]["content"]
        response = responses[sample_idx]
        move = extracted_moves[sample_idx]
        reward = rewards[sample_idx]
        print(f"\nPosition: {fen}")
        print(f"Response:\n{response}")
        print(f"Move: {move}, Reward: {reward}")

    return rewards


def is_valid_move(move_str: str, board: chess.Board) -> bool:
    """Check if a move string is valid for the given board position"""
    try:
        move = chess.Move.from_uci(move_str)
        return move in board.legal_moves
    except:
        return False


def legal_move_reward(completions, board_state, **kwargs) -> List[float]:
    """Reward function that checks if the move is legal"""
    responses = [completion[0]["content"] for completion in completions]
    extracted_moves = [extract_xml_answer(r) for r in responses]

    return [
        0.2 if is_valid_move(move, board) else 0.0
        for move, board in zip(extracted_moves, board_state)
    ]


def format_reward_func(completions, **kwargs) -> List[float]:
    """Reward function that checks if the completion has the correct format"""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.search(pattern, r, re.DOTALL) is not None for r in responses]
    return [0.3 if match else 0.0 for match in matches]


def count_xml(text) -> float:
    """Count XML tags for partial reward"""
    count = 0.0
    if "<reasoning>" in text:
        count += 0.1
    if "</reasoning>" in text:
        count += 0.1
    if "<answer>" in text:
        count += 0.1
    if "</answer>" in text:
        count += 0.1
    return count


def xmlcount_reward_func(completions, **kwargs) -> List[float]:
    """Reward function for having correct XML tags"""
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]


# Main function to prepare dataset and model
def prepare_data_and_model():
    print("Preparing dataset...")
    # Load a smaller dataset for testing, increase for full training
    dataset = prepare_chess_dataset(num_samples=200)

    # Prepare the dataset for training with prompts
    def format_dataset(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Analyze this chess position and give the best move: {example['fen']}",
                },
            ],
            "board_state": example["board_state"],
        }

    train_dataset = dataset.map(format_dataset)

    print("Setting up model...")
    # Model parameters
    max_seq_length = 1024
    lora_rank = 64

    # Load the Qwen 2.5 3B model with LoRA
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        fast_inference=True,
        max_lora_rank=lora_rank,
        gpu_memory_utilization=0.8,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=lora_rank,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    return model, tokenizer, train_dataset


# Train the model using GRPO
def train_model(model, tokenizer, train_dataset):
    from trl import GRPOConfig, GRPOTrainer

    training_args = GRPOConfig(
        use_vllm=True,
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        logging_steps=1,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=8,
        max_prompt_length=256,
        max_completion_length=200,
        num_train_epochs=3,
        save_steps=100,
        max_grad_norm=0.1,
        report_to="wandb",  # Report to wandb
        output_dir="outputs",
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            xmlcount_reward_func,
            format_reward_func,
            legal_move_reward,
            move_correctness_reward,
        ],
        args=training_args,
        train_dataset=train_dataset,
    )

    print("Starting training...")
    trainer.train()

    # Save the trained model
    model.save_lora("chess_reasoner_lora")
    print("Model saved to chess_reasoner_lora")

    return model


# Test the model on some example positions
def test_model(model, tokenizer):
    # Test positions
    test_positions = [
        chess.STARTING_FEN,  # Starting position
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",  # Common position after 1.e4 e5 2.Nf3 Nc6
        "rnbqkb1r/pp2pppp/3p1n2/2p5/4P3/2N2N2/PPPP1PPP/R1BQKB1R w KQkq - 0 4",  # Sicilian Defense
    ]

    print("\nTesting model on example positions...")

    for fen in test_positions:
        prompt = f"Analyze this chess position and give the best move: {fen}"
        print(f"\nPosition: {fen}")

        # Format with chat template
        text = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

        # Generate with our trained LoRA
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.95,
            max_tokens=1024,
        )

        # Generate without LoRA
        output_base = (
            model.fast_generate(
                [text],
                sampling_params=sampling_params,
                lora_request=None,
            )[0]
            .outputs[0]
            .text
        )

        # Generate with our trained LoRA
        output_lora = (
            model.fast_generate(
                [text],
                sampling_params=sampling_params,
                lora_request=model.load_lora("chess_reasoner_lora"),
            )[0]
            .outputs[0]
            .text
        )

        # Print results
        print("\nBase model response:")
        print(output_base)

        print("\nFine-tuned model response:")
        print(output_lora)

        # Evaluate the suggested move
        try:
            board = chess.Board(fen)
            base_move = extract_xml_answer(output_base)
            lora_move = extract_xml_answer(output_lora)

            print(f"\nBase model move: {base_move}")
            if base_move:
                base_score = evaluate_move(board, base_move)
                print(f"Base model score: {base_score:.2f}")

            print(f"Fine-tuned model move: {lora_move}")
            if lora_move:
                lora_score = evaluate_move(board, lora_move)
                print(f"Fine-tuned model score: {lora_score:.2f}")
        except Exception as e:
            print(f"Error evaluating moves: {e}")


# Main execution
if __name__ == "__main__":
    try:
        model, tokenizer, train_dataset = prepare_data_and_model()
        model = train_model(model, tokenizer, train_dataset)
        test_model(model, tokenizer)

        # Clean up the chess engine
        if engine:
            engine.quit()

        print("Training completed successfully!")
    except Exception as e:
        print(f"Error during training: {e}")
        # Clean up the chess engine if there was an error
        if engine:
            engine.quit()
