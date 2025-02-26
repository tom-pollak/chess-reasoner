"""
Chess Reasoner GRPO Training (Clean Implementation)

This script trains a model using GRPO to reason about chess positions
and provide high-quality moves. The model is trained to:
1. Generate reasoning about the chess position in <reasoning> tags
2. Provide the best move in <answer> tags
3. Get verified by a chess engine to determine the quality of the move

The training uses Unsloth for efficient fine-tuning and stockfish for evaluation.
"""

import io
import logging
import os
import random
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import chess
import chess.engine
import chess.pgn
import torch
import wandb
from datasets import Dataset, load_dataset

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(
        log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    ),
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
)

# Storage for cross-function metrics
format_results = []
xml_structure_scores = []
legal_checks = []
valid_uci_checks = []
initial_engine_scores = []
after_move_engine_scores = []
centipawn_losses = []
best_moves = []

# ======== CONFIGURATION PARAMETERS ========
# fmt: off
# Model settings
MODEL = "Qwen/Qwen2.5-3B-Instruct"
MAX_SEQ_LENGTH = 1024
LORA_RANK = 8
GPU_MEMORY_UTILIZATION = 0.95
CHECKPOINT_PATH = "outputs/checkpoint-5000"  # No checkpoint: None

# Dataset settings
NUM_SAMPLES = 10_000

# Training settings
LEARNING_RATE = 5e-6
WEIGHT_DECAY = 0.1
WARMUP_RATIO = 0.1
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 1
NUM_GENERATIONS = 8
MAX_STEPS = 10_000
SAVE_STEPS = 1000

# Generation length settings
MAX_PROMPT_LENGTH = 256
MAX_COMPLETION_LENGTH = 768

# Reward function weights
XML_COUNT_REWARD_WEIGHT = 0.05   # Has think tags
SOFT_FORMAT_REWARD_WEIGHT = 0.1  # Basic XML format
UCI_FORMAT_WEIGHT = 0.25         # Valid UCI notation
LEGAL_MOVE_WEIGHT = 0.5          # Legal move
MOVE_QUALITY_WEIGHT = 1.         # Good move quality

# Engine settings
ENGINE_ANALYSIS_TIME = 0.5  # Time limit for engine analysis in seconds
# fmt: on
# =========================================

from unsloth import FastLanguageModel, PatchFastRL

PatchFastRL("GRPO", FastLanguageModel)
from unsloth import is_bfloat16_supported

wandb.init(project="chess-reasoner", name=f"{MODEL.split('/')[-1]}-chess-grpo")


def setup_engine() -> chess.engine.SimpleEngine:
    try:
        return chess.engine.SimpleEngine.popen_uci("stockfish")
    except FileNotFoundError:
        raise FileNotFoundError(
            "Stockfish not found. Please install stockfish and make sure it's in your PATH."
        )


engine = setup_engine()
print("Chess engine initialized successfully!")

SYSTEM_PROMPT = """
You are a chess expert. Given a chess position in FEN notation, analyze it and suggest the best move.

Respond in the following format:
<think>
Your analysis here. Keep it brief.
</think>
e2e4

First your analysis in <think> tags, then just your move in UCI notation (like e2e4) after the </think> tag.
"""

XML_FORMAT = """\
<think>
{reasoning}
</think>
{answer}
"""


def extract_answer(text: str) -> str:
    """Extract the answer (move) which comes after the </think> tag"""
    try:
        parts = text.split("</think>")
        if len(parts) == 1:
            return ""
        return "".join(parts[-1].split())
    except Exception as e:
        print(f"Error extracting answer: {e}")
        return ""


def get_random_position(row) -> Tuple[str, chess.Board]:
    """Extract a random position from a chess game"""
    try:
        pgn = io.StringIO(row["text"])
        game = chess.pgn.read_game(pgn)
        if not game:
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


def prepare_chess_dataset() -> Dataset:
    """
    Create a dataset of chess positions from games

    Gets a random position

    """
    dataset = load_dataset(
        "Icannos/lichess_games", streaming=True, trust_remote_code=True
    )
    positions = []
    count = 0
    for row in dataset["train"]:
        if count >= NUM_SAMPLES:
            break
        try:
            fen, _ = get_random_position(row)
            positions.append(fen)
            count += 1
            if count % 100 == 0:
                print(f"Processed {count} positions...")
        except Exception as e:
            print(f"Error in game {count}: {e}")

    return Dataset.from_dict({"fen": positions})


def is_valid_uci_format(move_str: str) -> bool:
    """Check if a string is in valid UCI move format (e.g., e2e4)"""
    try:
        chess.Move.from_uci(move_str)
        return True
    except:
        return False


def is_valid_move(move_str: str, board: chess.Board) -> bool:
    """Check if a move string is valid for the given board position"""
    try:
        move = chess.Move.from_uci(move_str)
        return move in board.legal_moves
    except:
        return False


## GRPO Reward Functions


def engine_analysis_reward(prompts, completions, board_fen, **kwargs) -> List[float]:
    """
    Reward based on how good the suggested move is according to the engine.
    Uses centipawn loss to evaluate move quality.
    """
    global format_results, xml_structure_scores, legal_checks, valid_uci_checks

    # Create arrays to store evaluation metrics for logging
    initial_engine_scores = []
    after_move_engine_scores = []
    centipawn_losses = []
    best_moves = []

    responses = [completion[0]["content"] for completion in completions]
    extracted_moves = [extract_answer(r) for r in responses]

    move_rewards = []

    for i, move in enumerate(extracted_moves):
        move = move.strip()
        board = chess.Board(board_fen[i])

        # Skip evaluation for invalid moves
        if not move or not is_valid_uci_format(move) or not is_valid_move(move, board):
            move_rewards.append(0.0)
            initial_engine_scores.append(None)
            after_move_engine_scores.append(None)
            centipawn_losses.append(None)
            best_moves.append(None)
            continue

        # Engine analysis of current position
        initial_eval = engine.analyse(
            board, chess.engine.Limit(time=ENGINE_ANALYSIS_TIME)
        )
        best_move = initial_eval["pv"][0]
        initial_score = initial_eval["score"].relative.score(mate_score=10000)

        # Make player's move and get new evaluation
        player_move = chess.Move.from_uci(move)
        board.push(player_move)
        player_eval = engine.analyse(
            board, chess.engine.Limit(time=ENGINE_ANALYSIS_TIME)
        )

        # Negate because it's from opponent's perspective
        after_move_score = -player_eval["score"].relative.score(mate_score=10000)

        # Calculate centipawn loss
        centipawn_loss = initial_score - after_move_score

        initial_engine_scores.append(initial_score)
        after_move_engine_scores.append(after_move_score)
        centipawn_losses.append(centipawn_loss)
        best_moves.append(best_move)

        # Reward scaling
        # - Less than 300 (bishop / rook blunder) is 0.0
        # - Best move is 1.0
        reward = 0.0
        if centipawn_loss <= 0:
            reward = 1.0
        elif centipawn_loss >= 300:
            reward = 0.0
        else:
            reward = 1.0 - (centipawn_loss / 300.0)

        move_rewards.append(reward * MOVE_QUALITY_WEIGHT)

    print("\n--- Generation Summary ---")
    for i in range(len(extracted_moves)):
        valid_format = i < len(format_results) and format_results[i]
        xml_score = xml_structure_scores[i] if i < len(xml_structure_scores) else 0.0
        legal = legal_checks[i] if i < len(legal_checks) else False
        valid_uci = valid_uci_checks[i] if i < len(valid_uci_checks) else False
        move = extracted_moves[i]
        quality = move_rewards[i]

        symbol_fmt = lambda b: "✓" if b else "✗"
        print(
            f"Gen {i}: Move: {move or '-'} | "
            f"Format: {symbol_fmt(valid_format)} | "
            f"XML: {symbol_fmt(xml_score == XML_COUNT_REWARD_WEIGHT)} | "
            f"Valid UCI: {symbol_fmt(valid_uci)} | "
            f"Legal: {symbol_fmt(legal)} | "
            f"Quality: {quality:.2f}"
        )

        logging.info(f"\n==== GENERATION {i} COMPLETE SUMMARY ====")
        logging.info(f"RESPONSE:\n{responses[i]}")
        logging.info(f"EXTRACTED MOVE: '{move}'")
        logging.info(
            f"FORMAT CORRECT: {format_results[i] if i < len(format_results) else False}"
        )
        logging.info(f"XML STRUCTURE SCORE: {xml_score:.2f}")
        logging.info(f"VALID UCI FORMAT: {valid_uci}")
        logging.info(f"MOVE LEGAL: {legal}")
        logging.info(f"MOVE QUALITY: {quality:.2f}")

        if move and legal and i < len(initial_engine_scores) and initial_engine_scores[i] is not None:
            logging.info(f"INITIAL SCORE: {initial_engine_scores[i]}")
            logging.info(f"AFTER MOVE SCORE: {after_move_engine_scores[i]}")
            logging.info(f"CENTIPAWN LOSS: {centipawn_losses[i]}")
            logging.info(f"BOARD POSITION: {board_fen[i]}")
            if best_moves[i]:
                logging.info(f"ENGINE'S BEST MOVE: {best_moves[i].uci()}")
        logging.info("=" * 40)

    return move_rewards


def legal_move_reward(completions, board_fen, **kwargs) -> List[float]:
    """Reward function that checks if the move is legal"""
    responses = [completion[0]["content"] for completion in completions]
    extracted_moves = [extract_answer(r) for r in responses]

    rewards = []
    legality_results = []

    for i, move in enumerate(extracted_moves):
        move = move.strip()
        board = chess.Board(board_fen[i])

        legal = is_valid_move(move, board)
        rewards.append(LEGAL_MOVE_WEIGHT if legal else 0.0)
        legality_results.append(legal)

    global legal_checks
    legal_checks = legality_results

    return rewards


def valid_uci_reward(completions, board_fen, **kwargs) -> List[float]:
    """Reward function that checks if the move is a valid UCI format"""
    responses = [completion[0]["content"] for completion in completions]
    extracted_moves = [extract_answer(r) for r in responses]

    rewards = []
    valid_uci_results = []

    for i, move in enumerate(extracted_moves):
        move = move.strip()
        board = chess.Board(board_fen[i])

        valid_uci = is_valid_uci_format(move)
        valid_uci_results.append(valid_uci)
        rewards.append(UCI_FORMAT_WEIGHT if valid_uci else 0.0)

    global valid_uci_checks
    valid_uci_checks = valid_uci_results

    return rewards


def soft_format_reward(completions, **kwargs) -> List[float]:
    """Reward function that checks if the completion has the correct format"""
    pattern = r"<think>.*?</think>\s*\S+"  # <think> tags followed by non-whitespace (the move)
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.search(pattern, r, re.DOTALL) is not None for r in responses]

    global format_results
    format_results = matches

    return [SOFT_FORMAT_REWARD_WEIGHT if match else 0.0 for match in matches]


def count_xml(text) -> float:
    """Count XML tags for partial reward"""
    count = 0
    if "<think>" in text:
        count += XML_COUNT_REWARD_WEIGHT / 2
    if "</think>" in text:
        count += XML_COUNT_REWARD_WEIGHT / 2
    return count


def xmlcount_reward(completions, **kwargs) -> List[float]:
    """Reward function for having correct XML tags"""
    contents = [completion[0]["content"] for completion in completions]
    rewards = [count_xml(c) for c in contents]

    global xml_structure_scores
    xml_structure_scores = rewards

    return rewards


# Main function to prepare dataset and model
def prepare_data_and_model():
    print("Preparing dataset...")
    dataset = prepare_chess_dataset()

    def format_dataset(example):
        """Format dataset for GRPO training"""
        # Store FEN String
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Analyze this chess position and give the best move: {example['fen']}",
                },
            ],
            "board_fen": example["fen"],
        }

    train_dataset = dataset.map(format_dataset)
    print("Dataset prepared successfully!")

    print("Setting up model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        fast_inference=True,
        max_lora_rank=LORA_RANK,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=LORA_RANK,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    return model, tokenizer, train_dataset


# Train the model using GRPO
def train_model(model, tokenizer, train_dataset):
    from trl import GRPOConfig, GRPOTrainer

    training_args = GRPOConfig(
        use_vllm=True,
        learning_rate=LEARNING_RATE,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        logging_steps=1,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_generations=NUM_GENERATIONS,
        max_prompt_length=MAX_PROMPT_LENGTH,
        max_completion_length=MAX_COMPLETION_LENGTH,
        max_steps=MAX_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        max_grad_norm=0.1,
        report_to="wandb",
        output_dir="outputs",
        run_name="chess-reasoner-training",
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            xmlcount_reward,
            soft_format_reward,
            legal_move_reward,
            valid_uci_reward,
            engine_analysis_reward,
        ],
        args=training_args,
        train_dataset=train_dataset,
    )

    print("Starting training...")
    trainer.train(resume_from_checkpoint=CHECKPOINT_PATH)

    # Save the trained model
    model.save_lora("chess_reasoner_llama_8b_lora")
    print("Model saved to chess_reasoner_llama_8b_lora")

    return model


def push_to_hub(model, tokenizer, repo_id):
    """Upload the trained model to Hugging Face Hub"""
    print(f"Uploading model to Hugging Face Hub: {repo_id}")
    model.push_to_hub(repo_id)
    tokenizer.push_to_hub(repo_id)
    print(f"Successfully uploaded model to: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    from huggingface_hub import login

    login()

    model, tokenizer, train_dataset = prepare_data_and_model()
    model = train_model(model, tokenizer, train_dataset)
    push_to_hub(model, tokenizer, "tommyp111/chess-reasoner")
    engine.quit()
