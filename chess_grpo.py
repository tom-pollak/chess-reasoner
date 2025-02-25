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

# ======== CONFIGURATION PARAMETERS ========
# fmt: off
# Model settings
MODEL = "Qwen/Qwen2.5-3B-Instruct"
MAX_SEQ_LENGTH = 1024
LORA_RANK = 8
GPU_MEMORY_UTILIZATION = 0.8

# Dataset settings
NUM_SAMPLES = 1000

# Training settings
LEARNING_RATE = 5e-6
WEIGHT_DECAY = 0.1
WARMUP_RATIO = 0.1
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
NUM_GENERATIONS = 4
MAX_STEPS = 500
SAVE_STEPS = 100

# Generation length settings
MAX_PROMPT_LENGTH = 256
MAX_COMPLETION_LENGTH = 512

# Reward function weights
FORMAT_REWARD_WEIGHT = 0.1  # Basic XML format
UCI_FORMAT_WEIGHT = 0.2     # Valid UCI notation
LEGAL_MOVE_WEIGHT = 0.4     # Legal move
MOVE_QUALITY_WEIGHT = 0.7   # Good move quality

# Engine settings
ENGINE_ANALYSIS_TIME = 0.1  # Time limit for engine analysis in seconds
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
    dataset = load_dataset("Icannos/lichess_games", streaming=True)
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


def evaluate_move(board: chess.Board, move_str: str) -> float:
    """
    Evaluate a chess move using Stockfish engine by comparing position evaluation
    before and after the move. Returns a score between 0 and 1.
    """
    try:
        move = chess.Move.from_uci(move_str)
        if move not in board.legal_moves:
            return 0.0

        initial_result = engine.analyse(
            board, chess.engine.Limit(time=ENGINE_ANALYSIS_TIME)
        )
        best_move = initial_result["pv"][0]

        best_board = board.copy()
        best_board.push(best_move)
        best_eval = engine.analyse(
            best_board, chess.engine.Limit(time=ENGINE_ANALYSIS_TIME)
        )
        best_score = best_eval["score"].relative.score(mate_score=10000)

        player_board = board.copy()
        player_board.push(move)
        player_eval = engine.analyse(
            player_board, chess.engine.Limit(time=ENGINE_ANALYSIS_TIME)
        )
        player_score = player_eval["score"].relative.score(mate_score=10000)

        eval_diff = player_score - best_score

        # Simple linear scaling:
        # 0 or positive diff = best move (1.0)
        # -100 centipawns = decent move (0.5)
        # -300 centipawns or worse = bad move (0.0)
        if eval_diff >= 0:
            return 1.0
        elif eval_diff < -300:
            return 0.0
        else:
            return 1.0 + (eval_diff / 300.0)

    except Exception:
        return 0.0


## GRPO Reward Fns


def move_correctness_reward(prompts, completions, board_fen, **kwargs) -> List[float]:
    """
    Reward based on how good the suggested move is according to the engine.
    Uses position evaluation difference before and after the move.
    """
    global format_results, xml_structure_scores, legal_checks, valid_uci_checks

    responses = [completion[0]["content"] for completion in completions]
    extracted_moves = [extract_answer(r) for r in responses]

    rewards = []
    move_quality_scores = []

    for i, move in enumerate(extracted_moves):
        move = move.strip()
        board = chess.Board(board_fen[i])
        reward = evaluate_move(board, move)
        move_quality_scores.append(reward)
        weighted_reward = MOVE_QUALITY_WEIGHT * reward
        rewards.append(weighted_reward)
        wandb.log({"move_reward": reward})

    print("\n--- Generation Summary ---")
    for i in range(len(extracted_moves)):
        valid_format = i < len(format_results) and format_results[i]
        xml_score = xml_structure_scores[i] if i < len(xml_structure_scores) else 0.0
        legal = legal_checks[i] if i < len(legal_checks) else False
        valid_uci = valid_uci_checks[i] if i < len(valid_uci_checks) else False
        move = extracted_moves[i]
        quality = move_quality_scores[i]

        symbol_fmt = lambda b: "✓" if b else "✗"
        print(
            f"Gen {i}: Move: {move or '-'} | "
            f"Format: {symbol_fmt(valid_format)} | "
            f"XML: {xml_score:.2f} | "
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

        if move and legal:
            board = chess.Board(board_fen[i])
            analysis = engine.analyse(
                board, chess.engine.Limit(time=ENGINE_ANALYSIS_TIME)
            )
            best_move = analysis["pv"][0].uci()
            logging.info(f"BOARD POSITION: {board.fen()}")
            logging.info(f"ENGINE'S BEST MOVE: {best_move}")
        logging.info("=" * 40)

    wandb.log(
        {
            "avg_quality": sum(move_quality_scores) / max(1, len(move_quality_scores)),
            "avg_xml_score": sum(xml_structure_scores)
            / max(1, len(xml_structure_scores)),
            "format_correct_pct": sum(1 for f in format_results if f)
            / max(1, len(format_results)),
            "valid_uci_pct": sum(1 for u in valid_uci_checks if u)
            / max(1, len(valid_uci_checks)),
            "legal_moves_pct": sum(1 for l in legal_checks if l)
            / max(1, len(legal_checks)),
        }
    )

    return rewards


def legal_move_reward(completions, board_fen, **kwargs) -> List[float]:
    """Reward function that checks if the move is legal and/or valid UCI format"""
    responses = [completion[0]["content"] for completion in completions]
    extracted_moves = [extract_answer(r) for r in responses]

    rewards = []
    legality_results = []
    valid_uci_results = []

    for i, move in enumerate(extracted_moves):
        move = move.strip()
        board = chess.Board(board_fen[i])

        # Check if move is valid UCI format
        valid_uci = is_valid_uci_format(move)
        valid_uci_results.append(valid_uci)

        # Check if move is legal in this position
        legal = is_valid_move(move, board)
        legality_results.append(legal)

        # Log metrics
        wandb.log({"legal_move": 1 if legal else 0})
        wandb.log({"valid_uci_format": 1 if valid_uci else 0})

        # Give full reward for legal moves, partial reward for valid UCI format
        if legal:
            reward = LEGAL_MOVE_WEIGHT
        elif valid_uci:
            reward = UCI_FORMAT_WEIGHT
        else:
            reward = 0.0

        rewards.append(reward)

    global legal_checks, valid_uci_checks
    legal_checks = legality_results
    valid_uci_checks = valid_uci_results

    return rewards


def soft_format_reward_func(completions, **kwargs) -> List[float]:
    """Reward function that checks if the completion has the correct format"""
    pattern = r"<think>.*?</think>\s*\S+"  # <think> tags followed by non-whitespace (the move)
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.search(pattern, r, re.DOTALL) is not None for r in responses]

    for match in matches:
        wandb.log({"format_correct": 1 if match else 0})

    global format_results
    format_results = matches

    return [FORMAT_REWARD_WEIGHT if match else 0.0 for match in matches]


def count_xml(text) -> float:
    """Count XML tags for partial reward"""
    count = 0.0
    if "<think>" in text:
        count += 0.15
    if "</think>" in text:
        count += 0.15
    # Check if there's content after </think>
    if "</think>" in text and text.split("</think>")[-1].strip():
        count += 0.2
    return count


def xmlcount_reward_func(completions, **kwargs) -> List[float]:
    """Reward function for having correct XML tags"""
    contents = [completion[0]["content"] for completion in completions]
    rewards = [count_xml(c) for c in contents]

    for reward in rewards:
        wandb.log({"xml_structure_score": reward})

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
            xmlcount_reward_func,
            soft_format_reward_func,
            legal_move_reward,
            move_correctness_reward,
        ],
        args=training_args,
        train_dataset=train_dataset,
    )

    print("Starting training...")
    try:
        trainer.train()
    except Exception as e:
        print(f"Error during training: {e}")
        raise e

    # Save the trained model
    model.save_lora("chess_reasoner_llama_8b_lora")
    print("Model saved to chess_reasoner_llama_8b_lora")

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

        # Generation parameters
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=0.5,  # Lower temperature for more deterministic outputs
            top_p=0.95,
            max_tokens=MAX_COMPLETION_LENGTH,  # Match the training completion length
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

        lora_request = model.load_lora("chess_reasoner_llama_8b_lora")

        # Generate with our trained LoRA
        output_lora = (
            model.fast_generate(
                [text],
                sampling_params=sampling_params,
                lora_request=lora_request,
            )[0]
            .outputs[0]
            .text
        )

        # Print results
        print("\nBase model response:")
        print(output_base)

        print("\nFine-tuned model response:")
        print(output_lora)

        # Evaluate both models
        board = chess.Board(fen)
        base_move = extract_answer(output_base)
        lora_move = extract_answer(output_lora)

        # Check format correctness
        base_format = (
            re.search(
                r"<think>.*?</think>\s*\S+",
                output_base,
                re.DOTALL,
            )
            is not None
        )
        lora_format = (
            re.search(
                r"<think>.*?</think>\s*\S+",
                output_lora,
                re.DOTALL,
            )
            is not None
        )

        print(f"\nBase model format correct: {base_format}")
        print(f"Fine-tuned model format correct: {lora_format}")

        # Check move correctness
        print(f"\nBase model move: {base_move}")
        if base_move:
            base_legal = is_valid_move(base_move, board)
            print(f"Base model move legal: {base_legal}")
            if base_legal:
                base_score = evaluate_move(board, base_move)
                print(f"Base model score: {base_score:.2f}")

        print(f"Fine-tuned model move: {lora_move}")
        if lora_move:
            lora_legal = is_valid_move(lora_move, board)
            print(f"Fine-tuned model move legal: {lora_legal}")
            if lora_legal:
                lora_score = evaluate_move(board, lora_move)
                print(f"Fine-tuned model score: {lora_score:.2f}")

        # Get top engine move for comparison
        if engine is not None:
            analysis = engine.analyse(
                board, chess.engine.Limit(time=ENGINE_ANALYSIS_TIME)
            )
            best_move = analysis["pv"][0].uci()
            print(f"Engine's best move: {best_move}")


if __name__ == "__main__":
    try:
        model, tokenizer, train_dataset = prepare_data_and_model()
        model = train_model(model, tokenizer, train_dataset)
        test_model(model, tokenizer)
        print("Training completed successfully!")
    except Exception as e:
        print(f"Error during training: {e}")
    finally:
        # Clean up the chess engine
        if engine:
            engine.quit()
