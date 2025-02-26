"""
Chess Reasoner GRPO Training
"""

import io
import logging
import os
import random
import re
import time
from datetime import datetime
from pathlib import Path

import chess
import chess.engine
import chess.pgn
import torch
import wandb
from datasets import Dataset, load_dataset
from tqdm import tqdm

# Can I push to HF
if os.environ.get("HF_TOKEN") is None:
    raise ValueError("HF_TOKEN not found! Please set")

# Logging
log_file = (
    Path(__file__).parent
    / "logs"
    / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)
log_file.parent.mkdir(exist_ok=True)
logging.basicConfig(
    filename=log_file,
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
MODEL = "Qwen/Qwen2.5-14B-Instruct"
LORA_RANK = 128
GPU_MEMORY_UTILIZATION = 0.95
CHECKPOINT_PATH = None  # No checkpoint: None

# Generation length settings
MAX_SEQ_LENGTH = 4096
MAX_PROMPT_LENGTH = 256
MAX_COMPLETION_LENGTH = MAX_SEQ_LENGTH - MAX_PROMPT_LENGTH

# Dataset settings
NUM_SAMPLES = 10_000

# Training settings
LEARNING_RATE = 5e-6
WEIGHT_DECAY = 0.1
WARMUP_RATIO = 0.1
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 1
NUM_GENERATIONS = 6
MAX_STEPS = 10_000
SAVE_STEPS = 1000

# Reward function weights
XML_COUNT_REWARD_WEIGHT = 0.05   # Has think tags
SOFT_FORMAT_REWARD_WEIGHT = 0.1  # Basic XML format
UCI_FORMAT_WEIGHT = 0.25         # Valid UCI notation
LEGAL_MOVE_WEIGHT = 0.5          # Legal move
MOVE_QUALITY_WEIGHT = 2.         # Good move quality

# Engine settings
ENGINE_ANALYSIS_TIME = 1.  # Time limit for engine analysis in seconds
# fmt: on
# =========================================

from unsloth import FastLanguageModel, PatchFastRL

PatchFastRL("GRPO", FastLanguageModel)
from unsloth import is_bfloat16_supported

wandb.init(project="chess-reasoner")
wandb.save(log_file, policy="live")

engine = chess.engine.SimpleEngine.popen_uci("stockfish")

SYSTEM_PROMPT = """
Given a chess position in FEN notation, analyze it and suggest the best move in UCI notation.

Respond in the following format:
<think>
Your analysis here.
</think>
Best move in UCI notation (e.g. b7b3) here.
"""

XML_FORMAT = """\
<think>
{reasoning}
</think>
{answer}
"""

# ======== GRPO REWARD FNS ========


def extract_answer(text: str) -> str:
    """Extract the answer (move) which comes after the </think> tag"""
    parts = text.split("</think>")
    if len(parts) == 1:
        return ""
    return "".join(parts[-1].split())


def is_valid_uci_format(move_str: str) -> bool:
    """Check if a string is in valid UCI move format (e.g., e2e4)"""
    try:
        chess.Move.from_uci(move_str)
        return True
    except:
        return False


def is_legal_move(move_str: str, board: chess.Board) -> bool:
    """Check if a move string is valid for the given board position"""
    try:
        move = chess.Move.from_uci(move_str)
        return move in board.legal_moves
    except:
        return False


def count_xml(text) -> float:
    """Count XML tags for partial reward"""
    count = 0
    if text.count("<think>\n") == 1:
        count += XML_COUNT_REWARD_WEIGHT / 2
    if text.count("\n</think>\n") == 1:
        count += XML_COUNT_REWARD_WEIGHT / 2
    return count


def xmlcount_reward(completions, **kwargs) -> list[float]:
    """Reward function for having correct XML tags"""
    contents = [completion[0]["content"] for completion in completions]
    rewards = [count_xml(c) for c in contents]

    global xml_structure_scores
    xml_structure_scores = rewards

    return rewards


def soft_format_reward(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has the correct format"""
    pattern = r"<think>.*?</think>\s*\S+"  # <think> tags followed by non-whitespace (the move)
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.search(pattern, r, re.DOTALL) is not None for r in responses]

    global format_results
    format_results = matches

    return [SOFT_FORMAT_REWARD_WEIGHT if match else 0.0 for match in matches]


def valid_uci_reward(completions, fen, **kwargs) -> list[float]:
    """Reward function that checks if the move is a valid UCI format"""
    responses = [completion[0]["content"] for completion in completions]
    extracted_moves = [extract_answer(r) for r in responses]

    rewards = []
    valid_uci_results = []

    for i, move in enumerate(extracted_moves):
        move = move.strip()
        board = chess.Board(fen[i])

        valid_uci = is_valid_uci_format(move)
        valid_uci_results.append(valid_uci)
        rewards.append(UCI_FORMAT_WEIGHT if valid_uci else 0.0)

    global valid_uci_checks
    valid_uci_checks = valid_uci_results

    return rewards


def legal_move_reward(completions, fen, **kwargs) -> list[float]:
    """Reward function that checks if the move is legal"""
    responses = [completion[0]["content"] for completion in completions]
    extracted_moves = [extract_answer(r) for r in responses]

    rewards = []
    legality_results = []

    for i, move in enumerate(extracted_moves):
        move = move.strip()
        board = chess.Board(fen[i])

        legal = is_legal_move(move, board)
        rewards.append(LEGAL_MOVE_WEIGHT if legal else 0.0)
        legality_results.append(legal)

    global legal_checks
    legal_checks = legality_results

    return rewards


def engine_analysis_reward(prompts, completions, fen, **kwargs) -> list[float]:
    """
    Reward based on how good the suggested move is according to the engine.
    Uses centipawn loss to evaluate move quality.
    This is the final reward function, so it's responsible for calling the logging function.
    """
    global format_results, xml_structure_scores, legal_checks, valid_uci_checks
    global initial_engine_scores, after_move_engine_scores, centipawn_losses, best_moves

    # Reset global arrays to store evaluation metrics for logging
    initial_engine_scores = []
    after_move_engine_scores = []
    centipawn_losses = []
    best_moves = []
    engine_time = 0.0

    responses = [completion[0]["content"] for completion in completions]
    extracted_moves = [extract_answer(r) for r in responses]

    move_rewards = []

    for i, move in enumerate(extracted_moves):
        move = move.strip()
        board = chess.Board(fen[i])

        # Skip evaluation for invalid moves
        if not move or not is_valid_uci_format(move) or not is_legal_move(move, board):
            move_rewards.append(0.0)
            initial_engine_scores.append(None)
            after_move_engine_scores.append(None)
            centipawn_losses.append(None)
            best_moves.append(None)
            continue

        start_time = time.perf_counter()

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

        engine_time += time.perf_counter() - start_time

    wandb.log({"train/engine_time": engine_time})
    log_generation_results(
        responses,
        extracted_moves,
        move_rewards,
        fen,
        initial_engine_scores,
        after_move_engine_scores,
        centipawn_losses,
        best_moves,
        engine_time,
    )

    return move_rewards


# =========================================


def log_generation_results(
    responses,
    extracted_moves,
    move_rewards,
    fen,
    initial_engine_scores,
    after_move_engine_scores,
    centipawn_losses,
    best_moves,
    engine_time,
):
    """
    Central logging function for generation results that is tqdm-compatible.
    Logs both to file and to console with tqdm-safe output.
    """
    for i in range(len(extracted_moves)):
        valid_format = i < len(format_results) and format_results[i]
        xml_score = xml_structure_scores[i] if i < len(xml_structure_scores) else 0.0
        legal = legal_checks[i] if i < len(legal_checks) else False
        valid_uci = valid_uci_checks[i] if i < len(valid_uci_checks) else False
        move = extracted_moves[i]
        quality = move_rewards[i] if i < len(move_rewards) else 0.0

        symbol_fmt = lambda b: "✓" if b else "✗"
        tqdm.write(
            f"Gen {i}: Move: {move or '-'} | "
            f"Format: {symbol_fmt(valid_format)} | "
            f"XML: {symbol_fmt(xml_score == XML_COUNT_REWARD_WEIGHT)} | "
            f"Valid UCI: {symbol_fmt(valid_uci)} | "
            f"Legal: {symbol_fmt(legal)} | "
            f"Quality: {quality:.2f} | "
            f"Engine Time: {engine_time:.2f}s"
        )

        # Log detailed information to the log file
        logging.info(f"\n==== GENERATION {i} COMPLETE SUMMARY ====")
        logging.info(f"BOARD POSITION: {fen[i]}")
        logging.info(f"RESPONSE:\n{responses[i]}")
        logging.info(f"EXTRACTED MOVE: '{move}'")
        logging.info(
            f"FORMAT CORRECT: {format_results[i] if i < len(format_results) else False}"
        )
        logging.info(f"XML STRUCTURE SCORE: {xml_score:.2f}")
        logging.info(f"VALID UCI FORMAT: {valid_uci}")
        logging.info(f"MOVE LEGAL: {legal}")
        logging.info(f"MOVE QUALITY: {quality:.2f}")
        logging.info(f"ENGINE TIME: {engine_time:.2f}")

        if move and legal:
            logging.info(f"INITIAL SCORE: {initial_engine_scores[i]}")
            logging.info(f"AFTER MOVE SCORE: {after_move_engine_scores[i]}")
            logging.info(f"CENTIPAWN LOSS: {centipawn_losses[i]}")
            logging.info(f"ENGINE'S BEST MOVE: {best_moves[i].uci()}")
        logging.info("=" * 40)


def get_random_position(row) -> tuple[str, chess.Board]:
    """Extract a random position from a chess game"""
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

    return board.fen()


def prepare_data_and_model():
    def format_dataset(row):
        """Format dataset for GRPO training"""
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Analyze this chess position and give the best move: {row['fen']}",
                },
            ],
            "fen": row["fen"],
        }

    dataset = load_dataset(
        "Icannos/lichess_games",
        streaming=True,
        trust_remote_code=True,
    )
    positions = []
    for _, row in tqdm(
        zip(range(NUM_SAMPLES), dataset["train"]),
        desc="Loading chess positions",
    ):
        positions.append(get_random_position(row))

    dataset = Dataset.from_dict({"fen": positions})
    dataset = dataset.map(format_dataset)

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

    return model, tokenizer, dataset


# Train the model using GRPO
def train_model(model, tokenizer, dataset):
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
        train_dataset=dataset,
    )

    print("Starting training...")
    trainer.train(resume_from_checkpoint=CHECKPOINT_PATH)

    # Save the trained model
    model.save_lora("chess_reasoner_llama_8b_lora")
    print("Model saved to chess_reasoner_llama_8b_lora")

    return model


def push_to_hub(model, tokenizer, repo_id):
    """Upload the trained model to Hugging Face Hub"""
    tqdm.write(f"Uploading model to Hugging Face Hub: {repo_id}")
    model.push_to_hub(repo_id)
    tokenizer.push_to_hub(repo_id)
    tqdm.write(f"Successfully uploaded model to: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    model, tokenizer, dataset = prepare_data_and_model()
    model = train_model(model, tokenizer, dataset)
    push_to_hub(model, tokenizer, "tommyp111/chess-reasoner")
    engine.quit()
    tqdm.write("Training complete!")
