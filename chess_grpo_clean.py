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
import os
import random
import re
from typing import List, Dict, Any, Optional, Tuple, Union

import chess
import chess.engine
import chess.pgn
import torch
import wandb
from datasets import Dataset, load_dataset

# Initialize Unsloth and GRPO
from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)
from unsloth import is_bfloat16_supported

# Initialize wandb
MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
wandb.init(project="chess-reasoner", name=f"{MODEL.split('/')[-1]}-chess-grpo")

# Initialize chess engine
def setup_engine() -> chess.engine.SimpleEngine:
    try:
        return chess.engine.SimpleEngine.popen_uci("stockfish")
    except FileNotFoundError:
        raise FileNotFoundError(
            "Stockfish not found. Please install stockfish and make sure it's in your PATH."
        )

engine = setup_engine()
print("Chess engine initialized successfully!")

# System prompt and XML format
SYSTEM_PROMPT = """
You are a chess expert. Given a chess position in FEN notation, you will analyze the position and suggest the best move.

In your analysis:
1. Assess the position's key features (material, pawn structure, king safety)
2. Identify tactical opportunities (forks, pins, discovered attacks)
3. Consider strategic plans based on the position
4. Calculate concrete variations when needed
5. Compare candidate moves and evaluate their strengths and weaknesses

Respond in the following format:
<think>
[Your step-by-step analysis of the position, considering tactics, strategy, and possible continuations]
</think>

[Your chosen move in algebraic notation, e.g., e2e4, g8f6, etc.]
"""

XML_FORMAT = """\
<think>
{reasoning}
</think>

{answer}
"""

# Utility functions
def extract_answer(text: str) -> str:
    """Extract the answer (move) which comes after the </think> tag"""
    if "</think>" not in text:
        return ""
    try:
        # Get the text after the closing </think> tag
        answer = text.split("</think>")[-1].strip()
        # Take the first line of text after </think>
        answer = answer.split("\n")[0].strip()
        return answer
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

def prepare_chess_dataset(num_samples: int = 1000) -> Dataset:
    """Create a dataset of chess positions from games"""
    dataset = load_dataset("Icannos/lichess_games", streaming=True)
    positions = []

    # Get random positions from games
    count = 0
    for row in dataset["train"]:
        if count >= num_samples:
            break
        try:
            fen, _ = get_random_position(row)
            positions.append({"fen": fen})
            count += 1
            if count % 100 == 0:
                print(f"Processed {count} positions...")
        except Exception as e:
            print(f"Error in game {count}: {e}")

    return Dataset.from_dict({"fen": [p["fen"] for p in positions]})

def is_valid_move(move_str: str, board: chess.Board) -> bool:
    """Check if a move string is valid for the given board position"""
    try:
        move = chess.Move.from_uci(move_str)
        return move in board.legal_moves
    except:
        return False

def evaluate_move(board: chess.Board, move_str: str, time_limit: float = 0.1) -> float:
    """
    Evaluate a chess move using Stockfish engine by comparing position evaluation
    before and after the move. Returns a score between 0 and 1.
    """
    try:
        move = chess.Move.from_uci(move_str)
        if move not in board.legal_moves:
            return 0.0

        # Get the initial position evaluation
        initial_result = engine.analyse(board, chess.engine.Limit(time=time_limit))
        best_move = initial_result["pv"][0]

        # Create a copy of the board and apply the best move
        best_board = board.copy()
        best_board.push(best_move)
        best_eval = engine.analyse(best_board, chess.engine.Limit(time=time_limit))
        best_score = best_eval["score"].relative.score(mate_score=10000)

        # Now evaluate the player's move
        player_board = board.copy()
        player_board.push(move)
        player_eval = engine.analyse(player_board, chess.engine.Limit(time=time_limit))
        player_score = player_eval["score"].relative.score(mate_score=10000)

        # Calculate evaluation difference (negative means worse than best move)
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
            # Linear scale from 0.0 to 1.0
            return 1.0 + (eval_diff / 300.0)

    except Exception as e:
        print(f"Error evaluating move: {e}")
        return 0.0

# Reward functions for GRPO
def move_correctness_reward(prompts, completions, board_state, **kwargs) -> List[float]:
    """
    Reward based on how good the suggested move is according to the engine.
    Uses position evaluation difference before and after the move.
    """
    responses = [completion[0]["content"] for completion in completions]
    extracted_moves = [extract_answer(r) for r in responses]

    rewards = []
    for i, move in enumerate(extracted_moves):
        # Clean up the move string
        move = move.strip()
        
        # Get the board for this completion
        board = board_state[i]
        
        # Evaluate the move
        reward = evaluate_move(board, move)
        
        # Apply weight of 0.7 to emphasize move quality
        weighted_reward = 0.7 * reward
        rewards.append(weighted_reward)

        # Log quality
        quality = "illegal move"
        if reward == 1.0:
            quality = "best move"
        elif reward >= 0.7:
            quality = "good move"
        elif reward >= 0.3:
            quality = "okay move"
        elif reward > 0.0:
            quality = "weak move"

        wandb.log({"move_reward": reward, "move_quality": quality})

    # Log a sample for debugging
    if len(rewards) > 0:
        sample_idx = random.randint(0, len(rewards) - 1)
        fen = board_state[sample_idx].fen()
        move = extracted_moves[sample_idx]
        reward = rewards[sample_idx]
        
        # Get engine's best move
        analysis = engine.analyse(board_state[sample_idx], chess.engine.Limit(time=0.1))
        best_move = analysis["pv"][0].uci()
        
        print(f"\nPosition: {fen}")
        print(f"Move: {move}, Reward: {reward:.2f}")
        print(f"Engine's best move: {best_move}")

    return rewards

def legal_move_reward(completions, board_state, **kwargs) -> List[float]:
    """Reward function that checks if the move is legal (weight: 0.3)"""
    responses = [completion[0]["content"] for completion in completions]
    extracted_moves = [extract_answer(r) for r in responses]

    rewards = []
    for i, move in enumerate(extracted_moves):
        move = move.strip()
        legal = is_valid_move(move, board_state[i])
        wandb.log({"legal_move": 1 if legal else 0})
        rewards.append(0.3 if legal else 0.0)

    return rewards

def soft_format_reward_func(completions, **kwargs) -> List[float]:
    """Reward function that checks if the completion has the correct format"""
    pattern = r"<think>.*?</think>\s*\S+"  # <think> tags followed by non-whitespace (the move)
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.search(pattern, r, re.DOTALL) is not None for r in responses]
    for match in matches:
        wandb.log({"format_correct": 1 if match else 0})
    return [0.3 if match else 0.0 for match in matches]

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
    return [count_xml(c) for c in contents]

# Main function to prepare dataset and model
def prepare_data_and_model():
    print("Preparing dataset...")
    dataset = prepare_chess_dataset(num_samples=1000)

    def format_dataset(example):
        """Format dataset for GRPO training"""
        # Convert the chess board for use in the reward function
        board = chess.Board(example["fen"])
        
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Analyze this chess position and give the best move: {example['fen']}",
                },
            ],
            "board_state": board,
        }

    train_dataset = dataset.map(format_dataset)
    print("Dataset prepared successfully!")

    print("Setting up model...")
    # Model parameters
    max_seq_length = 1024
    lora_rank = 64

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        fast_inference=True,
        max_lora_rank=lora_rank,
        gpu_memory_utilization=0.9,  # Increased for the larger model
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
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
        per_device_train_batch_size=1,  # Reduced for the larger model
        gradient_accumulation_steps=8,  # Increased for the larger model
        num_generations=4,  # Reduced for the larger model
        max_prompt_length=256,
        max_completion_length=256,
        max_steps=500,
        save_steps=100,
        save_total_limit=3,
        max_grad_norm=0.1,
        report_to="wandb",
        output_dir="outputs",
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

        try:
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
                temperature=0.7,
                top_p=0.95,
                max_tokens=1024,
            )

            # Generate without LoRA
            output_base = model.fast_generate(
                [text],
                sampling_params=sampling_params,
                lora_request=None,
            )[0].outputs[0].text

            # Try to load the trained LoRA model
            try:
                lora_request = model.load_lora("chess_reasoner_llama_8b_lora")

                # Generate with our trained LoRA
                output_lora = model.fast_generate(
                    [text],
                    sampling_params=sampling_params,
                    lora_request=lora_request,
                )[0].outputs[0].text

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
                base_format = re.search(
                    r"<think>.*?</think>\s*\S+",
                    output_base,
                    re.DOTALL,
                ) is not None
                lora_format = re.search(
                    r"<think>.*?</think>\s*\S+",
                    output_lora,
                    re.DOTALL,
                ) is not None

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
                    analysis = engine.analyse(board, chess.engine.Limit(time=0.2))
                    best_move = analysis["pv"][0].uci()
                    print(f"Engine's best move: {best_move}")

            except Exception as e:
                print(f"Error with LoRA model: {e}")
                print("Only testing base model...")
                # Print base model results
                print("\nBase model response:")
                print(output_base)

                board = chess.Board(fen)
                base_move = extract_answer(output_base)
                print(f"\nBase model move: {base_move}")
                if base_move and is_valid_move(base_move, board):
                    base_score = evaluate_move(board, base_move)
                    print(f"Base model score: {base_score:.2f}")

        except Exception as e:
            print(f"Error testing position: {e}")

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