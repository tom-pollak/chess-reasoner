"""
Test script for chess reasoner model
Loads and evaluates a model from Hugging Face Hub
"""

import re

import chess
import chess.engine

# Import constants and functions from training script
from chess_grpo import (
    ENGINE_ANALYSIS_TIME,
    MAX_COMPLETION_LENGTH,
    SYSTEM_PROMPT,
    evaluate_move,
    extract_answer,
    is_valid_move,
    setup_engine,
)
from unsloth import FastLanguageModel

# Initialize engine
engine = setup_engine()


def test_model_from_hub(repo_id="tommyp1111/chess-reasoner"):
    """Load and test a model from Hugging Face Hub"""
    print(f"Loading model from HuggingFace Hub: {repo_id}")

    # Load model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        max_seq_length=1024,
        load_in_4bit=True,
        fast_inference=True,
        gpu_memory_utilization=0.8,
    )

    # Test positions
    test_positions = [
        chess.STARTING_FEN,  # Starting position
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",  # Common position after 1.e4 e5 2.Nf3 Nc6
        "rnbqkb1r/pp2pppp/3p1n2/2p5/4P3/2N2N2/PPPP1PPP/R1BQKB1R w KQkq - 0 4",  # Sicilian Defense
    ]

    print("\nTesting model on example positions...")

    # Load model from hub
    try:
        lora_request = model.load_lora_from_hub(repo_id)
        print(f"Successfully loaded model from {repo_id}")
    except Exception as e:
        print(f"Error loading model from hub: {e}")
        return

    from vllm import SamplingParams

    sampling_params = SamplingParams(
        temperature=0.5,
        top_p=0.95,
        max_tokens=MAX_COMPLETION_LENGTH,
    )

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
        output = (
            model.fast_generate(
                [text],
                sampling_params=sampling_params,
                lora_request=lora_request,
            )[0]
            .outputs[0]
            .text
        )

        # Print results
        print("\nModel response:")
        print(output)

        # Evaluate model
        board = chess.Board(fen)
        move = extract_answer(output)

        # Check format correctness
        format_correct = (
            re.search(
                r"<think>.*?</think>\s*\S+",
                output,
                re.DOTALL,
            )
            is not None
        )

        print(f"\nFormat correct: {format_correct}")

        # Check move correctness
        print(f"Model move: {move}")
        if move:
            is_legal = is_valid_move(move, board)
            print(f"Move legal: {is_legal}")
            if is_legal:
                score = evaluate_move(board, move)
                print(f"Move quality score: {score:.2f}")

        # Get top engine move for comparison
        if engine is not None:
            analysis = engine.analyse(
                board, chess.engine.Limit(time=ENGINE_ANALYSIS_TIME)
            )
            best_move = analysis["pv"][0].uci()
            print(f"Engine's best move: {best_move}")


if __name__ == "__main__":
    test_model_from_hub()
    engine.quit()
