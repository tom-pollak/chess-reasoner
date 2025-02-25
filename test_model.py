

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


