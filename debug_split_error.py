"""
Debugging script to identify the 'list' object has no attribute 'split' error
in the chess-reasoner project.
"""

import io
import chess
import chess.pgn
import random
from typing import List, Tuple

# Mock functions to simulate the environment

def extract_xml_answer(text):
    """
    Copy of the original function that's causing the error.
    This will help us identify what type of data is being passed.
    """
    print(f"extract_xml_answer received: {type(text)}")
    print(f"Value: {text}")
    
    if isinstance(text, list):
        print("ERROR: Received a list instead of a string!")
        # Try to handle the list case for debugging
        if text and isinstance(text[0], str):
            text = text[0]
        else:
            return "ERROR: list input"
    
    if "<answer>" not in text or "</answer>" not in text:
        return ""
    try:
        answer = text.split("<answer>")[-1]
        answer = answer.split("</answer>")[0]
        return answer.strip()
    except Exception as e:
        print(f"Error extracting answer: {e}")
        return ""

# Mock data structures to simulate different scenarios

# Test case 1: Normal string input (should work)
test_normal = "Some text <answer>the correct move</answer> more text"

# Test case 2: List of strings (potential problem)
test_list = ["<answer>e4</answer>", "additional text"]

# Test case 3: Nested structure similar to what might be in completions
test_nested = [{"content": "<answer>d4</answer>"}]

# Test case 4: Nested structure with list as content (potential issue)
test_problematic = [{"content": ["<answer>Nf3</answer>", "extra"]}]

# Simulate the move_correctness_reward function
def mock_move_correctness_reward():
    print("\nSimulating normal case (string content):")
    responses = ["<answer>e4</answer>"]
    extracted_moves = [extract_xml_answer(r) for r in responses]
    print(f"Extracted moves: {extracted_moves}")
    
    print("\nSimulating problematic case (list content):")
    responses_list = [["<answer>d4</answer>"]]
    try:
        extracted_moves = [extract_xml_answer(r) for r in responses_list]
        print(f"Extracted moves: {extracted_moves}")
    except Exception as e:
        print(f"Error occurred: {e}")
    
    print("\nSimulating potential real-world case:")
    completions = [[{"content": "<answer>e4</answer>"}]]
    try:
        responses = [completion[0]["content"] for completion in completions]
        print(f"Response types: {[type(r) for r in responses]}")
        extracted_moves = [extract_xml_answer(r) for r in responses]
        print(f"Extracted moves: {extracted_moves}")
    except Exception as e:
        print(f"Error occurred: {e}")
    
    print("\nSimulating potential error case:")
    completions_error = [[{"content": ["<answer>d4</answer>"]}]]
    try:
        responses = [completion[0]["content"] for completion in completions_error]
        print(f"Response types: {[type(r) for r in responses]}")
        extracted_moves = [extract_xml_answer(r) for r in responses]
        print(f"Extracted moves: {extracted_moves}")
    except Exception as e:
        print(f"Error occurred: {e}")

# Test all cases
def run_all_tests():
    print("\n=== Testing extract_xml_answer with different inputs ===")
    
    print("\nTest 1: Normal string input")
    result1 = extract_xml_answer(test_normal)
    print(f"Result: {result1}")
    
    print("\nTest 2: List of strings input")
    try:
        result2 = extract_xml_answer(test_list)
        print(f"Result: {result2}")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\nTest 3: Nested structure")
    try:
        result3 = extract_xml_answer(test_nested[0]["content"])
        print(f"Result: {result3}")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\nTest 4: Problematic nested structure")
    try:
        result4 = extract_xml_answer(test_problematic[0]["content"])
        print(f"Result: {result4}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test the move_correctness_reward function
    print("\n=== Testing mock_move_correctness_reward function ===")
    mock_move_correctness_reward()

if __name__ == "__main__":
    run_all_tests()