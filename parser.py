"""
# Parser
    This module provides a parser for text.

## Functions
    - *parse_total_bits_sent*: Parses the log text and extracts the 'Total bits sent'.

## Author
    Miquel P. Baztan Grau

## Date
    23/09/2024
"""

import re
from utils import read_file_to_string, plot_two_dicts

## Constants
N = 3
FILE1 = f"merge/merge{N}.txt"
FILE2 = f"tests/test{N}/results.txt"
MAXITER = 80

PLOTNAME = f"plots/data{N}.png"

def parse_total_bits_sent(log_text):
    """
    Parses the log text and extracts the 'Total bits sent' from the last agent action
    within each iteration.
    
    Args:
        log_text (str): The log text to parse.
    
    Returns:
        dict: A dictionary mapping iteration numbers to their 'Total bits sent' values.
    """

    # Regular expression patterns
    iteration_pattern = re.compile(r'^Iteration\s+(\d+):', re.MULTILINE)
    total_bits_pattern = re.compile(r'^Total bits sent:\s*(\d+)', re.MULTILINE)
    
    # Find all iteration starts
    iterations = list(iteration_pattern.finditer(log_text))
    
    results = {}
    
    for i, iter_match in enumerate(iterations):
        iter_num = int(iter_match.group(1))
        start_index = iter_match.end()
        # Determine the end of this iteration block
        if i + 1 < len(iterations):
            end_index = iterations[i + 1].start()
        else:
            end_index = len(log_text)
        
        # Extract the block for this iteration
        iter_block = log_text[start_index:end_index]
        
        # Find all 'Total bits sent' in this block
        bits_matches = list(total_bits_pattern.finditer(iter_block))
        
        if bits_matches:
            # Take the last match in this block
            last_bits = int(bits_matches[-1].group(1))
            results[iter_num] = last_bits
        else:
            # If no match found, set to None or a default value
            results[iter_num] = None
    
    return results

def main(file1 : str = FILE1, file2 : str = FILE2):
    # Sample log text1
    log_text1 = read_file_to_string(file1)

    # Parse the log1
    parsed_results1_long = parse_total_bits_sent(log_text1)
    parsed_results1 = {}
    for i in range(MAXITER):
        parsed_results1[i] = parsed_results1_long[i]


    # Sample log text2
    log_text2 = read_file_to_string(file2)

    # Parse the log2
    parsed_results2 = parse_total_bits_sent(log_text2)

    plot_two_dicts(parsed_results1, parsed_results2, "NLO and merged agent comparison", "merged", "NLO", "Iterations", "Bits sent", PLOTNAME)

if __name__ == "__main__":
    main()