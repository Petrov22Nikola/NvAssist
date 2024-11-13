import random
import json

def process_file(filename, output_file='chatml.json'):
    # Read the file content
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    num_lines = len(lines)
    result = []
    
    # Iterate through the file lines
    for i in range(num_lines):
        # Select random line count for context and target
        before_lines_count = random.randint(1, 10)
        after_lines_count = random.randint(1, 10)
        target_lines_count = random.randint(1, 10)
        
        # Determine start and end indices for context
        start_before = max(0, i - before_lines_count)
        start_target = i
        end_target = min(num_lines, i + target_lines_count)
        end_after = min(num_lines, end_target + after_lines_count)
        
        # Assemble context sections
        prefix = ''.join(lines[start_before:start_target])
        target = ''.join(lines[start_target:end_target])
        suffix = ''.join(lines[end_target:end_after])
        
        # Format with user and assistant roles
        formatted_entry = {
            "user": f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>",
            "assistant": f"{target}<|fim_middle|>"
        }
        result.append(formatted_entry)
        
    # Write to JSON file
    with open(output_file, 'w') as outfile:
        json.dump(result, outfile, indent=2)

# Run the function with a specified file
process_file('C:\\VVL\\tests\\unit\\memory.cpp')
