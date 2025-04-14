import re

# Read the file
file_path = '/Users/adam/Development/vinkeljernet/api_clients_optimized.py'
with open(file_path, 'r') as file:
    lines = file.readlines()

# Print the line where the error occurs (line 1355)
line_num = 1355
if len(lines) >= line_num:
    print(f"Line {line_num}: {repr(lines[line_num-1])}")
    
    # Check if this line contains 'eller' without spaces around it
    if 'eller' in lines[line_num-1]:
        # Replace all instances of 'eller' with 'or' in this line
        fixed_line = lines[line_num-1].replace('eller', 'or')
        lines[line_num-1] = fixed_line
        print(f"Fixed line {line_num}: {repr(fixed_line)}")
        
        # Write the fixed content back to the file
        with open(file_path, 'w') as file:
            file.writelines(lines)
        print(f"File {file_path} has been updated.")
    else:
        print(f"No 'eller' found in line {line_num}")
else:
    print(f"File has only {len(lines)} lines, but we need to check line {line_num}")

print("Done! Please try running your program again.")