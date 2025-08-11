import re
with open("union.py", "r") as file:
    lines = file.readlines()

file_blocks = {}
current_file = None
current_content = []

for line in lines:
    match = re.match(r'#\s*(\S+\.py)', line)
    if match:
        if current_file and current_content:
            file_blocks[current_file] = ''.join(current_content)
        current_file = match.group(1)
        current_content = []
    elif current_file:
        current_content.append(line)

if current_file and current_content:
    file_blocks[current_file] = ''.join(current_content)

for fname, content in file_blocks.items():
    with open(fname, "w") as f:
        f.write(content)
with open("union.py", "r") as file:
    text = file.read()
    