
import os

# Exploration space
output_dir="./output"
os.makedirs(output_dir, exist_ok=True)

with open(os.path.join(output_dir, "test_file.txt"), "w") as f:
    f.write("This is a test file.")