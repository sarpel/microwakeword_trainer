import re

with open("src/training/rich_logger.py", "r") as f:
    content = f.read()

print("Occurrences of box=None in rich_logger.py:")
for m in re.finditer(r"box=None", content):
    print(m.start())
