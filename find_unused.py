import os
import ast

def get_defined_names(file_path):
    try:
        with open(file_path, 'r') as f:
            tree = ast.parse(f.read())
    except:
        return []
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            names.append(node.name)
        elif isinstance(node, ast.ClassDef):
            names.append(node.name)
    return names

def is_used(name, file_path, all_files):
    for f in all_files:
        if f == file_path:
            continue
        try:
            with open(f, 'r') as file:
                content = file.read()
                # Simple check, but to avoid false positives, check if name is word
                if re.search(r'\b' + re.escape(name) + r'\b', content):
                    return True
        except:
            pass
    return False

all_files = []
for root, dirs, files in os.walk('src'):
    for file in files:
        if file.endswith('.py'):
            all_files.append(os.path.join(root, file))
for root, dirs, files in os.walk('scripts'):
    for file in files:
        if file.endswith('.py'):
            all_files.append(os.path.join(root, file))

unused = []
for file_path in all_files:
    names = get_defined_names(file_path)
    for name in names:
        if not is_used(name, file_path, all_files):
            unused.append((file_path, name))

for f, n in unused:
    print(f"{f}: {n}")
