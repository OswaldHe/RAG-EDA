import json
import re
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer

def extract_code_blocks(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    code_blocks = set()
    code_pattern = re.compile(r'```(.+?)```', re.DOTALL)
    inline_code_pattern = re.compile(r'`(.+?)`')

    for topic in data:
        for entry in topic['knowledge']:
            content = entry['content']
            unique_code_blocks = set(code_pattern.findall(content))
            unique_code_blocks.update(inline_code_pattern.findall(content))
            code_blocks.update(unique_code_blocks)
    code_blocks = list(code_blocks)
    return code_blocks

def plot_code_block_lengths(code_blocks):
    lengths = [len(block) for block in code_blocks]
    plt.hist(lengths, bins=50, edgecolor='black')
    plt.title('Distribution of Code Block Lengths')
    plt.xlabel('Length of Code Block')
    plt.ylabel('Frequency')
    plt.savefig('code_block_lengths_distribution.png')
    plt.show()

def tokenize(code_blocks):
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B")
    tokenized_blocks = [tokenizer(block)["input_ids"] for block in code_blocks]
    return tokenized_blocks

def dump_code_db(code_blocks, output_path='code_db.json'):
    code_db = {f"__code_{i}__": block for i, block in enumerate(code_blocks)}
    with open(output_path, 'w') as file:
        json.dump(code_db, file, indent=4)
    
    return code_db


def reconstruct_db(file_path, code_db):
    with open(file_path, 'r') as file:
        data = json.load(file)

    for topic in data:
        for entry in topic['knowledge']:
            content = entry['content']
            first_newline = content.find('\n')
            second_newline = content.find('\n\n', first_newline + 1)
            if first_newline != -1 and second_newline != -1:
                entry["summary"] = content[first_newline + 1:second_newline].strip()
            else:
                entry["summary"] = ""
            entry["summary"] = entry["summary"].replace('#', '').replace('\n', ',').strip()
            entry['code_keys'] = []
            for key, value in code_db.items():
                if f'```{value}```' in content:
                    entry['code_keys'].append(key)
                    content = content.replace(f'```{value}```', f'[{key}]')
                elif f'`{value}`' in content:
                    entry['code_keys'].append(key)
                    content = content.replace(f'`{value}`', f'[{key}]')
                
            entry['content'] = content

    with open('updated_openroad_documentation.json', 'w') as file:
        json.dump(data, file, indent=4)

if __name__ == "__main__":
    file_path = '/home/oswaldhe/RAG-EDA/benchmark/openroad_documentation.json'
    code_blocks = extract_code_blocks(file_path)
    print(len(code_blocks))
    print(code_blocks[0])
    db_dict = dump_code_db(code_blocks)
    code_blocks = tokenize(code_blocks)
    plot_code_block_lengths(code_blocks)
    average_length = np.mean([len(block) for block in code_blocks])
    print(f'Average length of code blocks: {average_length}')

    # 27% of the data are code blocks or inline code
    reconstruct_db(file_path, db_dict)