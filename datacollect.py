import os
from work.similarity_net.utils import ArgConfig

args = ArgConfig()
args = args.build_conf()

text = []
with open(args.infer_data_dir, "r",encoding="utf-8") as file:
    for line in file:
        text.append(line)

with open(args.train_data_dir,encoding="utf-8") as file:
    for line in file:
        query, pos_title, neg_title = line.strip().split("\t")
        if query not in text:
            text.append(query)
        if pos_title not in text:
            text.append(pos_title)
        if neg_title not in text:
            text.append(neg_title)
print("OK")

with open(args.valid_data_dir,encoding="utf-8") as file:
    for line in file:
        query, title, label = line.strip().split("\t")
        if query not in text:
            text.append(query)
        if title not in text:
            text.append(title)
print("OK")

with open(args.test_data_dir,encoding="utf-8") as file:
    for line in file:
        query, title, label = line.strip().split("\t")
        if query not in text:
            text.append(query)
        if title not in text:
            text.append(title)
print("OK")

with open(args.infer_data_dir, "a",encoding="utf-8") as file:
    for line in text:
        file.write(line + "\n")