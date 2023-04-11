import os, json, sys
all_df = []
folder = sys.argv[1]
for file in os.listdir(folder):
    path = os.path.join(folder, file)
    if not file.startswith('score'):
        continue
    print(file)

    with open(path, 'r') as f:
        df = json.load(f)
    print(len(df))

    for x in df:
        all_df.append({'query':x["prompt"], 'responses':x['response'], 'scores':x['scores']})

print(f'---{len(all_df)}')

with open(sys.argv[1] + '/train_data.json', 'w') as f:
    for line in all_df:
        f.write(json.dumps(line).strip() + '\n')