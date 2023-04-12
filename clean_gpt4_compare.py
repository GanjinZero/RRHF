import json
import os

with open('comparision_data.json', 'r') as f:
    df = json.load(f)

new_df = []
for x in df:
    query = x['user_input']
    res = [x['completion_a'], x['completion_b']]
    scores = [2, 1]
    new_df.append({'query': query, 'responses':res, 'scores':scores})

with open('gpt4_compare_train.json', 'w') as f:
    for line in new_df:
        f.write(json.dumps(line).strip() + '\n')
