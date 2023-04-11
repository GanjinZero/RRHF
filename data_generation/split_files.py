from datasets import load_dataset
import json
import sys

dataset = load_dataset('Dahoas/rm-static')['train']
with open(sys.argv[1] + '/raw_generation.json', 'r') as f:
    samples = [json.loads(item) for item in f.readlines()]

samples = samples[:len(dataset)]

assert len(dataset) == len(samples)

print('=='*10)
print(samples[-1])
print(dataset[-1])

import tqdm

buffer = []
count = 0
for idx in tqdm.tqdm(range(len(samples))):
    temp = [samples[idx][0][0], [item[1] for item in samples[idx]]]
    temp.append(dataset[idx]['chosen'])
    temp.append(dataset[idx]['rejected'])
    temp[1] = [i.replace(temp[0], "") for i in temp[1]]
    buffer.append(temp)
    if len(buffer) == 10000:
        with open(sys.argv[2] + f'/beam4_{count}.txt', 'w') as f:
            for item in buffer:
                f.write(json.dumps(item) + '\n')
        count += 1
        buffer = []

with open(sys.argv[2] + f'/beam4_{count}.txt', 'w') as f:
    for item in buffer:
        f.write(json.dumps(item) + '\n')
        


        