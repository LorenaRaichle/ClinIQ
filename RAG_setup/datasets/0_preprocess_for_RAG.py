
import json
import pprint

with open("train_dataset.json", "r") as f:
    data = json.load(f)

print(data["multiple_choice"])



repl_all_mask = [1 if entry['options'][entry['correct_answer']] in ['All', 'All.', 'All of the above', 'All of the above.', 'All of these', 'All of these.', 'ALL', 'ALL.', 'ALL OF THE ABOVE', 'ALL OF THE ABOVE.', 'ALL OF THESE', 'ALL OF THESE.'] else 0 for entry in data['multiple_choice']]
repl_none_mask = [1 if entry['options'][entry['correct_answer']] in ['None', 'None.', 'None of the above', 'None of the above.', 'None of these', 'None of these.', 'NONE', 'NONE.', 'NONE OF THE ABOVE', 'NONE OF THE ABOVE.', 'NONE OF THESE', 'NONE OF THESE.'] else 0 for entry in data['multiple_choice']]


for idx, entry in enumerate(data['multiple_choice']):
    if repl_all_mask[idx]:
        entry['correct_answer'] = entry['options']['A'] + ', ' + entry['options']['B'] + ', ' + entry['options']['C']
    elif repl_none_mask[idx]:
        entry['correct_answer'] = ''
    else:
        entry['correct_answer'] = entry['options'][entry['correct_answer']]

pprint.pp(data['multiple_choice'])

with open("train_dataset_rag.json", "w") as f:
    data = json.dump(data, f)



