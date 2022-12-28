import json

with open('save_data/match_ids.json') as f:
    match_ids = json.loads(f.read())
with open('save_data/match_ids_checked.json') as f:
    match_ids_checked = json.loads(f.read())

print(len(match_ids))
match_ids = set(match_ids)
match_ids_checked = set(match_ids_checked)

match_ids = match_ids.difference(match_ids_checked)

match_ids = list(match_ids)

with open('save_data/match_ids.json', 'w') as f:
    f.write(json.dumps(match_ids)) # new match_ids