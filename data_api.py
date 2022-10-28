import json
from riotwatcher import LolWatcher, ApiError


lol_watcher = LolWatcher("RGAPI-bf21eefa-9b28-4b78-bb3f-6a519b0fd367")
my_region = 'na1'

match_id = 'NA1_4438021106'


def open_matches():
    # Open saved info
    with open('save_data/match_ids.json') as f:
        json_content = f.read()
        match_ids = json.loads(json_content)
    with open('save_data/match_info.json') as f:
        json_content = f.read()
        match_info = json.loads(json_content)

    for i, match_id in enumerate(match_ids):
        print(f"Geting match data #{i} for: {match_id}")
        match = lol_watcher.match.by_id(my_region, match_id)

        match_data = [match['info']['teams'][0]['win']]

        for i in range(0,10):
            champ_name =  match['info']['participants'][i]['championName']
            match_data.append(champ_name)

        match_info[match_id] = match_data
        # Remove match_id from json
        del match_ids[i]

        # Save every match incase of error
        with open('save_data/match_info.json', 'w') as f:
            json_content = json.dumps(match_info)
            f.write(json_content)
        with open('save_data/match_ids.json', 'w') as f:
            json_content = json.dumps(match_ids)
            f.write(json_content)




def get_more_match_ids():
    # Open saved information
    with open('save_data/match_ids.json') as f:
        json_content = f.read()
        match_ids = json.loads(json_content)
    with open('save_data/puuids.json') as f:
        json_content = f.read()
        puuids = json.loads(json_content)
    with open('save_data/puuids_checked.json') as f:
        json_content = f.read()
        puuids_checked = json.loads(json_content)


    # Get match history for each puuid 
    print(f"Geting match history for each puuid, count:{len(puuids)}")
    for i, puuid in enumerate(puuids):
        print(f"getting match history #{i} for {puuid}")
        player_match_ids = lol_watcher.match.matchlist_by_puuid(my_region, puuid, queue=420, type="ranked", count=100)
        match_ids += player_match_ids

    match_ids = list(set(match_ids)) # Delete Dupes

    # all puuids have been checked
    puuids_checked += puuids
    puuids_checked = list(set(puuids_checked)) # Delete Dupes

    # Save variables back to json
    with open('save_data/puuids.json', 'w') as f:
        json_content = json.dumps(puuids)
        f.write(json_content)
    with open('save_data/puuids_checked.json', 'w') as f:
        json_content = json.dumps(puuids_checked)
        f.write(json_content)
    with open('save_data/match_ids.json', 'w') as f:
        json_content = json.dumps(match_ids)
        f.write(json_content)




def get_more_puuids():
    # Open saved information
    with open('save_data/match_ids.json') as f:
        json_content = f.read()
        match_ids = json.loads(json_content)
    with open('save_data/puuids.json') as f:
        json_content = f.read()
        puuids = json.loads(json_content)
    with open('save_data/puuids_checked.json') as f:
        json_content = f.read()
        puuids_checked = json.loads(json_content)


    # Gather more puuids
    print(f"Gather more puuids")
    new_puuids = []
    for match_id in match_ids[0:5]:
        print(f"Getting participants from: {match_id}")
        match = lol_watcher.match.by_id(my_region, match_id)
        new_puuids += match['metadata']['participants']

    new_puuids = list(set(new_puuids)) # Delete dupes

    puuids_checked = set(puuids_checked)

    for puuid in new_puuids:
        if puuid in puuids_checked:
            print(f"{puuid:10} already been checked")
        else:
            puuids.append(puuid)

    puuids_checked = list(puuids_checked)

    # Save variables back to json
    with open('save_data/puuids.json', 'w') as f:
        json_content = json.dumps(puuids)
        f.write(json_content)
    with open('save_data/puuids_checked.json', 'w') as f:
        json_content = json.dumps(puuids_checked)
        f.write(json_content)
    with open('save_data/match_ids.json', 'w') as f:
        json_content = json.dumps(match_ids)
        f.write(json_content)



open_matches()

