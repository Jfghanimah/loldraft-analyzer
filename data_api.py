import json
from riotwatcher import LolWatcher, ApiError


lol_watcher = LolWatcher("RGAPI-9cd9f7ab-9125-4641-8820-990d1aa517d7")
my_region = 'na1'


def open_matches():
    # Open saved info
    with open('save_data/match_ids.json') as f:
        match_ids = json.loads(f.read())
    with open('save_data/match_ids_checked.json') as f:
        match_ids_checked = json.loads(f.read())
    with open('save_data/match_info.json') as f:
        match_info = json.loads(f.read())

    for i, match_id in enumerate(match_ids):
        print(f"Geting match data #{i} for: {match_id}")
        match = lol_watcher.match.by_id(my_region, match_id)

        # If the data is empty we will get an index error on the zero
        try:
            match_data = [match['info']['teams'][0]['win']]
            for j in range(0,10):
                champ_name =  match['info']['participants'][j]['championName']
                match_data.append(champ_name)
                match_info[match_id] = match_data
        except IndexError:
            continue

        # Saved matchID as checked and Remove match_id from list of matches to check
        match_ids_checked.append(match_id)
        del match_ids[i]

        if i % 100 == 0:
            print("saving data")
            # Save every 100 incase of error
            with open('save_data/match_ids.json', 'w') as f:
                f.write(json.dumps(match_ids))
            with open('save_data/match_ids_checked.json', 'w') as f:
                f.write(json.dumps(match_ids_checked))
            with open('save_data/match_info.json', 'w') as f:
                f.write(json.dumps(match_info))




def get_more_match_ids():
    # Open saved information
    with open('save_data/match_ids.json') as f:
        match_ids = json.loads(f.read())
    with open('save_data/puuids.json') as f:
        puuids = json.loads(f.read())
    with open('save_data/puuids_checked.json') as f:
        puuids_checked = json.loads(f.read())


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
        f.write(json.dumps(puuids))
    with open('save_data/puuids_checked.json', 'w') as f:
        f.write(json.dumps(puuids_checked))
    with open('save_data/match_ids.json', 'w') as f:
        f.write(json.dumps(match_ids))




def get_more_puuids():
    # Open saved information
    with open('save_data/match_ids.json') as f:
        match_ids = json.loads(f.read())
    with open('save_data/puuids.json') as f:
        puuids = json.loads(f.read())
    with open('save_data/puuids_checked.json') as f:
        puuids_checked = json.loads(f.read())


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
        f.write(json.dumps(puuids))
    with open('save_data/puuids_checked.json', 'w') as f:
        f.write(json.dumps(puuids_checked))
    with open('save_data/match_ids.json', 'w') as f:
        f.write(json.dumps(match_ids))



open_matches()

