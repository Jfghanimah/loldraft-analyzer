import json
from riotwatcher import LolWatcher, ApiError


lol_watcher = LolWatcher("RGAPI-1b552721-0ae9-459f-8d83-6f48278d58a6")
my_region = 'na1'


def open_matches():
    # Open saved info
    with open('save_data/match_ids.json') as f:
        match_ids = json.loads(f.read())
    with open('save_data/match_ids_checked.json') as f:
        match_ids_checked = json.loads(f.read())
    with open('save_data/match_info.json') as f:
        match_info = json.loads(f.read())

    for i, match_id in enumerate(match_ids.copy()):
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
            # To my knowledge there cant be dupes because matchid returns 1 info
            # Save every 100 incase of error
            with open('save_data/match_ids.json', 'w') as f:
                f.write(json.dumps(match_ids)) # used up some match_ids
            with open('save_data/match_ids_checked.json', 'w') as f:
                f.write(json.dumps(match_ids_checked)) # mark down which ones we used
            with open('save_data/match_info.json', 'w') as f:
                f.write(json.dumps(match_info)) # new match_info! :D


# This function is broken please dont use
def get_more_match_ids():
    print("Opening saved data ...")
    with open('save_data/match_ids.json') as f:
        match_ids = json.loads(f.read())
    with open('save_data/puuids.json') as f:
        puuids = json.loads(f.read())
    with open('save_data/puuids_checked.json') as f:
        puuids_checked = json.loads(f.read())
    with open('save_data/match_ids_checked.json') as f:
        match_ids_checked = json.loads(f.read())

    # Get match history for each puuid 
    print(f"Geting match history for each puuid, count:{len(puuids)}")
    new_match_ids = []
    for i, puuid in enumerate(puuids.copy()): #1000 puuids =~ 100000 match_ids
        print(f"#{i} Getting match history from: {puuid}")
        players_matches = lol_watcher.match.matchlist_by_puuid(my_region, puuid, queue=420, type="ranked", count=100)
        new_match_ids += players_matches

        # Checked puuid so move it to checked and remove from original
        puuids_checked.append(puuid)
        del puuids[i]

        if i+1 % 100 == 0:
            print("Saving progress... ")
            seen_before = 0
            new_match_ids = list(set(new_match_ids)) # Delete Dupes

            # only append if not already checked
            for new_id in new_match_ids:
                if new_id in match_ids_checked: # These matches have been opened
                    seen_before += 1
                else:
                    match_ids.append(new_id)

            print(f"We've already checked {seen_before} of the new match_ids")
            new_match_ids = [] # reset for next batch of 100

            # Save process
            match_ids = list(set(match_ids)) # Delete Dupes before
            puuids_checked = list(set(puuids_checked)) # Delete Dupes
            # Save variables back to json
            with open('save_data/puuids.json', 'w') as f:
                f.write(json.dumps(puuids)) # used up some puuids
            with open('save_data/puuids_checked.json', 'w') as f:
                f.write(json.dumps(puuids_checked)) # mark down which ones we used
            with open('save_data/match_ids.json', 'w') as f:
                f.write(json.dumps(match_ids)) # new match_ids
            print("Done saving!")


def get_more_puuids():
    print("Opening saved data ...")
    with open('save_data/match_ids.json') as f:
        match_ids = json.loads(f.read())
    with open('save_data/puuids.json') as f:
        puuids = json.loads(f.read())
    with open('save_data/puuids_checked.json') as f:
        puuids_checked = json.loads(f.read()) # checked means we have its match_ids

    # Gather more puuids
    print(f"Gathering more puuids:")
    new_puuids = []
    for i, match_id in enumerate(match_ids):
        print(f"#{i} Getting participants from: {match_id}")
        match = lol_watcher.match.by_id(my_region, match_id)
        new_puuids += match['metadata']['participants']

        if i+1 % 100 == 0:
            print("Saving progress... ")
            seen_before = 0
            new_puuids = list(set(new_puuids)) # Delete dupes
            # only append if not already checked
            for new_puuid in new_puuids:
                if new_puuid in puuids_checked: # We got the matches from these
                    seen_before += 1
                else:
                    puuids.append(new_puuid)
            
            print(f"We've already checked {seen_before} of the new puuids")
            new_puuids = [] # reset for next batch of 100

            # Save process
            puuids = list(set(puuids)) # Delete Dupes
            # Save new_puuids back to json
            puuids = list(set(puuids)) # Delete dupes
            with open('save_data/puuids.json', 'w') as f:
                f.write(json.dumps(puuids)) # new puuids is all there is to change
            print("Done saving!")

    print("Done getting more puuids!")


if __name__ == "__main__":
    #get_more_puuids()
    #get_more_match_ids()
    open_matches()
