import sys
import json
from riotwatcher import LolWatcher, ApiError


lol_watcher = LolWatcher("RGAPI-d72aa8f9-1d42-4687-8729-c00320870b43")
my_region = 'na1'



def open_matches():
    # Open saved info
    with open('save_data/match_ids.json') as f:
        match_ids = set(json.loads(f.read()))
    with open('save_data/match_ids_checked.json') as f:
        match_ids_checked = set(json.loads(f.read()))
    with open('save_data/match_info.json') as f:
        match_info = json.loads(f.read()) # dict

    try:
        for i, match_id in enumerate(match_ids.copy()):
            print(f"Geting match data #{i} for: {match_id}")
            match = lol_watcher.match.by_id(my_region, match_id)

            # If the data is empty we will get an index error on the zero
            try:
                match_data = [match['info']['teams'][0]['win']] 
                # get all 10 players on the list
                match_data += ([match['info']['participants'][j]['championName'] for j in range(10)])
                match_info[match_id] = match_data
            except IndexError:
                continue

            # Saved matchID as checked and Remove match_id from list of matches to check
            match_ids_checked.add(match_id)
    except Exception as e:
        raise Exception(f"An error occurred at line {sys.exc_info()[-1].tb_lineno}: {e}")
    finally:
            match_ids -= match_ids_checked # removes from match_ids for next time
            print("saving data")
            with open('save_data/match_ids.json', 'w') as f:
                f.write(json.dumps(list(match_ids))) # used up some match_ids
            with open('save_data/match_ids_checked.json', 'w') as f:
                f.write(json.dumps(list(match_ids_checked))) # mark down which ones we used
            with open('save_data/match_info.json', 'w') as f:
                f.write(json.dumps(match_info)) # new match_info! :D


# This function will iterate over puuids.json and retrieve more match_ids that havent already been checked and add them to the set
def get_more_match_ids():
    print("Opening saved data ...")
    with open('save_data/match_ids.json') as f:
        match_ids = set(json.loads(f.read()))
    with open('save_data/puuids.json') as f:
        puuids = set(json.loads(f.read()))
    with open('save_data/puuids_checked.json') as f:
        puuids_checked = set(json.loads(f.read()))
    with open('save_data/match_ids_checked.json') as f:
        match_ids_checked = set(json.loads(f.read()))

    # Get match history for each puuid 
    print(f"Geting match history for each puuid, count:{len(puuids)}")
    try:
        for i, puuid in enumerate(puuids):
            print(f"#{i} Getting match history from: {puuid}")
            players_matches = lol_watcher.match.matchlist_by_puuid(my_region, puuid, queue=420, type="ranked", count=100)

            for new_match_id in players_matches:
                if new_match_id not in match_ids_checked:
                    match_ids.add(new_match_id)
            #Done with this puuid
            puuids_checked.add(puuid)
    except Exception as e:
        raise Exception(f"An error occurred at line {sys.exc_info()[-1].tb_lineno}: {e}")
    finally:
        puuids -= puuids_checked # removes from puuids for next time
        # Save variables back to json
        with open('save_data/puuids.json', 'w') as f:
            f.write(json.dumps(list(puuids))) # used up some puuids
        with open('save_data/puuids_checked.json', 'w') as f:
            f.write(json.dumps(list(puuids_checked))) # mark down which ones we used
        with open('save_data/match_ids.json', 'w') as f:
            f.write(json.dumps(list(match_ids))) # new match_ids
        print("Done saving!")


#probably broken needs to be fixed like we did for get_more_match_ids
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
    try:
        for i, match_id in enumerate(match_ids):
            print(f"#{i} Getting participants from: {match_id}")
            match = lol_watcher.match.by_id(my_region, match_id)
            new_puuids += match['metadata']['participants']

            for new_puuid in match['metadata']['participants']:
                if new_puuid not in puuids_checked: # We got the matches from these
                        puuids.append(new_puuid)

    finally:
            with open('save_data/puuids.json', 'w') as f:
                f.write(json.dumps(puuids)) # new puuids is all there is to change
            print("Done saving!")

    print("Done getting more puuids!")


if __name__ == "__main__":
    #get_more_puuids()
    #get_more_match_ids()
    open_matches()
