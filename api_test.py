from riotwatcher import LolWatcher, ApiError

lol_watcher = LolWatcher('RGAPI-1812c618-07d4-479c-afac-ce8db5ca4985')

my_region = 'na1'

me = lol_watcher.summoner.by_name(my_region, 'drdoughnutdudee')
print(me['profileIconId'])