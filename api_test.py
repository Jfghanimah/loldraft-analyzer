from riotwatcher import LolWatcher, ApiError
import os
from dotenv import load_dotenv

load_dotenv()
RIOT_API_KEY = os.getenv("RIOT_API_KEY")
if not RIOT_API_KEY:
    raise ValueError("RIOT_API_KEY not found in .env file")

lol_watcher = LolWatcher(RIOT_API_KEY)

my_region = 'na1'

me = lol_watcher.summoner.by_name(my_region, 'drdoughnutdude')
print(me['profileIconId'])