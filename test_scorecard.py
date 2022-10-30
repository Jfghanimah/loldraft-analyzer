import json
from scorecard import team_scorecard

# First I want to test to see if my scorecard can predict the winner better than 50%
with open('save_data/match_info.json') as f:
        match_info = json.loads(f.read())


match_keys = list(match_info.keys())
correct = 0
wrong = 0
total = 0
for key in match_keys:
    match =  match_info[key]
    team1 = match[1:6]
    team2 = match[6:]

    score1 = team_scorecard(team1)
    score2 = team_scorecard(team2)

    if abs(score1-score2) < 1.75:
        continue

    if score1>score2:
        if match[0] is True:
            correct += 1
        else:
            wrong += 1
    else:
        if match[0] is False:
            correct += 1
        else:
            wrong += 1

    total += 1

print(correct, total, f" Percent: {correct/total}")

