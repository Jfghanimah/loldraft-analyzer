import csv 


champion_features = {}
with open("champion_features.csv") as f:
    for row in csv.DictReader(f):
        champion_features[row['Champion']] = row



game1 = ['Blue', 'Wukong', 'Ekko', 'Irelia', 'Xayah', 'Rakan', 'Sett', 'Kindred', 'Swain', 'Ezreal', 'Senna']
game2 = ['Red', 'Quinn', 'Nocturne', 'Akshan', 'Miss Fortune', 'Senna', 'Camille', 'Hecarim', 'Syndra', 'Lucian', 'Nami']
game3 = ['Blue', 'Wukong', 'Xin Zhao', 'Irelia', 'Seraphine', 'Sona', 'Kayle', 'Evelynn', 'Ahri', 'Ezreal', 'Lux']
game4 = ['Blue', 'Ornn', 'Graves', 'Azir', 'Ashe', 'Nautilus', 'Nasus', 'Lee Sin', 'Akali', 'Miss Fortune', 'Lux']
game5 = ['Red', 'Fiora', 'Shaco', "Vel'Koz", 'Jinx', 'Morgana', 'Malphite', 'Viego', 'Orianna', 'Twitch', 'Rell']




def team_scorecard(champions):
    # returns score for team using various tags
    
    # Score will be out of 8 points for the following areas
    # Tank, Damage Split, Range, Power Curve, Heavy CC, Engage/Dis, Poke, wave clear 
    total_score = 0

    tank_count, magic_count, phyiscal_count, melee_count, heavycc, engage_dis, poke, wave_clear = 0, 0, 0, 0, 0, 0, 0, 0
    power_curve = [0, 0, 0]

    #count up attributes for whole team
    for champ in champions:
        champ_class = champion_features[champ]['Class']
        if champ_class == 'Vanguard' or champ_class == 'Warden' or champ_class == 'Juggernaut':
            tank_count += 1
        elif champ_class == 'Diver' :
            tank_count += 0.75
        elif champ_class == 'Skirmisher':
            tank_count += 0.5

        if champion_features[champ]["Damage"] == 'Physical':
            phyiscal_count += 1
        elif champion_features[champ]["Damage"] == 'Magical':
            magic_count += 1
        else:
            phyiscal_count += 1
            magic_count += 1

        if champion_features[champ]["Range"] == 'Melee':
            melee_count += 1

        if champion_features[champ]["HeavyCC"] == '1':
            heavycc += 1

        if champion_features[champ]["Engage"] == '1' or champion_features[champ]["Disengage"] == '1':
            engage_dis += 1

        if champion_features[champ]["Poke"] == '1':
            poke += 1

        if champion_features[champ]["Waveclear"] == '1':
            wave_clear += 1

        if champion_features[champ]["Early"] == '1':
            power_curve[0] += 1
        if champion_features[champ]["Mid"] == '1':
            power_curve[1] += 1
        if champion_features[champ]["Late"] == '1':
            power_curve[2] += 1

        
    #Start Scoring team
    print(f"Tank Points: {tank_count}")
    if tank_count >= 1.5:
        total_score += 1
    elif tank_count >= 1:
        total_score += 0.5

    print(f"Magic: {magic_count} - Physical: {phyiscal_count}")
    if magic_count >= 2 and phyiscal_count >= 2:
        total_score += 1
   
    print(f"Melee Champions: {melee_count}")
    if melee_count == 2 or melee_count == 3:
        total_score += 1

    print(f"Heavy CC Champions: {heavycc}")
    if heavycc >= 4:
        total_score += 1
    elif heavycc == 3:
        total_score += 0.5

    print(f"Dis/Engage Champions: {engage_dis}")
    if engage_dis >= 3:
        total_score += 1
    elif engage_dis == 2:
        total_score += 0.5

    print(f"Poke Champions: {poke}")
    if poke >= 1:
        total_score += 1

    print(f"Waveclear Champions: {wave_clear}")
    if wave_clear >= 1:
        total_score +=1

    print(f"Powercurve {power_curve}")
    if power_curve[0] >= 2 and power_curve[1] >= 1 and power_curve[2] >= 2:
        total_score += 1
    elif power_curve[0] >= 1 and power_curve[1] >= 1 and power_curve[2] >= 1:
        total_score += 0.75

    return (total_score/8) * 100



print(team_scorecard(game5[1:6]))
print(" ")
print(team_scorecard(game5[6:]))
