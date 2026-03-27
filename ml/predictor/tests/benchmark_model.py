import torch
import json
import os
from ml.predictor.models_pytorch import WinPredictorModel

# --- Configuration ---
MODEL_PATH = 'ml/save_data/best_win_predictor.pth'
CHAMP_LIST_PATH = 'ml/save_data/champion_list.json'
EMBEDDING_DIM = 128
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# (Blue, Red, RoleIndex, Real_Lolalytics_WR)
# Roles: 0=Top, 1=Jgl, 2=Mid, 3=Bot, 4=Sup
MATCHUPS = [
    # TOP LANE
    ("Kayle", "Irelia", 0, 46.68),
    ("Malphite", "Sylas", 0, 47.00),
    ("Sion", "Singed", 0, 46.62),

    # MID LANE
    ("Kassadin", "Yone", 2, 47.84),
    ("Vex", "Katarina", 2, 52.64), # Flipped: Blue (Vex) should WIN
    ("Yasuo", "Malzahar", 2, 45.65)
]

def load_resources():
    # 1. Load Champ List
    with open(CHAMP_LIST_PATH, 'r', encoding='utf-8') as f:
        champ_dict = json.load(f)
    num_champions = len(champ_dict)

    # 2. Init Model (Flattened Architecture)
    model = WinPredictorModel(
        num_champions=num_champions,
        embedding_dim=EMBEDDING_DIM,
        num_layers=2,
        dropout=0.3
    ).to(DEVICE)

    # 3. Load Weights
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()
        print("Model loaded successfully.")
    else:
        print(f"CRITICAL: Model not found at {MODEL_PATH}")
        exit()

    return model, champ_dict

def predict_matchup(model, blue_name, red_name, role_idx, champ_dict):
    if blue_name not in champ_dict or red_name not in champ_dict:
        print(f"Error: Could not find {blue_name} or {red_name} in dictionary.")
        return 50.0

    blue_id = champ_dict[blue_name]
    red_id = champ_dict[red_name]

    blue_team = [0, 1, 2, 3, 4]
    red_team = [0, 1, 2, 3, 4]

    blue_team[role_idx] = blue_id
    red_team[role_idx] = red_id

    champ_ids = torch.tensor([blue_team + red_team], device=DEVICE)
    role_ids = torch.tensor([[0, 1, 2, 3, 4, 0, 1, 2, 3, 4]], device=DEVICE)
    team_ids = torch.tensor([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]], device=DEVICE)
    blue_side = torch.ones((1, 1), device=DEVICE)

    with torch.no_grad():
        logits = model(champ_ids, role_ids, team_ids, blue_side)
        prob = torch.sigmoid(logits).item() * 100

    return prob

if __name__ == "__main__":
    model, champ_dict = load_resources()

    print(f"\n{'Blue (You)':<12} vs {'Red (Enemy)':<12} | {'Real WR':<8} | {'Model WR':<8} | {'Diff':<6}")
    print("-" * 65)

    total_error = 0

    for blue, red, role, real_wr in MATCHUPS:
        pred_wr = predict_matchup(model, blue, red, role, champ_dict)
        diff = pred_wr - real_wr
        total_error += abs(diff)

        print(f"{blue:<12} vs {red:<12} | {real_wr:.1f}%    | {pred_wr:.1f}%    | {diff:+.1f}%")

    print("-" * 65)
    print(f"Average Error: {total_error / len(MATCHUPS):.2f}%")

    print("\nVERDICT:")
    if total_error / len(MATCHUPS) < 3.0:
        print(">> EXCELLENT. The model understands specific counters.")
    elif total_error / len(MATCHUPS) < 6.0:
        print(">> OKAY. The model sees the trend but is dampening the severity.")
    else:
        print(">> FAIL. The model is ignoring the matchup mechanics (Mean Reversion).")
