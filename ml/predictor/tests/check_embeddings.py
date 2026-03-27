import torch
import torch.nn as nn
import json
import os

# --- Configuration ---
EMBEDDING_DIM = 128
EMBEDDING_PATH = 'ml/save_data/pretrained_champ_embeddings.pth'
CHAMP_LIST_PATH = 'ml/save_data/champion_list.json'

def load_data():
    with open(CHAMP_LIST_PATH, 'r', encoding='utf-8') as f:
        name_to_id = json.load(f)
    id_to_name = {v: k for k, v in name_to_id.items()}
    return name_to_id, id_to_name

def get_similar_champs(target_champ_name, embedding_matrix, name_to_id, id_to_name, top_k=5):
    if target_champ_name not in name_to_id:
        print(f"Error: {target_champ_name} not found in champion list.")
        return

    target_id = name_to_id[target_champ_name]
    target_vec = embedding_matrix[target_id].unsqueeze(0)

    norm_matrix = torch.nn.functional.normalize(embedding_matrix, p=2, dim=1)
    norm_target = torch.nn.functional.normalize(target_vec, p=2, dim=1)

    similarities = torch.mm(norm_target, norm_matrix.t()).squeeze(0)

    scores, indices = torch.topk(similarities, top_k + 1)

    print(f"\n--- Champions similar to {target_champ_name} ---")
    for score, idx in zip(scores, indices):
        idx = idx.item()
        name = id_to_name.get(idx, "Unknown")
        if name == target_champ_name: continue
        print(f"{name}: {score:.4f}")

if __name__ == "__main__":
    name_to_id, id_to_name = load_data()
    num_champs = len(name_to_id)

    vocab_size = num_champs + 2
    emb_layer = nn.Embedding(vocab_size, EMBEDDING_DIM)

    state_dict = torch.load(EMBEDDING_PATH)
    emb_layer.load_state_dict(state_dict)

    weights = emb_layer.weight.detach()

    test_champs = ["Yasuo", "Lulu", "Malphite", "Ezreal", "Zed", "Jinx"]

    print("Checking Embedding Logic...")
    for champ in test_champs:
        get_similar_champs(champ, weights, name_to_id, id_to_name)
