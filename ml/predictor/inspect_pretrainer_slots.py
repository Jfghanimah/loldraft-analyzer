import argparse
import json
import sqlite3
from pathlib import Path

import torch
import torch.nn.functional as F

from ml.data.match_format import ROLE_ORDER
from ml.data.match_storage import connect_sqlite
from ml.predictor.models_pytorch import MLMPretrainModel
from ml.runtime_config import get_db_path, load_runtime_env


DEFAULT_PRETRAIN_PATH = Path("ml/save_data/pretrained_mlm_full.pth")
DEFAULT_CHAMPION_LIST_PATH = Path("ml/save_data/champion_list.json")
ROLE_NAME_TO_ID = {
    "top": 0,
    "jg": 1,
    "jungle": 1,
    "mid": 2,
    "middle": 2,
    "adc": 3,
    "bot": 3,
    "bottom": 3,
    "sup": 4,
    "support": 4,
    "utility": 4,
}
ROLE_IDS = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4], dtype=torch.long)
TEAM_IDS = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=torch.long)


def _load_champion_maps(champion_list_path):
    with open(champion_list_path, "r", encoding="utf-8") as f:
        name_to_id = json.load(f)
    id_to_name = {idx: name for name, idx in name_to_id.items()}
    return name_to_id, id_to_name


def _load_pretrainer(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" not in checkpoint:
        raise ValueError(
            f"{checkpoint_path} is not a full MLM checkpoint. Re-run Phase 1 after the latest update."
        )

    model = MLMPretrainModel(
        num_champions=checkpoint["num_champions"],
        embedding_dim=checkpoint["embedding_dim"],
        nhead=checkpoint.get("nhead", 4),
        dim_feedforward=checkpoint.get("dim_feedforward", 256),
        num_layers=checkpoint.get("num_layers", 3),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint["num_champions"]


def _slot_index(team, role):
    role_id = ROLE_NAME_TO_ID[role.lower()]
    return role_id if team == "blue" else 5 + role_id


def _top_predictions(logits, mask_index, id_to_name, top_k, disallow_ids=None):
    probs = F.softmax(logits[0, mask_index, :], dim=0)
    probs = probs.clone()
    invalid_ids = set(disallow_ids or [])
    invalid_ids.update(idx for idx in range(probs.size(0)) if idx not in id_to_name)
    probs[torch.tensor(sorted(invalid_ids), dtype=torch.long)] = 0.0
    probs = probs / probs.sum()

    scores, indices = torch.topk(probs, top_k)
    return [(id_to_name[idx.item()], float(score)) for score, idx in zip(scores, indices)]


def _predict_for_masked_draft(model, num_champions, champion_ids, mask_index, top_k, id_to_name):
    device = next(model.parameters()).device
    mask_token_id = num_champions
    masked = list(champion_ids)
    masked[mask_index] = mask_token_id
    input_ids = torch.tensor([masked], dtype=torch.long, device=device)
    role_ids = ROLE_IDS.unsqueeze(0).to(device)
    team_ids = TEAM_IDS.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(input_ids, role_ids, team_ids)
    return _top_predictions(logits, mask_index, id_to_name, top_k, disallow_ids=set(champion_ids))


def _load_ordered_records(conn, queue_id):
    rows = conn.execute(
        """
        SELECT ordered_match_json
        FROM matches
        WHERE ordered_match_json IS NOT NULL
          AND (? IS NULL OR queue_id = ?)
        """,
        (queue_id, queue_id),
    ).fetchall()
    return [json.loads(row[0]) for row in rows]


def _aggregate_role_substitutes(model, num_champions, ordered_records, name_to_id, id_to_name, champion, role, top_k, team):
    role_name = ROLE_ORDER[ROLE_NAME_TO_ID[role.lower()]]
    slot_index = _slot_index(team, role)
    device = next(model.parameters()).device
    mask_token_id = num_champions
    aggregate_probs = None
    contexts = 0

    for record in ordered_records:
        champions = record.get("champions")
        if not isinstance(champions, list) or len(champions) != 10:
            continue
        if champions[slot_index] != champion:
            continue

        champion_ids = [name_to_id[name] for name in champions]
        masked = champion_ids[:]
        masked[slot_index] = mask_token_id
        input_ids = torch.tensor([masked], dtype=torch.long, device=device)
        role_ids = ROLE_IDS.unsqueeze(0).to(device)
        team_ids = TEAM_IDS.unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(input_ids, role_ids, team_ids)
            probs = F.softmax(logits[0, slot_index, :], dim=0).cpu()

        invalid_ids = set(champion_ids)
        invalid_ids.update(idx for idx in range(probs.size(0)) if idx not in id_to_name)
        disallow = torch.tensor(sorted(invalid_ids), dtype=torch.long)
        probs[disallow] = 0.0
        probs = probs / probs.sum()

        aggregate_probs = probs if aggregate_probs is None else aggregate_probs + probs
        contexts += 1

    if aggregate_probs is None or contexts == 0:
        raise ValueError(f"No ordered match contexts found for {champion} in {team} {role_name.lower()}.")

    aggregate_probs = aggregate_probs / contexts
    scores, indices = torch.topk(aggregate_probs, top_k)
    return contexts, [(id_to_name[idx.item()], float(score)) for score, idx in zip(scores, indices)]


def main():
    load_runtime_env()

    parser = argparse.ArgumentParser(description="Inspect true masked-slot predictions from the Phase 1 pretrainer.")
    parser.add_argument("--top-k", type=int, default=5, help="How many predictions to print (default: 5).")
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_PRETRAIN_PATH),
        help="Path to a full MLM checkpoint produced by Phase 1.",
    )
    parser.add_argument(
        "--champion-list",
        default=str(DEFAULT_CHAMPION_LIST_PATH),
        help="Path to champion_list.json.",
    )
    parser.add_argument(
        "--draft",
        nargs=10,
        help="Exact 10-champion draft in strict top,jg,mid,adc,sup order for blue then red.",
    )
    parser.add_argument(
        "--team",
        choices=["blue", "red"],
        default="blue",
        help="Team for the masked slot or aggregate role query (default: blue).",
    )
    parser.add_argument(
        "--role",
        choices=sorted(ROLE_NAME_TO_ID.keys()),
        required=True,
        help="Role to mask or analyze.",
    )
    parser.add_argument(
        "--champion",
        help="Champion name for aggregate substitute analysis. Uses real DB contexts where this champ appears in the selected role.",
    )
    parser.add_argument(
        "--db-path",
        default=get_db_path(),
        help="SQLite DB path for aggregate context analysis.",
    )
    parser.add_argument(
        "--queue-id",
        type=int,
        default=420,
        help="Queue ID to use for aggregate context analysis (default: 420).",
    )
    args = parser.parse_args()

    if bool(args.draft) == bool(args.champion):
        raise SystemExit("Choose exactly one mode: either --draft ... or --champion <name>.")

    name_to_id, id_to_name = _load_champion_maps(args.champion_list)
    model, num_champions = _load_pretrainer(args.checkpoint)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")

    if args.draft:
        try:
            champion_ids = [name_to_id[name] for name in args.draft]
        except KeyError as exc:
            raise SystemExit(f"Unknown champion in draft: {exc}") from exc

        mask_index = _slot_index(args.team, args.role)
        results = _predict_for_masked_draft(
            model,
            num_champions,
            champion_ids,
            mask_index,
            args.top_k,
            id_to_name,
        )
        print(f"Mode: masked draft slot ({args.team} {args.role})")
        print(f"Masked original slot: {args.draft[mask_index]}")
        for name, score in results:
            print(f"  {name:<16} {score:.4f}")
        return

    conn = connect_sqlite(args.db_path, read_only=True)
    try:
        ordered_records = _load_ordered_records(conn, args.queue_id)
    finally:
        conn.close()

    contexts, results = _aggregate_role_substitutes(
        model,
        num_champions,
        ordered_records,
        name_to_id,
        id_to_name,
        args.champion,
        args.role,
        args.top_k,
        args.team,
    )
    print(f"Mode: aggregate substitutes for {args.champion} in {args.team} {args.role}")
    print(f"Contexts used: {contexts}")
    for name, score in results:
        print(f"  {name:<16} {score:.4f}")


if __name__ == "__main__":
    main()
