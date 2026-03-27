import json
import time

from riotwatcher import ApiError


SOLO_QUEUE = "RANKED_SOLO_5x5"
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def extract_solo_queue_snapshot(entries, fetched_ts):
    ranked_entry = None
    for entry in entries or []:
        if entry.get("queueType") == SOLO_QUEUE:
            ranked_entry = entry
            break

    if ranked_entry is None:
        return {
            "queue_type": SOLO_QUEUE,
            "status": "unranked",
            "snapshot_fetched_ts": fetched_ts,
            "tier": None,
            "rank": None,
            "league_points": None,
            "wins": None,
            "losses": None,
        }

    return {
        "queue_type": SOLO_QUEUE,
        "status": "ranked",
        "snapshot_fetched_ts": fetched_ts,
        "tier": ranked_entry.get("tier"),
        "rank": ranked_entry.get("rank"),
        "league_points": ranked_entry.get("leaguePoints"),
        "wins": ranked_entry.get("wins"),
        "losses": ranked_entry.get("losses"),
    }


def get_cached_rank_entry(conn, puuid):
    return conn.execute(
        """
        SELECT platform, encrypted_summoner_id, snapshot_json, fetched_ts
        FROM puuid_rank_cache
        WHERE puuid = ?
        """,
        (puuid,),
    ).fetchone()


def store_rank_snapshot(conn, puuid, platform, encrypted_summoner_id, snapshot, fetched_ts):
    conn.execute(
        """
        INSERT INTO puuid_rank_cache (
            puuid, platform, encrypted_summoner_id, snapshot_json, fetched_ts
        ) VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(puuid) DO UPDATE SET
            platform = excluded.platform,
            encrypted_summoner_id = excluded.encrypted_summoner_id,
            snapshot_json = excluded.snapshot_json,
            fetched_ts = excluded.fetched_ts
        """,
        (puuid, platform, encrypted_summoner_id, json.dumps(snapshot), fetched_ts),
    )


def _load_cached_snapshot(row):
    if row is None or not row[2]:
        return None
    return json.loads(row[2])


def _error_snapshot(participant, platform, now_ts, *, summoner_id=None, status="error", error_status_code=None, error_message=None):
    snapshot = {
        "puuid": participant["puuid"],
        "summoner_id": summoner_id,
        "platform": platform,
        "team_id": participant.get("teamId"),
        "position": participant.get("teamPosition") or participant.get("individualPosition"),
        "champion": participant.get("championName"),
        "queue_type": SOLO_QUEUE,
        "status": status,
        "snapshot_fetched_ts": now_ts,
    }
    if error_status_code is not None:
        snapshot["error_status_code"] = error_status_code
    if error_message is not None:
        snapshot["error_message"] = error_message
    return snapshot


def _fetch_snapshot_for_participant(
    conn,
    watcher,
    platform,
    participant,
    ttl_seconds,
    now_ts,
    on_api_call=None,
):
    puuid = participant["puuid"]
    cached = get_cached_rank_entry(conn, puuid)
    if cached is not None and cached[3] is not None and now_ts - int(cached[3]) <= ttl_seconds:
        snapshot = _load_cached_snapshot(cached)
        if snapshot is not None:
            return {
                "puuid": puuid,
                "summoner_id": cached[1],
                "platform": platform,
                "team_id": participant.get("teamId"),
                "position": participant.get("teamPosition") or participant.get("individualPosition"),
                "champion": participant.get("championName"),
                **snapshot,
            }

    summoner_id = cached[1] if cached is not None else None
    participant_summoner_id = participant.get("summonerId")
    if not summoner_id and participant_summoner_id:
        summoner_id = participant_summoner_id

    try:
        if not summoner_id:
            summoner = watcher.summoner.by_puuid(platform, puuid)
            if on_api_call is not None:
                on_api_call()
            summoner_id = summoner.get("id")
            if not summoner_id:
                return _error_snapshot(
                    participant,
                    platform,
                    now_ts,
                    status="missing_summoner_id",
                    error_message=f"summoner.by_puuid returned keys: {sorted(summoner.keys())}" if isinstance(summoner, dict) else f"unexpected summoner payload type: {type(summoner).__name__}",
                )

        league_entries = watcher.league.by_summoner(platform, summoner_id)
        if on_api_call is not None:
            on_api_call()
    except ApiError as exc:
        status_code = exc.response.status_code
        status = "retryable_error" if status_code in RETRYABLE_STATUS_CODES else "error"
        return _error_snapshot(
            participant,
            platform,
            now_ts,
            summoner_id=summoner_id,
            status=status,
            error_status_code=status_code,
        )

    snapshot = extract_solo_queue_snapshot(league_entries, now_ts)
    store_rank_snapshot(conn, puuid, platform, summoner_id, snapshot, now_ts)
    return {
        "puuid": puuid,
        "summoner_id": summoner_id,
        "platform": platform,
        "team_id": participant.get("teamId"),
        "position": participant.get("teamPosition") or participant.get("individualPosition"),
        "champion": participant.get("championName"),
        **snapshot,
    }


def get_or_fetch_rank_snapshots(
    conn,
    watcher,
    platform,
    ordered_participants,
    ttl_seconds,
    on_api_call=None,
):
    now_ts = int(time.time())
    snapshots = []
    for participant in ordered_participants:
        snapshots.append(
            _fetch_snapshot_for_participant(
                conn=conn,
                watcher=watcher,
                platform=platform,
                participant=participant,
                ttl_seconds=ttl_seconds,
                now_ts=now_ts,
                on_api_call=on_api_call,
            )
        )
    return snapshots
