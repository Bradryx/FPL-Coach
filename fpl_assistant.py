"""Fantasy Premier League helper functions for the Streamlit UI.

- Safe to import (no Streamlit code).
- Uses only public FPL endpoints.
- Fails with readable errors instead of hard crashes.

Endpoints used:
- /bootstrap-static/ (players, teams, events)
- /fixtures/
- /entry/{id}/event/{gw}/picks/
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

BASE_URL = "https://fantasy.premierleague.com/api"


class FPLError(RuntimeError):
    """Raised when the FPL public API cannot be used (HTTP, parse, etc.)."""


# -------------------------
# Low-level API helpers
# -------------------------

def _get_json(path: str, timeout: int = 20, retries: int = 2) -> dict:
    """GET JSON from FPL with small retries and a stable User-Agent."""
    url = path if path.startswith("http") else f"{BASE_URL}{path}"
    last_exc: Optional[Exception] = None

    for _ in range(max(1, int(retries) + 1)):
        try:
            r = requests.get(url, timeout=timeout, headers={"User-Agent": "fpl-streamlit-assistant"})
            if r.status_code == 404:
                raise FPLError(f"404 Not Found: {url}")
            r.raise_for_status()
            return r.json()
        except FPLError:
            raise
        except Exception as e:
            last_exc = e

    raise FPLError(f"FPL request failed: {url} ({type(last_exc).__name__}: {last_exc})")


# -------------------------
# Bootstrap data
# -------------------------

@lru_cache(maxsize=1)
def load_bootstrap() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load players + teams from bootstrap-static.

    Returns:
        players_df, teams_df
    """
    data = _get_json("/bootstrap-static/")
    players_df = pd.DataFrame(data.get("elements", []))
    teams_df = pd.DataFrame(data.get("teams", []))

    # Make sure common columns exist (don\'t break if FPL changes fields)
    for col in [
        "id",
        "web_name",
        "first_name",
        "second_name",
        "team",
        "element_type",
        "now_cost",
        "total_points",
        "form",
        "points_per_game",
        "ep_next",
        "status",
    ]:
        if col not in players_df.columns:
            players_df[col] = pd.NA

    for col in ["id", "name", "short_name"]:
        if col not in teams_df.columns:
            teams_df[col] = pd.NA

    return players_df, teams_df


@lru_cache(maxsize=1)
def load_events() -> pd.DataFrame:
    """Load events (gameweeks) from bootstrap-static."""
    data = _get_json("/bootstrap-static/")
    events_df = pd.DataFrame(data.get("events", []))
    if events_df.empty:
        # Keep expected columns for safety
        events_df = pd.DataFrame(columns=["id", "name", "is_current", "is_next", "finished"])
    for col in ["id", "name", "is_current", "is_next", "finished"]:
        if col not in events_df.columns:
            events_df[col] = pd.NA
    return events_df


def get_current_gameweek() -> int:
    """Best-effort current GW.

    Priority:
    1) event where is_current==True
    2) event where is_next==True (fallback)
    3) last finished event
    4) 1
    """
    events = load_events()
    if events is None or events.empty or "id" not in events.columns:
        return 1

    def _to_int(v) -> Optional[int]:
        try:
            if pd.isna(v):
                return None
            return int(v)
        except Exception:
            return None

    cur = events[events.get("is_current") == True]  # noqa: E712
    if not cur.empty:
        v = _to_int(cur.iloc[0].get("id"))
        if v:
            return v

    nxt = events[events.get("is_next") == True]  # noqa: E712
    if not nxt.empty:
        v = _to_int(nxt.iloc[0].get("id"))
        if v:
            # In early week (before deadline) the "next" is often the playable GW.
            return max(1, v - 1)

    fin = events[events.get("finished") == True]  # noqa: E712
    if not fin.empty:
        v = _to_int(fin.iloc[-1].get("id"))
        if v:
            return v

    return 1


# -------------------------
# Fixtures
# -------------------------

@lru_cache(maxsize=1)
def load_fixtures() -> pd.DataFrame:
    """Load all fixtures."""
    data = _get_json("/fixtures/")
    fixtures_df = pd.DataFrame(data)

    for col in [
        "event",
        "team_h",
        "team_a",
        "team_h_difficulty",
        "team_a_difficulty",
        "kickoff_time",
    ]:
        if col not in fixtures_df.columns:
            fixtures_df[col] = pd.NA

    return fixtures_df


# -------------------------
# Entry / squad
# -------------------------

def load_entry_picks(manager_id: int, gameweek: int) -> dict:
    """Public endpoint for a manager\'s picks for a given GW."""
    mid = int(manager_id)
    gw = int(gameweek)
    return _get_json(f"/entry/{mid}/event/{gw}/picks/")


def resolve_gameweek(manager_id: int, preferred_gameweek: int, max_fallbacks: int = 3) -> int:
    """Try preferred GW; if unavailable (404), fall back to earlier GWs.

    This avoids Streamlit 'stops' when the user selects a GW that\'s not available
    for that manager (private team, GW not started, etc.).
    """
    gw = int(preferred_gameweek)
    for i in range(max(0, int(max_fallbacks)) + 1):
        try:
            load_entry_picks(int(manager_id), gw)
            return gw
        except FPLError as e:
            # Only fallback on 404
            if "404" not in str(e):
                raise
            gw = max(1, gw - 1)
    return max(1, int(preferred_gameweek))


# -------------------------
# Fixture difficulty helpers
# -------------------------

def compute_fdr_for_team(
    team_id: int,
    fixtures: pd.DataFrame,
    start_gameweek: int,
    weeks_ahead: int = 5,
) -> Optional[float]:
    """Average FDR for `team_id` in a GW window (inclusive)."""
    if fixtures is None or fixtures.empty:
        return None

    end_gw = int(start_gameweek) + max(int(weeks_ahead), 1) - 1
    f = fixtures.copy()
    f = f[f["event"].notna()]
    f = f[(f["event"] >= int(start_gameweek)) & (f["event"] <= end_gw)]

    mask = (f["team_h"] == int(team_id)) | (f["team_a"] == int(team_id))
    f = f[mask]
    if f.empty:
        return None

    def _row_diff(row) -> float:
        if int(row["team_h"]) == int(team_id):
            return float(row["team_h_difficulty"])
        return float(row["team_a_difficulty"])

    diffs = f.apply(_row_diff, axis=1).astype(float)
    return float(diffs.mean())


def get_upcoming_fixtures(
    team_id: int,
    fixtures: pd.DataFrame,
    teams_df: pd.DataFrame,
    current_gw: int,
    num_games: int = 5,
) -> str:
    """Compact string of upcoming opponents for a team."""
    if fixtures is None or fixtures.empty or teams_df is None or teams_df.empty:
        return ""

    teams_map = {
        int(row["id"]): str(row.get("short_name") or row.get("name") or row.get("id"))
        for _, row in teams_df.iterrows()
        if pd.notna(row.get("id"))
    }

    # NOTE: Start from the *next* GW.
    # The "current" GW often includes a fixture that has already been played,
    # which makes both the upcoming list and FDR window misleading.
    start_gw = int(current_gw) + 1

    f = fixtures.copy()
    f = f[f["event"].notna()]
    f = f[f["event"] >= start_gw]
    f = f[(f["team_h"] == int(team_id)) | (f["team_a"] == int(team_id))]

    if f.empty:
        return ""

    sort_cols = ["event"] + (["kickoff_time"] if "kickoff_time" in f.columns else [])
    f = f.sort_values(sort_cols, na_position="last")

    parts: List[str] = []
    for _, row in f.head(int(num_games)).iterrows():
        is_home = int(row["team_h"]) == int(team_id)
        opp_id = int(row["team_a"]) if is_home else int(row["team_h"])
        opp = teams_map.get(opp_id, str(opp_id))
        diff = float(row["team_h_difficulty"] if is_home else row["team_a_difficulty"])
        ha = "H" if is_home else "A"
        parts.append(f"{opp} ({ha},{int(diff)})")

    return "; ".join(parts)


# -------------------------
# Scoring + suggestions
# -------------------------

POSITION_NAME = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


def _safe_float(x, default: float = 0.0) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def _player_display_name(row: pd.Series) -> str:
    wn = str(row.get("web_name") or "").strip()
    if wn:
        return wn
    fn = str(row.get("first_name") or "").strip()
    sn = str(row.get("second_name") or "").strip()
    return (fn + " " + sn).strip() or str(row.get("id"))


def _build_team_short_map(teams_df: pd.DataFrame) -> Dict[int, str]:
    mp: Dict[int, str] = {}
    for _, r in teams_df.iterrows():
        if pd.isna(r.get("id")):
            continue
        tid = int(r["id"])
        mp[tid] = str(r.get("short_name") or r.get("name") or tid)
    return mp


def _score_players(
    players_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
    current_gw: int,
    weeks_ahead: int = 5,
) -> pd.DataFrame:
    """Players df with columns: name, team_short, position, price, fdr, upcoming, score."""
    df = players_df.copy()

    # IMPORTANT: Skip one GW when calculating fixture difficulty.
    # When you are "going to" GW22, the API/current selection often still
    # reflects GW21, and the first fixture in that window may already be played.
    # So we start the FDR window from the next GW.
    fdr_start_gw = int(current_gw) + 1

    team_map = _build_team_short_map(teams_df)

    df["name"] = df.apply(_player_display_name, axis=1)
    df["team_short"] = df["team"].apply(lambda t: team_map.get(int(t), str(t)) if pd.notna(t) else "")
    df["position"] = df["element_type"].apply(lambda p: POSITION_NAME.get(int(p), str(p)) if pd.notna(p) else "")
    df["price"] = df["now_cost"].apply(lambda c: _safe_float(c) / 10.0)

    def _fdr(team_id: int) -> float:
        v = compute_fdr_for_team(team_id, fixtures_df, int(fdr_start_gw), int(weeks_ahead))
        return float(v) if v is not None else 3.0

    df["fdr"] = df["team"].apply(lambda t: _fdr(int(t)) if pd.notna(t) else 3.0)
    df["upcoming"] = df["team"].apply(
        lambda t: get_upcoming_fixtures(int(t), fixtures_df, teams_df, int(current_gw), 3) if pd.notna(t) else ""
    )

    ep_next = df["ep_next"].apply(_safe_float)
    ppg = df["points_per_game"].apply(_safe_float)
    form = df["form"].apply(_safe_float)

    # Lower FDR is better; map 1..5 -> multiplier ~1.25..0.75
    fdr_mult = 1.25 - ((df["fdr"].clip(1, 5) - 1) * (0.5 / 4.0))

    raw = (ep_next * 0.60) + (form * 0.25) + (ppg * 0.15)
    df["score"] = (raw * fdr_mult) / (df["price"].replace(0, pd.NA)).fillna(4.5)

    # Availability
    df["is_available"] = True
    if "status" in df.columns:
        df["is_available"] = df["status"].astype(str).isin(["a", "d"])  # a=available, d=doubtful

    return df


def show_current_team(
    manager_id: int,
    gameweek: int,
    players_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
) -> pd.DataFrame:
    picks = load_entry_picks(int(manager_id), int(gameweek))
    element_ids = [p["element"] for p in picks.get("picks", [])]

    scored = _score_players(players_df, teams_df, fixtures_df, current_gw=int(gameweek))
    squad = scored[scored["id"].isin(element_ids)].copy()

    order = {eid: i for i, eid in enumerate(element_ids)}
    squad["_order"] = squad["id"].map(order)
    squad = squad.sort_values("_order").drop(columns=["_order"])

    cols = ["name", "team_short", "position", "price", "total_points", "fdr", "upcoming", "score"]
    cols = [c for c in cols if c in squad.columns]
    return squad[cols].reset_index(drop=True)


def generate_transfer_suggestions(
    manager_id: int,
    gameweek: int,
    players_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """Top target players excluding your current 15-man squad."""
    picks = load_entry_picks(int(manager_id), int(gameweek))
    element_ids = {p["element"] for p in picks.get("picks", [])}

    scored = _score_players(players_df, teams_df, fixtures_df, current_gw=int(gameweek))
    targets = scored[~scored["id"].isin(element_ids)].copy()
    targets = targets[targets["is_available"]]

    cols = ["name", "team_short", "position", "price", "total_points", "fdr", "upcoming", "score"]
    cols = [c for c in cols if c in targets.columns]

    return targets.sort_values("score", ascending=False).head(int(top_n))[cols].reset_index(drop=True)


def suggest_transfer_moves(
    manager_id: int,
    gameweek: int,
    players_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
    weeks_ahead: int = 5,
    max_moves: int = 3,
) -> List[Tuple[str, str, float]]:
    """Suggest up to `max_moves` single-player swaps (simple heuristic)."""
    picks = load_entry_picks(int(manager_id), int(gameweek))
    squad_ids = [p["element"] for p in picks.get("picks", [])]

    bank_tenths = int(picks.get("entry_history", {}).get("bank", 0) or 0)

    scored = _score_players(players_df, teams_df, fixtures_df, current_gw=int(gameweek), weeks_ahead=int(weeks_ahead))
    scored = scored[scored["is_available"]].copy()

    squad = scored[scored["id"].isin(squad_ids)].copy()
    pool = scored[~scored["id"].isin(squad_ids)].copy()

    if squad.empty or pool.empty:
        return []

    # Team counts
    team_counts: Dict[int, int] = squad["team"].dropna().astype(int).value_counts().to_dict()

    moves: List[Tuple[str, str, float, float]] = []  # sell, buy, delta_m, score_gain

    # Ensure numeric types where needed
    pool = pool[pool["element_type"].notna() & pool["now_cost"].notna()]
    squad = squad[squad["element_type"].notna() & squad["now_cost"].notna()]

    for _, sell in squad.iterrows():
        sell_team = int(sell["team"]) if pd.notna(sell.get("team")) else -1
        sell_pos = int(sell["element_type"])
        sell_cost_t = int(round(_safe_float(sell.get("now_cost", 0)) or 0))

        max_buy_cost_t = sell_cost_t + bank_tenths

        counts_after_sell = dict(team_counts)
        counts_after_sell[sell_team] = max(0, counts_after_sell.get(sell_team, 0) - 1)

        candidates = pool[
            (pool["element_type"].astype(int) == sell_pos)
            & (pool["now_cost"].apply(lambda x: int(round(_safe_float(x))) <= max_buy_cost_t))
        ].copy()

        if candidates.empty:
            continue

        def _team_ok(row) -> bool:
            t = int(row["team"]) if pd.notna(row.get("team")) else -1
            return counts_after_sell.get(t, 0) + 1 <= 3

        candidates = candidates[candidates.apply(_team_ok, axis=1)]
        if candidates.empty:
            continue

        buy = candidates.sort_values("score", ascending=False).head(1).iloc[0]

        score_gain = float(buy["score"] - sell["score"])
        if score_gain <= 0:
            continue

        delta_m = (_safe_float(buy.get("now_cost", 0)) - _safe_float(sell.get("now_cost", 0))) / 10.0
        moves.append((_player_display_name(sell), _player_display_name(buy), float(delta_m), score_gain))

    moves.sort(key=lambda x: x[3], reverse=True)
    return [(s, b, d) for (s, b, d, _g) in moves[: int(max_moves)]]


def build_wildcard_team(
    manager_id: int,
    gameweek: int,
    players_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
    budget_m: float = 100.0,
    weeks_ahead: int = 5,
) -> pd.DataFrame:
    """Build a simple 15-player wildcard squad (greedy heuristic)."""
    scored = _score_players(players_df, teams_df, fixtures_df, current_gw=int(gameweek), weeks_ahead=int(weeks_ahead))
    scored = scored[scored["is_available"]].copy()

    budget_t = int(round(float(budget_m) * 10))
    need = {1: 2, 2: 5, 3: 5, 4: 3}

    picked_ids: List[int] = []
    team_counts: Dict[int, int] = {}
    spent_t = 0

    scored = scored.sort_values(["score", "total_points"], ascending=[False, False])

    # Track picked positions without expensive dataframe lookups
    picked_pos_counts = {1: 0, 2: 0, 3: 0, 4: 0}

    def _try_pick(row) -> bool:
        nonlocal spent_t
        pid = int(row["id"])
        pos = int(row["element_type"])
        team = int(row["team"]) if pd.notna(row.get("team")) else -1
        cost_t = int(round(_safe_float(row.get("now_cost"), 0.0)))

        if pid in picked_ids:
            return False
        if picked_pos_counts.get(pos, 0) >= need.get(pos, 0):
            return False
        if team_counts.get(team, 0) >= 3:
            return False
        if spent_t + cost_t > budget_t:
            return False

        picked_ids.append(pid)
        team_counts[team] = team_counts.get(team, 0) + 1
        picked_pos_counts[pos] = picked_pos_counts.get(pos, 0) + 1
        spent_t += cost_t
        return True

    # First pass: best scores
    for _, row in scored.iterrows():
        _try_pick(row)
        if all(picked_pos_counts[p] >= need[p] for p in need):
            break

    # Second pass: if still missing, fill cheapest
    if not all(picked_pos_counts[p] >= need[p] for p in need):
        cheap = scored.sort_values(["price"], ascending=True)
        for _, row in cheap.iterrows():
            _try_pick(row)
            if all(picked_pos_counts[p] >= need[p] for p in need):
                break

    squad = scored[scored["id"].isin(picked_ids)].copy()
    cols = ["name", "team_short", "position", "price", "total_points", "fdr", "upcoming", "score"]
    cols = [c for c in cols if c in squad.columns]

    squad = squad.sort_values(["position", "score"], ascending=[True, False])[cols].reset_index(drop=True)
    squad.attrs["budget_spent_m"] = spent_t / 10.0
    squad.attrs["budget_left_m"] = (budget_t - spent_t) / 10.0
    return squad


def suggest_chip_play(
    manager_id: int,
    gameweek: int,
    players_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
) -> Optional[str]:
    """Light chip suggestion based on (double/blank) gameweek signals."""
    next_gw = int(gameweek) + 1
    f = fixtures_df.copy()
    f = f[f["event"].notna()]
    f_next = f[f["event"] == next_gw]
    if f_next.empty:
        return None

    counts = pd.concat([f_next["team_h"], f_next["team_a"]]).value_counts()

    if (counts >= 2).any():
        return "Bench Boost"

    teams_with_fixture = set(counts.index.astype(int).tolist())
    all_teams = set(teams_df["id"].dropna().astype(int).tolist())
    blank_teams = all_teams - teams_with_fixture
    if len(blank_teams) >= 6:
        return "Free Hit"

    return None
