"""Fantasy Premier League helper functions used by the Streamlit UI.

This module intentionally contains *no* Streamlit code so it can be imported safely.
It uses the public (no-auth) FPL endpoints.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests

BASE_URL = "https://fantasy.premierleague.com/api"


# -------------------------
# Low-level API helpers
# -------------------------

def _get_json(url: str, timeout: int = 20):
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "fpl-streamlit-assistant"})
    r.raise_for_status()
    return r.json()


@lru_cache(maxsize=1)
def load_bootstrap() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load players + teams from bootstrap-static."""
    data = _get_json(f"{BASE_URL}/bootstrap-static/")
    players_df = pd.DataFrame(data.get("elements", []))
    teams_df = pd.DataFrame(data.get("teams", []))

    # Keep common columns if present; don't fail if FPL adds/removes fields.
    for col in ["id", "web_name", "first_name", "second_name", "team", "element_type", "now_cost", "total_points", "form", "points_per_game", "ep_next", "status"]:
        if col not in players_df.columns:
            players_df[col] = pd.NA

    for col in ["id", "name", "short_name"]:
        if col not in teams_df.columns:
            teams_df[col] = pd.NA

    return players_df, teams_df


@lru_cache(maxsize=1)
def load_fixtures() -> pd.DataFrame:
    """Load all fixtures."""
    data = _get_json(f"{BASE_URL}/fixtures/")
    fixtures_df = pd.DataFrame(data)

    # Normalize expected columns
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


def load_entry_picks(manager_id: int, gameweek: int) -> dict:
    """Public endpoint for a manager's picks for a given GW."""
    return _get_json(f"{BASE_URL}/entry/{manager_id}/event/{gameweek}/picks/")


# -------------------------
# Fixture difficulty helpers
# -------------------------

def compute_fdr_for_team(
    team_id: int,
    fixtures: pd.DataFrame,
    start_gameweek: int,
    weeks_ahead: int = 5,
) -> Optional[float]:
    """Average FDR for `team_id` in a gameweek window.

    Window is inclusive: start_gameweek .. start_gameweek+weeks_ahead-1
    Only fixtures with a real event (not None/NaN) are considered.
    """
    if fixtures is None or fixtures.empty:
        return None

    end_gw = start_gameweek + max(weeks_ahead, 1) - 1
    f = fixtures.copy()
    f = f[f["event"].notna()]
    f = f[(f["event"] >= start_gameweek) & (f["event"] <= end_gw)]

    # fixtures involving this team
    mask = (f["team_h"] == team_id) | (f["team_a"] == team_id)
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
    """Return a compact string of upcoming opponents for a team."""
    if fixtures is None or fixtures.empty or teams_df is None or teams_df.empty:
        return ""

    teams_map = {
        int(row["id"]): str(row.get("short_name") or row.get("name") or row.get("id"))
        for _, row in teams_df.iterrows()
        if pd.notna(row.get("id"))
    }

    f = fixtures.copy()
    f = f[f["event"].notna()]
    f = f[f["event"] >= current_gw]
    mask = (f["team_h"] == team_id) | (f["team_a"] == team_id)
    f = f[mask]

    if f.empty:
        return ""

    # Sort by GW then kickoff_time
    if "kickoff_time" in f.columns:
        f = f.sort_values(["event", "kickoff_time"], na_position="last")
    else:
        f = f.sort_values(["event"], na_position="last")

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
    """Return players_df with extra columns: price, fdr, fixtures, score."""
    team_map = _build_team_short_map(teams_df)

    df = players_df.copy()

    df["name"] = df.apply(_player_display_name, axis=1)
    df["team_short"] = df["team"].apply(lambda t: team_map.get(int(t), str(t)) if pd.notna(t) else "")
    df["position"] = df["element_type"].apply(lambda p: POSITION_NAME.get(int(p), str(p)) if pd.notna(p) else "")
    df["price"] = df["now_cost"].apply(lambda c: _safe_float(c) / 10.0)

    # Fixture context
    def _fdr(team_id: int) -> float:
        v = compute_fdr_for_team(team_id, fixtures_df, current_gw, weeks_ahead)
        return float(v) if v is not None else 3.0

    df["fdr"] = df["team"].apply(lambda t: _fdr(int(t)) if pd.notna(t) else 3.0)
    df["upcoming"] = df["team"].apply(lambda t: get_upcoming_fixtures(int(t), fixtures_df, teams_df, current_gw, 3) if pd.notna(t) else "")

    # Base performance signals (public + widely available)
    ep_next = df["ep_next"].apply(_safe_float)
    ppg = df["points_per_game"].apply(_safe_float)
    form = df["form"].apply(_safe_float)

    # Lower FDR is better; map 1..5 into multiplier ~1.25..0.75
    fdr_mult = 1.25 - ((df["fdr"].clip(1, 5) - 1) * (0.5 / 4.0))

    # Score: combine expected points + form + ppg; normalize by price a bit
    raw = (ep_next * 0.60) + (form * 0.25) + (ppg * 0.15)
    df["score"] = (raw * fdr_mult) / (df["price"].replace(0, pd.NA)).fillna(4.5)

    # Filter out players unavailable (e.g. injured/suspended) if status exists
    if "status" in df.columns:
        df["is_available"] = df["status"].astype(str).isin(["a", "d"])  # a=available, d=doubtful
    else:
        df["is_available"] = True

    return df


def show_current_team(
    manager_id: int,
    gameweek: int,
    players_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
) -> pd.DataFrame:
    picks = load_entry_picks(manager_id, gameweek)
    element_ids = [p["element"] for p in picks.get("picks", [])]

    scored = _score_players(players_df, teams_df, fixtures_df, current_gw=gameweek)
    squad = scored[scored["id"].isin(element_ids)].copy()

    # Preserve pick order
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
    picks = load_entry_picks(manager_id, gameweek)
    element_ids = {p["element"] for p in picks.get("picks", [])}

    scored = _score_players(players_df, teams_df, fixtures_df, current_gw=gameweek)
    targets = scored[~scored["id"].isin(element_ids)].copy()

    # Prefer available players
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
    """Suggest up to `max_moves` single-player swaps.

    This is a simple heuristic (not a full optimizer):
    - only same-position swaps
    - respect max 3 players per real team
    - approximate budget using current 'now_cost' + bank from the public endpoint
    """
    picks = load_entry_picks(manager_id, gameweek)
    squad_ids = [p["element"] for p in picks.get("picks", [])]

    bank_tenths = int(picks.get("entry_history", {}).get("bank", 0))

    scored = _score_players(players_df, teams_df, fixtures_df, current_gw=gameweek, weeks_ahead=weeks_ahead)
    scored = scored[scored["is_available"]].copy()

    squad = scored[scored["id"].isin(squad_ids)].copy()
    pool = scored[~scored["id"].isin(squad_ids)].copy()

    if squad.empty or pool.empty:
        return []

    # Team counts in squad
    team_counts: Dict[int, int] = squad["team"].astype(int).value_counts().to_dict()

    moves: List[Tuple[str, str, float, float]] = []  # sell, buy, delta_m, score_gain

    # For each sell candidate, find best buy candidate
    for _, sell in squad.iterrows():
        sell_id = int(sell["id"])
        sell_team = int(sell["team"])
        sell_pos = int(sell["element_type"]) if pd.notna(sell.get("element_type")) else None
        sell_cost_t = int(round(float(sell.get("now_cost", 0)) or 0))

        if sell_pos is None:
            continue

        # Max spend in tenths
        max_buy_cost_t = sell_cost_t + bank_tenths

        # Team cap after selling
        counts_after_sell = dict(team_counts)
        counts_after_sell[sell_team] = max(0, counts_after_sell.get(sell_team, 0) - 1)

        candidates = pool[
            (pool["element_type"].astype(int) == sell_pos)
            & (pool["now_cost"].apply(lambda x: int(round(_safe_float(x))) <= max_buy_cost_t))
        ].copy()

        if candidates.empty:
            continue

        # Enforce 3-per-team
        def _team_ok(row) -> bool:
            t = int(row["team"])
            return counts_after_sell.get(t, 0) + 1 <= 3

        candidates = candidates[candidates.apply(_team_ok, axis=1)]
        if candidates.empty:
            continue

        # Choose highest score
        buy = candidates.sort_values("score", ascending=False).head(1).iloc[0]

        score_gain = float(buy["score"] - sell["score"])
        if score_gain <= 0:
            continue

        delta_m = (float(buy.get("now_cost", 0)) - float(sell.get("now_cost", 0))) / 10.0
        moves.append((_player_display_name(sell), _player_display_name(buy), float(delta_m), score_gain))

    moves.sort(key=lambda x: x[3], reverse=True)
    best = [(s, b, d) for (s, b, d, _g) in moves[: int(max_moves)]]
    return best


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
    scored = _score_players(players_df, teams_df, fixtures_df, current_gw=gameweek, weeks_ahead=weeks_ahead)
    scored = scored[scored["is_available"]].copy()

    budget_t = int(round(budget_m * 10))

    need = {1: 2, 2: 5, 3: 5, 4: 3}
    picked_ids: List[int] = []
    team_counts: Dict[int, int] = {}
    spent_t = 0

    # Sort by score, with a mild price preference
    scored = scored.sort_values(["score", "total_points"], ascending=[False, False])

    for pos, n in need.items():
        pos_df = scored[scored["element_type"].astype(int) == pos]
        for _, row in pos_df.iterrows():
            if len([pid for pid in picked_ids if int(scored.loc[scored["id"] == pid, "element_type"].iloc[0]) == pos]) >= n:
                break
            pid = int(row["id"])
            if pid in picked_ids:
                continue
            cost_t = int(round(_safe_float(row.get("now_cost"), 0.0)))
            t = int(row["team"]) if pd.notna(row.get("team")) else -1

            if team_counts.get(t, 0) >= 3:
                continue
            if spent_t + cost_t > budget_t:
                continue

            picked_ids.append(pid)
            team_counts[t] = team_counts.get(t, 0) + 1
            spent_t += cost_t

        # If we failed to fill a position, relax score ordering by allowing cheaper players
        if len([pid for pid in picked_ids if int(scored.loc[scored["id"] == pid, "element_type"].iloc[0]) == pos]) < n:
            pos_df = scored[scored["element_type"].astype(int) == pos].sort_values(["price"], ascending=True)
            for _, row in pos_df.iterrows():
                if len([pid for pid in picked_ids if int(scored.loc[scored["id"] == pid, "element_type"].iloc[0]) == pos]) >= n:
                    break
                pid = int(row["id"])
                if pid in picked_ids:
                    continue
                cost_t = int(round(_safe_float(row.get("now_cost"), 0.0)))
                t = int(row["team"]) if pd.notna(row.get("team")) else -1
                if team_counts.get(t, 0) >= 3:
                    continue
                if spent_t + cost_t > budget_t:
                    continue
                picked_ids.append(pid)
                team_counts[t] = team_counts.get(t, 0) + 1
                spent_t += cost_t

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
    """Very lightweight chip suggestion based on (double/blank) gameweek signals."""

    # Detect doubles/blanks for next GW
    next_gw = int(gameweek) + 1
    f = fixtures_df.copy()
    f = f[f["event"].notna()]
    f_next = f[f["event"] == next_gw]
    if f_next.empty:
        return None

    counts = pd.concat([f_next["team_h"], f_next["team_a"]]).value_counts()

    if (counts >= 2).any():
        return "Bench Boost"  # common play in double GWs

    # Blank GW signal: if many teams have 0 fixtures, Free Hit can help
    teams_with_fixture = set(counts.index.astype(int).tolist())
    all_teams = set(teams_df["id"].dropna().astype(int).tolist())
    blank_teams = all_teams - teams_with_fixture
    if len(blank_teams) >= 6:
        return "Free Hit"

    return None
