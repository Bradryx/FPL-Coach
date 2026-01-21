"""FPL helper module (public API only).

Features
- FDR uses the next N fixtures starting from (current_gw + 1).
- Optional recent-minutes penalty using /element-summary/{player_id}/ (last N matches).
- Multi-transfer planning is sequential: selling a premium can fund later upgrades.

Notes
- Sell price is approximated using current price (now_cost).
- This module is safe to import (no Streamlit code).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import time

import pandas as pd
import requests

BASE_URL = "https://fantasy.premierleague.com/api"


class FPLError(RuntimeError):
    """Raised when a public FPL API request fails."""


def _get_json(path: str, timeout: int = 20, retries: int = 3) -> dict:
    """GET JSON from an FPL API path (or full URL).

    Adds lightweight retry handling for transient failures (timeouts, 429).
    """
    url = path if path.startswith("http") else f"{BASE_URL}{path}"
    last_exc: Optional[Exception] = None

    for attempt in range(max(1, int(retries))):
        try:
            r = requests.get(url, timeout=timeout, headers={"User-Agent": "fpl-streamlit"})
            if r.status_code == 429:
                time.sleep(0.8 * (attempt + 1))
                continue
            if r.status_code == 404:
                raise FPLError(f"404 Not Found: {url}")
            r.raise_for_status()
            return r.json()
        except FPLError:
            raise
        except Exception as e:
            last_exc = e
            time.sleep(0.3 * (attempt + 1))

    raise FPLError(f"FPL request failed: {url} ({type(last_exc).__name__}: {last_exc})")


# -------------------------
# Bootstrap + fixtures
# -------------------------


@lru_cache(maxsize=1)
def load_bootstrap() -> Tuple[pd.DataFrame, pd.DataFrame]:
    data = _get_json("/bootstrap-static/")
    players_df = pd.DataFrame(data.get("elements", []))
    teams_df = pd.DataFrame(data.get("teams", []))

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
        "chance_of_playing_next_round",
    ]:
        if col not in players_df.columns:
            players_df[col] = pd.NA

    for col in ["id", "name", "short_name"]:
        if col not in teams_df.columns:
            teams_df[col] = pd.NA

    return players_df, teams_df


@lru_cache(maxsize=1)
def load_events() -> pd.DataFrame:
    data = _get_json("/bootstrap-static/")
    events_df = pd.DataFrame(data.get("events", []))
    if events_df.empty:
        events_df = pd.DataFrame(columns=["id", "name", "is_current", "is_next", "finished"])
    for col in ["id", "name", "is_current", "is_next", "finished"]:
        if col not in events_df.columns:
            events_df[col] = pd.NA
    return events_df


def get_current_gameweek() -> int:
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
            return max(1, v - 1)

    fin = events[events.get("finished") == True]  # noqa: E712
    if not fin.empty:
        v = _to_int(fin.iloc[-1].get("id"))
        if v:
            return v

    return 1


@lru_cache(maxsize=1)
def load_fixtures() -> pd.DataFrame:
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
# Entry / picks
# -------------------------


def load_entry_picks(manager_id: int, gameweek: int) -> dict:
    mid = int(manager_id)
    gw = int(gameweek)
    return _get_json(f"/entry/{mid}/event/{gw}/picks/")


def resolve_gameweek(manager_id: int, preferred_gameweek: int, max_fallbacks: int = 3) -> int:
    """Find the newest gameweek where /picks/ works for this manager."""
    gw = int(preferred_gameweek)
    for _ in range(max(0, int(max_fallbacks)) + 1):
        try:
            load_entry_picks(int(manager_id), gw)
            return gw
        except FPLError as e:
            if "404" not in str(e):
                raise
            gw = max(1, gw - 1)
    return max(1, int(preferred_gameweek))


# -------------------------
# Helpers
# -------------------------


def _safe_float(v, default: float = 0.0) -> float:
    try:
        if v is None or pd.isna(v):
            return float(default)
        if isinstance(v, str):
            v = v.strip().replace(",", ".")
        return float(v)
    except Exception:
        return float(default)


def _name(r) -> str:
    w = r.get("web_name")
    if pd.notna(w) and str(w).strip():
        return str(w)
    fn = str(r.get("first_name") or "").strip()
    sn = str(r.get("second_name") or "").strip()
    full = (fn + " " + sn).strip()
    return full if full else str(r.get("id"))


def _position_name(element_type: int) -> str:
    return {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}.get(int(element_type), str(element_type))


def _team_short_map(teams_df: pd.DataFrame) -> Dict[int, str]:
    m: Dict[int, str] = {}
    if teams_df is None or teams_df.empty:
        return m
    for _, r in teams_df.iterrows():
        try:
            m[int(r.get("id"))] = str(r.get("short_name") or r.get("name") or "")
        except Exception:
            continue
    return m


# -------------------------
# Minutes (recent matches)
# -------------------------


@lru_cache(maxsize=5000)
def load_element_summary(player_id: int) -> dict:
    return _get_json(f"/element-summary/{int(player_id)}/", retries=2)


def recent_minutes(player_id: int, last_matches: int = 2) -> Tuple[int, int]:
    """Return (minutes_sum, matches_count) over last N played matches."""
    try:
        data = load_element_summary(int(player_id))
        hist = data.get("history", []) or []
        if not hist:
            return 0, 0
        mins: List[int] = []
        for row in reversed(hist):
            mins.append(int(row.get("minutes", 0) or 0))
            if len(mins) >= int(last_matches):
                break
        return sum(mins), len(mins)
    except Exception:
        # Unknown -> don't punish
        return 0, 0


def apply_minutes_penalty(
    scored_df: pd.DataFrame,
    player_ids: List[int],
    last_matches: int,
    weight: float,
) -> pd.DataFrame:
    """Apply minutes penalty to a subset of players.

    Adds columns: minutes_last_n, minutes_games, minutes_ratio, avg_minutes.
    Updates score := score * ((1-w) + w*minutes_ratio).

    If minutes data is missing, minutes_ratio defaults to 1.0 (no penalty).
    """
    if scored_df is None or scored_df.empty:
        return scored_df

    n = max(0, int(last_matches))
    w = max(0.0, min(1.0, float(weight)))
    if n == 0 or w == 0.0 or not player_ids:
        return scored_df

    df = scored_df.copy()
    pid_set = {int(x) for x in player_ids}

    mins_sum: Dict[int, int] = {}
    mins_games: Dict[int, int] = {}
    mins_ratio: Dict[int, float] = {}
    mins_avg: Dict[int, float] = {}

    for pid in pid_set:
        s, k = recent_minutes(pid, last_matches=n)
        mins_sum[pid] = int(s)
        mins_games[pid] = int(k)
        denom = 90 * k
        r = (s / denom) if denom else 1.0
        r = max(0.0, min(1.0, float(r)))
        mins_ratio[pid] = r
        mins_avg[pid] = (s / k) if k else 90.0

    df.loc[df["id"].astype(int).isin(pid_set), "minutes_last_n"] = df["id"].astype(int).map(mins_sum)
    df.loc[df["id"].astype(int).isin(pid_set), "minutes_games"] = df["id"].astype(int).map(mins_games)
    df.loc[df["id"].astype(int).isin(pid_set), "minutes_ratio"] = df["id"].astype(int).map(mins_ratio)
    df.loc[df["id"].astype(int).isin(pid_set), "avg_minutes"] = df["id"].astype(int).map(mins_avg)

    # Default non-touched players
    df["minutes_last_n"] = df.get("minutes_last_n", pd.Series([pd.NA] * len(df)))
    df["minutes_games"] = df.get("minutes_games", pd.Series([pd.NA] * len(df)))
    df["minutes_ratio"] = df.get("minutes_ratio", pd.Series([pd.NA] * len(df)))
    df["avg_minutes"] = df.get("avg_minutes", pd.Series([pd.NA] * len(df)))

    mask = df["id"].astype(int).isin(pid_set)
    ratio = df.loc[mask, "minutes_ratio"].astype(float).fillna(1.0)
    factor = (1.0 - w) + (w * ratio)
    df.loc[mask, "score"] = df.loc[mask, "score"].astype(float) * factor

    return df


# -------------------------
# FDR (next N fixtures)
# -------------------------


def get_upcoming_fixtures(
    team_id: int,
    fixtures_df: pd.DataFrame,
    start_gameweek: int,
    fixtures_ahead: int,
) -> List[Dict]:
    """Return the next N fixtures for a team starting from start_gameweek."""
    if fixtures_df is None or fixtures_df.empty:
        return []

    t = int(team_id)
    start_gw = int(start_gameweek)
    n = max(0, int(fixtures_ahead))

    f = fixtures_df.copy()
    f = f[f["event"].notna()]
    f["event"] = f["event"].astype(int)

    f = f[(f["team_h"].astype(int) == t) | (f["team_a"].astype(int) == t)]
    f = f[f["event"] >= start_gw]
    f = f.sort_values(["event", "kickoff_time"], na_position="last")

    out: List[Dict] = []
    for _, row in f.head(n).iterrows():
        is_home = int(row["team_h"]) == t
        opp = int(row["team_a"]) if is_home else int(row["team_h"])
        diff = int(row["team_h_difficulty"]) if is_home else int(row["team_a_difficulty"])
        out.append({"gw": int(row["event"]), "opp": opp, "ha": "H" if is_home else "A", "diff": diff})
    return out


def compute_fdr_for_team(
    team_id: int,
    fixtures_df: pd.DataFrame,
    start_gameweek: int,
    fixtures_ahead: int,
) -> float:
    upcoming = get_upcoming_fixtures(team_id, fixtures_df, start_gameweek, fixtures_ahead)
    if not upcoming:
        return 3.0
    diffs = [int(x["diff"]) for x in upcoming if x.get("diff") is not None]
    return round(sum(diffs) / max(1, len(diffs)), 2)


# -------------------------
# Scoring
# -------------------------


def _availability(status: str, chance_next: Optional[float]) -> Tuple[bool, str]:
    s = (status or "").strip().lower()
    chance = _safe_float(chance_next, default=100.0)

    if s == "i":
        return False, "Injured"
    if s == "s":
        return False, "Suspended"
    if s == "u":
        return False, "Unavailable"
    if chance is not None and 0 < chance < 75:
        return True, f"Chance {int(chance)}%"
    if s in ["a", "d"]:
        return True, "Available" if s == "a" else "Doubtful"
    return True, "Available"


def _score_players_base(
    players_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
    current_gw: int,
    fixtures_ahead: int,
) -> pd.DataFrame:
    if players_df is None or players_df.empty:
        return pd.DataFrame()

    start_gw = int(current_gw) + 1  # skip current GW

    team_short = _team_short_map(teams_df)
    df = players_df.copy()

    df["name"] = df.apply(lambda r: _name(r), axis=1)
    df["team_short"] = df["team"].apply(lambda t: team_short.get(int(t), "") if pd.notna(t) else "")
    df["position"] = df["element_type"].apply(lambda x: _position_name(int(x)) if pd.notna(x) else "")
    df["price"] = df["now_cost"].apply(lambda x: round(_safe_float(x) / 10.0, 1))

    df["fdr"] = df["team"].apply(
        lambda t: compute_fdr_for_team(int(t), fixtures_df, start_gw, fixtures_ahead) if pd.notna(t) else 3.0
    )

    df["upcoming"] = df["team"].apply(
        lambda t: get_upcoming_fixtures(int(t), fixtures_df, start_gw, fixtures_ahead) if pd.notna(t) else []
    )

    ppg = df["points_per_game"].apply(lambda x: _safe_float(x))
    form = df["form"].apply(lambda x: _safe_float(x))
    epn = df["ep_next"].apply(lambda x: _safe_float(x))

    df["score"] = (ppg * 2.0 + form * 0.6 + epn * 0.7) - (df["fdr"] * 0.9)

    avail = df.apply(lambda r: _availability(str(r.get("status") or ""), r.get("chance_of_playing_next_round")), axis=1)
    df["is_available"] = [a[0] for a in avail]
    df["availability_reason"] = [a[1] for a in avail]

    return df


# -------------------------
# Views / suggestions
# -------------------------


def show_current_team(
    manager_id: int,
    gameweek: int,
    players_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
    fixtures_ahead: int = 5,
    minutes_lookback: int = 0,
    minutes_weight: float = 0.7,
) -> pd.DataFrame:
    picks = load_entry_picks(int(manager_id), int(gameweek))
    element_ids = [p["element"] for p in picks.get("picks", [])]

    scored = _score_players(players_df, teams_df, fixtures_df, current_gw=int(gameweek), fixtures_ahead=int(fixtures_ahead))
    squad = scored[scored["id"].isin(element_ids)].copy()

    # Apply minutes penalty for display/score if requested (only 15 players)
    if int(minutes_lookback) > 0:
        squad = apply_minutes_penalty(squad, last_matches=int(minutes_lookback), weight=float(minutes_weight), adjust_score=True)

    order = {eid: i for i, eid in enumerate(element_ids)}
    squad["_order"] = squad["id"].map(order)
    squad = squad.sort_values("_order").drop(columns=["_order"])

    cols = [
        "name",
        "team_short",
        "position",
        "price",
        "total_points",
        "fdr",
        "minutes_last_n",
        "avg_minutes",
        "availability_reason",
        "score",
    ]
    cols = [c for c in cols if c in squad.columns]
    return squad[cols].reset_index(drop=True)

def generate_transfer_suggestions(
    manager_id: int,
    gameweek: int,
    players_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
    top_n: int = 10,
    fixtures_ahead: int = 5,
    minutes_lookback: int = 0,
    minutes_weight: float = 0.7,
) -> pd.DataFrame:
    picks = load_entry_picks(int(manager_id), int(gameweek))
    element_ids = {p["element"] for p in picks.get("picks", [])}

    scored = _score_players(players_df, teams_df, fixtures_df, current_gw=int(gameweek), fixtures_ahead=int(fixtures_ahead))
    targets = scored[~scored["id"].isin(element_ids)].copy()
    targets = targets[targets["is_available"]]

    # Efficiency: apply minutes only to a shortlist, then re-rank
    if int(minutes_lookback) > 0 and not targets.empty:
        shortlist = targets.sort_values("score", ascending=False).head(max(50, int(top_n) * 30)).copy()
        shortlist = apply_minutes_penalty(shortlist, last_matches=int(minutes_lookback), weight=float(minutes_weight), adjust_score=True)
        targets = shortlist

    cols = ["name", "team_short", "position", "price", "total_points", "fdr", "minutes_last_n", "avg_minutes", "score"]
    cols = [c for c in cols if c in targets.columns]

    return targets.sort_values("score", ascending=False).head(int(top_n))[cols].reset_index(drop=True)

# -------------------------
# Multi-transfer planning
# -------------------------


@dataclass(frozen=True)
class _Move:
    sell_id: int
    buy_id: int
    sell_name: str
    buy_name: str
    pos: str
    sell_cost_t: int
    buy_cost_t: int
    bank_before_t: int  # bank before this step (tenths of million)
    bank_after_t: int  # bank after this step (tenths of million)
    gain: float
    priority: str
    reason: str


@dataclass
class _State:
    squad_ids: List[int]
    bank_t: int
    team_counts: Dict[int, int]
    total_gain: float
    moves: List[_Move]


def _priority_for_move(sell_row: pd.Series, gain: float) -> Tuple[str, str]:
    status = str(sell_row.get("status") or "").lower()
    chance = _safe_float(sell_row.get("chance_of_playing_next_round"), default=100.0)
    if status in ["i", "s", "u"] or chance < 50:
        return "P1", "High risk / likely out"
    if gain >= 1.5:
        return "P1", "Big score improvement"
    if gain >= 0.8:
        return "P2", "Good improvement"
    return "P3", "Minor improvement"


def suggest_transfer_plans(
    manager_id: int,
    gameweek: int,
    players_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
    fixtures_ahead: int = 5,
    num_transfers: int = 2,
    free_budget_m: Optional[float] = None,
    top_plans: int = 3,
    beam_width: int = 30,
    minutes_lookback: int = 0,
    minutes_weight: float = 0.7,
) -> pd.DataFrame:
    """Return top multi-transfer plans.

    This plans transfers sequentially, so selling a premium can fund later upgrades.

    free_budget_m:
      - If set, overrides the bank from FPL (value in millions).

    minutes_lookback / minutes_weight:
      - If enabled, penalizes players who played few minutes in recent matches (helps avoid benched/injured players).
    """
    gw = int(gameweek)
    picks = load_entry_picks(int(manager_id), gw)
    squad_ids = [int(p["element"]) for p in picks.get("picks", [])]

    bank_from_api_t = int(picks.get("entry_history", {}).get("bank", 0) or 0)  # tenths of million
    bank_t = int(round(float(free_budget_m) * 10)) if free_budget_m is not None else bank_from_api_t

    scored = _score_players(players_df, teams_df, fixtures_df, current_gw=gw, fixtures_ahead=int(fixtures_ahead))

    squad = scored[scored["id"].isin(squad_ids)].copy()
    pool_all = scored[~scored["id"].isin(squad_ids)].copy()
    pool_all = pool_all[pool_all["is_available"]]

    if squad.empty or pool_all.empty:
        return pd.DataFrame(columns=[
            "Plan",
            "Step",
            "Priority",
            "Sell",
            "Buy",
            "Pos",
            "ScoreGain",
            "SellPrice(m)",
            "BuyPrice(m)",
            "BankAfter(m)",
            "Reason",
        ])

    # Apply minutes penalty efficiently:
    # - Squad is always small (<=15)
    # - Pool is limited to a shortlist per position
    if int(minutes_lookback) > 0:
        squad = apply_minutes_penalty(squad, last_matches=int(minutes_lookback), weight=float(minutes_weight), adjust_score=True)

    # Current team counts
    team_counts: Dict[int, int] = squad["team"].dropna().astype(int).value_counts().to_dict()

    init = _State(
        squad_ids=list(squad_ids),
        bank_t=int(bank_t),
        team_counts=dict(team_counts),
        total_gain=0.0,
        moves=[],
    )

    # Candidate sell set: include premiums + low scorers + risky
    squad_sorted_exp = squad.sort_values("now_cost", ascending=False).head(8)
    squad_sorted_low = squad.sort_values("score", ascending=True).head(8)
    squad_risky = squad[~squad["is_available"]].head(8)
    sell_candidates_ids = set(squad_sorted_exp["id"].astype(int).tolist() + squad_sorted_low["id"].astype(int).tolist())
    sell_candidates_ids |= set(squad_risky["id"].astype(int).tolist())

    # Preindex rows (use penalized squad rows)
    squad_by_id = {int(r["id"]): r for _, r in squad.iterrows()}

    # Build pool per position (shortlist for minutes calls)
    pool_by_pos: Dict[int, pd.DataFrame] = {}
    per_pos_shortlist = 120  # keeps API calls reasonable
    for pos in [1, 2, 3, 4]:
        pos_df = pool_all[pool_all["element_type"].astype(int) == pos].copy().sort_values("score", ascending=False)
        pos_df = pos_df.head(per_pos_shortlist).copy()
        if int(minutes_lookback) > 0 and not pos_df.empty:
            pos_df = apply_minutes_penalty(pos_df, last_matches=int(minutes_lookback), weight=float(minutes_weight), adjust_score=True)
            pos_df = pos_df.sort_values("score", ascending=False)
        pool_by_pos[pos] = pos_df

    states: List[_State] = [init]
    steps = max(1, int(num_transfers))

    for _step in range(steps):
        new_states: List[_State] = []

        for st in states:
            current_squad_set = set(st.squad_ids)

            for sell_id in list(current_squad_set):
                if sell_id not in sell_candidates_ids and len(st.moves) == 0:
                    continue

                sell_row = squad_by_id.get(int(sell_id))
                if sell_row is None:
                    continue

                sell_team = int(sell_row.get("team")) if pd.notna(sell_row.get("team")) else -1
                sell_pos = int(sell_row.get("element_type")) if pd.notna(sell_row.get("element_type")) else 0
                sell_cost_t = int(round(_safe_float(sell_row.get("now_cost", 0)) or 0))

                bank_after_sell_t = int(st.bank_t + sell_cost_t)

                counts_after_sell = dict(st.team_counts)
                if sell_team != -1:
                    counts_after_sell[sell_team] = max(0, counts_after_sell.get(sell_team, 0) - 1)

                candidates = pool_by_pos.get(sell_pos)
                if candidates is None or candidates.empty:
                    continue

                def _ok(buy_r: pd.Series) -> bool:
                    buy_id = int(buy_r["id"])
                    if buy_id in current_squad_set:
                        return False
                    buy_cost_t = int(round(_safe_float(buy_r.get("now_cost", 0)) or 0))
                    if buy_cost_t > bank_after_sell_t:
                        return False
                    buy_team = int(buy_r.get("team")) if pd.notna(buy_r.get("team")) else -1
                    if buy_team != -1 and counts_after_sell.get(buy_team, 0) + 1 > 3:
                        return False
                    return True

                cand_ok = candidates[candidates.apply(_ok, axis=1)].head(20)
                if cand_ok.empty:
                    continue

                sell_score = _safe_float(sell_row.get("score"), 0.0)

                for _, buy_row in cand_ok.iterrows():
                    buy_id = int(buy_row["id"])
                    buy_cost_t = int(round(_safe_float(buy_row.get("now_cost", 0)) or 0))
                    gain = float(_safe_float(buy_row.get("score"), 0.0) - sell_score)

                    prio, reason = _priority_for_move(sell_row, gain)

                    bank_before_t = int(st.bank_t)
                    new_bank_t = bank_before_t + sell_cost_t - buy_cost_t
                    new_squad_ids = [x for x in st.squad_ids if int(x) != int(sell_id)] + [buy_id]

                    new_counts = dict(counts_after_sell)
                    buy_team = int(buy_row.get("team")) if pd.notna(buy_row.get("team")) else -1
                    if buy_team != -1:
                        new_counts[buy_team] = new_counts.get(buy_team, 0) + 1

                    mv = _Move(
                        sell_id=int(sell_id),
                        buy_id=buy_id,
                        sell_name=str(sell_row.get("name")),
                        buy_name=str(buy_row.get("name")),
                        pos=str(buy_row.get("position")),
                        sell_cost_t=sell_cost_t,
                        buy_cost_t=buy_cost_t,
                        bank_before_t=bank_before_t,
                        bank_after_t=new_bank_t,
                        gain=round(gain, 3),
                        priority=prio,
                        reason=reason,
                    )

                    new_states.append(
                        _State(
                            squad_ids=new_squad_ids,
                            bank_t=new_bank_t,
                            team_counts=new_counts,
                            total_gain=st.total_gain + gain,
                            moves=st.moves + [mv],
                        )
                    )

        new_states.sort(key=lambda s: s.total_gain, reverse=True)
        states = new_states[: max(1, int(beam_width))]
        if not states:
            break

    rows: List[Dict] = []
    states.sort(key=lambda s: s.total_gain, reverse=True)
    for plan_idx, st in enumerate(states[: max(1, int(top_plans))], start=1):
        for step_idx, mv in enumerate(st.moves, start=1):
            rows.append(
                {
                    "Plan": plan_idx,
                    "Step": step_idx,
                    "Priority": mv.priority,
                    "Sell": mv.sell_name,
                    "Buy": mv.buy_name,
                    "Pos": mv.pos,
                    "ScoreGain": round(mv.gain, 2),
                    "SellPrice(m)": round(mv.sell_cost_t / 10.0, 1),
                    "BuyPrice(m)": round(mv.buy_cost_t / 10.0, 1),
                    "BankBefore(m)": round(mv.bank_before_t / 10.0, 1),
                    "BankAfter(m)": round(mv.bank_after_t / 10.0, 1),
                    "Reason": mv.reason,
                }
            )

    return pd.DataFrame(rows)