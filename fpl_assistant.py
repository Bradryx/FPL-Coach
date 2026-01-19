"""FPL helper module (public API only).

- FDR uses the next N fixtures starting from (current_gw + 1).
- Sell price is approximated using current price (now_cost).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import requests
import time

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
                # Rate limited; wait a bit and retry.
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
# Per-player history (minutes)
# -------------------------


@lru_cache(maxsize=5000)
def load_element_summary(player_id: int) -> dict:
    """Public element summary.

    Contains match-by-match history including minutes.
    """
    return _get_json(f"/element-summary/{int(player_id)}/")


def recent_minutes(player_id: int, last_matches: int = 3) -> Tuple[int, int]:
    """Return (minutes_sum, matches_count) over the last N played matches."""
    n = max(0, int(last_matches))
    if n == 0:
        return 0, 0

    data = load_element_summary(int(player_id))
    hist = data.get("history", []) or []
    if not hist:
        return 0, 0

    mins: List[int] = []
    for row in reversed(hist):
        m = row.get("minutes", 0) or 0
        mins.append(int(m))
        if len(mins) >= n:
            break

    return int(sum(mins)), int(len(mins))


def _minutes_stats_for_ids(player_ids: Sequence[int], last_matches: int) -> Dict[int, Dict[str, float]]:
    """Fetch minutes stats for a set of ids.

    Returns dict: id -> {minutes_last_n, minutes_games, minutes_ratio, avg_minutes}
    """
    out: Dict[int, Dict[str, float]] = {}
    n = max(0, int(last_matches))
    if n == 0:
        return out

    for pid in set(int(x) for x in player_ids if x is not None):
        try:
            s, k = recent_minutes(pid, last_matches=n)
        except Exception:
            # If history fetch fails (rate limit etc.), don't punish.
            s, k = 0, 0

        denom = 90 * k if k else 0
        ratio = (float(s) / float(denom)) if denom else 1.0
        ratio = max(0.0, min(1.0, ratio))
        avg = (float(s) / float(k)) if k else 90.0
        out[pid] = {
            "minutes_last_n": float(s),
            "minutes_games": float(k),
            "minutes_ratio": float(ratio),
            "avg_minutes": float(avg),
        }
    return out


def apply_minutes_penalty(
    df: pd.DataFrame,
    player_ids: Sequence[int],
    last_matches: int,
    weight: float = 0.7,
) -> pd.DataFrame:
    """Apply a minutes-based penalty to df["score"] for the given ids.

    score *= ((1 - w) + w * minutes_ratio)
    where minutes_ratio is based on the last N played matches.
    """
    if df is None or df.empty:
        return df

    n = max(0, int(last_matches))
    if n == 0:
        return df

    w = max(0.0, min(1.0, float(weight)))
    stats = _minutes_stats_for_ids(player_ids, last_matches=n)
    if not stats:
        return df

    out_df = df.copy()
    out_df["minutes_last_n"] = out_df.get("minutes_last_n", pd.NA)
    out_df["avg_minutes"] = out_df.get("avg_minutes", pd.NA)
    out_df["minutes_ratio"] = out_df.get("minutes_ratio", pd.NA)

    id_set = set(int(x) for x in player_ids)
    for idx, r in out_df.iterrows():
        try:
            pid = int(r.get("id"))
        except Exception:
            continue
        if pid not in id_set:
            continue
        st = stats.get(pid)
        if not st:
            continue

        ratio = float(st["minutes_ratio"])
        factor = (1.0 - w) + (w * ratio)
        out_df.at[idx, "score"] = float(_safe_float(r.get("score"), 0.0)) * float(factor)
        out_df.at[idx, "minutes_last_n"] = float(st["minutes_last_n"])
        out_df.at[idx, "avg_minutes"] = float(st["avg_minutes"])
        out_df.at[idx, "minutes_ratio"] = float(st["minutes_ratio"])

    return out_df

# -------------------------
# Bootstrap + fixtures
# -------------------------

@lru_cache(maxsize=1)
def load_bootstrap() -> Tuple[pd.DataFrame, pd.DataFrame]:
    data = _get_json("/bootstrap-static/")
    players_df = pd.DataFrame(data.get("elements", []))
    teams_df = pd.DataFrame(data.get("teams", []))

    # Ensure expected columns exist
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
    """Find the newest gameweek where /picks/ works for this manager.

    This avoids crashing when you request a GW that is not available yet.
    """
    gw = int(preferred_gameweek)
    for _ in range(max(0, int(max_fallbacks)) + 1):
        try:
            load_entry_picks(int(manager_id), gw)
            return gw
        except FPLError as e:
            # Usually 404 if GW not available or entry is private.
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
        diff = int(row["team_h_difficulty"]) if is_home else int(row["team_a_difficulty"])  # 1..5
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
    # doubtful (d) is still allowed but flagged
    if chance is not None and chance > 0 and chance < 75:
        return True, f"Chance {int(chance)}%"
    if s in ["a", "d"]:
        return True, "Available" if s == "a" else "Doubtful"
    return True, "Available"


def _score_players(
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

    # FDR + upcoming
    df["fdr"] = df["team"].apply(
        lambda t: compute_fdr_for_team(int(t), fixtures_df, start_gw, fixtures_ahead) if pd.notna(t) else 3.0
    )

    df["upcoming"] = df["team"].apply(
        lambda t: get_upcoming_fixtures(int(t), fixtures_df, start_gw, fixtures_ahead) if pd.notna(t) else []
    )

    # Base features
    ppg = df["points_per_game"].apply(lambda x: _safe_float(x))
    form = df["form"].apply(lambda x: _safe_float(x))
    epn = df["ep_next"].apply(lambda x: _safe_float(x))

    # Heuristic score: reward ppg/form/ep_next, penalize FDR
    df["score"] = (ppg * 2.0 + form * 0.6 + epn * 0.7) - (df["fdr"] * 0.9)

    # Availability
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
    **_: object,
) -> pd.DataFrame:
    picks = load_entry_picks(int(manager_id), int(gameweek))
    element_ids = [p["element"] for p in picks.get("picks", [])]

    scored = _score_players(players_df, teams_df, fixtures_df, current_gw=int(gameweek), fixtures_ahead=int(fixtures_ahead))
    squad = scored[scored["id"].isin(element_ids)].copy()

    # Minutes penalty only for the squad (fast, avoids huge API fan-out)
    if int(minutes_lookback) > 0 and not squad.empty:
        squad = apply_minutes_penalty(
            squad,
            player_ids=squad["id"].astype(int).tolist(),
            last_matches=int(minutes_lookback),
            weight=float(minutes_weight),
        )

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
    **_: object,
) -> pd.DataFrame:
    picks = load_entry_picks(int(manager_id), int(gameweek))
    element_ids = {p["element"] for p in picks.get("picks", [])}

    scored = _score_players(players_df, teams_df, fixtures_df, current_gw=int(gameweek), fixtures_ahead=int(fixtures_ahead))
    targets = scored[~scored["id"].isin(element_ids)].copy()
    targets = targets[targets["is_available"]]

    # Apply minutes penalty on a limited shortlist (keeps API calls low)
    if int(minutes_lookback) > 0 and not targets.empty:
        shortlist_n = max(int(top_n) * 8, 60)
        shortlist = targets.sort_values("score", ascending=False).head(shortlist_n).copy()
        shortlist = apply_minutes_penalty(
            shortlist,
            player_ids=shortlist["id"].astype(int).tolist(),
            last_matches=int(minutes_lookback),
            weight=float(minutes_weight),
        )
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
    **_: object,
) -> pd.DataFrame:
    """Return top multi-transfer plans.

    This plans transfers sequentially, so selling a premium can fund later upgrades.

    free_budget_m:
      - If set, overrides the bank from FPL (value in millions).
    """
    gw = int(gameweek)
    picks = load_entry_picks(int(manager_id), gw)
    squad_ids = [int(p["element"]) for p in picks.get("picks", [])]

    bank_from_api_t = int(picks.get("entry_history", {}).get("bank", 0) or 0)  # tenths of million
    bank_t = int(round(float(free_budget_m) * 10)) if free_budget_m is not None else bank_from_api_t

    scored = _score_players(players_df, teams_df, fixtures_df, current_gw=gw, fixtures_ahead=int(fixtures_ahead))

    squad = scored[scored["id"].isin(squad_ids)].copy()
    pool = scored[~scored["id"].isin(squad_ids)].copy()
    pool = pool[pool["is_available"]]

    # Minutes penalty: apply to squad + a capped pool shortlist (keeps API calls low)
    if int(minutes_lookback) > 0:
        if not squad.empty:
            squad = apply_minutes_penalty(
                squad,
                player_ids=squad["id"].astype(int).tolist(),
                last_matches=int(minutes_lookback),
                weight=float(minutes_weight),
            )
        if not pool.empty:
            pool_parts = []
            for pos in [1, 2, 3, 4]:
                part = pool[pool["element_type"].astype(int) == pos].sort_values("score", ascending=False).head(120).copy()
                if not part.empty:
                    part = apply_minutes_penalty(
                        part,
                        player_ids=part["id"].astype(int).tolist(),
                        last_matches=int(minutes_lookback),
                        weight=float(minutes_weight),
                    )
                pool_parts.append(part)
            pool = pd.concat(pool_parts, ignore_index=True) if pool_parts else pool

    if squad.empty or pool.empty:
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

    # Preindex rows
    squad_by_id = {int(r["id"]): r for _, r in squad.iterrows()}
    pool_by_pos: Dict[int, pd.DataFrame] = {}
    for pos in [1, 2, 3, 4]:
        pool_by_pos[pos] = pool[pool["element_type"].astype(int) == pos].copy().sort_values("score", ascending=False)

    states: List[_State] = [init]
    steps = max(1, int(num_transfers))

    for _step in range(steps):
        new_states: List[_State] = []

        for st in states:
            current_squad_set = set(st.squad_ids)

            # Refresh squad dataframe for this state (only for sells)
            for sell_id in list(current_squad_set):
                if sell_id not in sell_candidates_ids and len(st.moves) == 0:
                    # first step: restrict sells for speed
                    continue

                sell_row = squad_by_id.get(int(sell_id))
                if sell_row is None:
                    continue

                sell_team = int(sell_row.get("team")) if pd.notna(sell_row.get("team")) else -1
                sell_pos = int(sell_row.get("element_type")) if pd.notna(sell_row.get("element_type")) else 0
                sell_cost_t = int(round(_safe_float(sell_row.get("now_cost", 0)) or 0))

                # Bank after sell (sell price approx = current price)
                bank_after_sell_t = int(st.bank_t + sell_cost_t)

                # team counts after sell
                counts_after_sell = dict(st.team_counts)
                if sell_team != -1:
                    counts_after_sell[sell_team] = max(0, counts_after_sell.get(sell_team, 0) - 1)

                candidates = pool_by_pos.get(sell_pos)
                if candidates is None or candidates.empty:
                    continue

                # Affordable + team limit + not already in squad
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

                cand_ok = candidates[candidates.apply(_ok, axis=1)].head(15)  # branch factor
                if cand_ok.empty:
                    continue

                sell_score = _safe_float(sell_row.get("score"), 0.0)

                for _, buy_row in cand_ok.iterrows():
                    buy_id = int(buy_row["id"])
                    buy_cost_t = int(round(_safe_float(buy_row.get("now_cost", 0)) or 0))
                    gain = float(_safe_float(buy_row.get("score"), 0.0) - sell_score)

                    prio, reason = _priority_for_move(sell_row, gain)

                    # Apply move
                    new_bank_t = bank_after_sell_t - buy_cost_t
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

        # Keep best beam
        new_states.sort(key=lambda s: s.total_gain, reverse=True)
        states = new_states[: max(1, int(beam_width))]
        if not states:
            break

    # Build table
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
                    "BankAfter(m)": round(st.bank_t / 10.0, 1),
                    "Reason": mv.reason,
                }
            )

    return pd.DataFrame(rows)
