"""Fantasy Premier League helper functions for a Streamlit UI.

Safe to import (no Streamlit code).

Public endpoints used:
- /bootstrap-static/ (players, teams, events)
- /fixtures/
- /entry/{id}/event/{gw}/picks/

Behavior:
- FDR is computed over the next N *fixtures* (works with blanks/doubles).
- By default, fixtures/FDR start at (gameweek + 1) to skip the current GW.

This is a best-effort assistant, not a perfect optimizer.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Tuple

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


# -------------------------
# Fixtures
# -------------------------

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
# Entry / squad
# -------------------------

def load_entry_picks(manager_id: int, gameweek: int) -> dict:
    mid = int(manager_id)
    gw = int(gameweek)
    return _get_json(f"/entry/{mid}/event/{gw}/picks/")


def resolve_gameweek(manager_id: int, preferred_gameweek: int, max_fallbacks: int = 3) -> int:
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

def _to_float(v, default: float = 0.0) -> float:
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
    f["event"] = pd.to_numeric(f["event"], errors="coerce")
    f = f[f["event"] >= start_gw]
    f = f[(f["team_h"] == t) | (f["team_a"] == t)]

    # kickoff_time sorts postponed (NaT) last
    f["kickoff_time"] = pd.to_datetime(f["kickoff_time"], errors="coerce", utc=True)
    f = f.sort_values(["event", "kickoff_time"], ascending=[True, True])

    out: List[Dict] = []
    for _, r in f.iterrows():
        is_home = int(r.get("team_h")) == t
        opp = int(r.get("team_a")) if is_home else int(r.get("team_h"))
        diff = _to_float(r.get("team_h_difficulty" if is_home else "team_a_difficulty"), 3.0)
        out.append({"gw": int(r.get("event")), "home": bool(is_home), "opponent": opp, "difficulty": diff})
        if len(out) >= n:
            break
    return out


def compute_fdr_for_team(
    team_id: int,
    fixtures_df: pd.DataFrame,
    start_gameweek: int,
    fixtures_ahead: int,
) -> float:
    fixtures = get_upcoming_fixtures(team_id, fixtures_df, start_gameweek, fixtures_ahead)
    if not fixtures:
        return 3.0
    diffs = [float(x.get("difficulty", 3.0)) for x in fixtures]
    return float(sum(diffs) / len(diffs))


def player_score(player_row, fdr_avg: float) -> float:
    """Heuristic score; higher is better."""
    ep = _to_float(player_row.get("ep_next"), 0.0)
    form = _to_float(player_row.get("form"), 0.0)
    ppg = _to_float(player_row.get("points_per_game"), 0.0)
    status = str(player_row.get("status", "")).lower().strip()

    score = (1.0 * ep) + (0.30 * form) + (0.20 * ppg) - (0.70 * float(fdr_avg))

    cop = _to_float(player_row.get("chance_of_playing_next_round"), 100.0)
    if cop and cop < 75:
        score -= 1.0
    if status != "a":
        score -= 1.5

    return float(score)


# -------------------------
# Team view
# -------------------------

def show_current_team(
    manager_id: int,
    gameweek: int,
    players_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
    fixtures_ahead: int = 5,
    bank_override_m: Optional[float] = None,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    gw = resolve_gameweek(int(manager_id), int(gameweek))
    data = load_entry_picks(int(manager_id), int(gw))

    picks = data.get("picks", []) or []
    entry_hist = data.get("entry_history", {}) or {}

    bank_10 = int(entry_hist.get("bank", 0) or 0)  # 0.1m units
    value_10 = int(entry_hist.get("value", 0) or 0)  # 0.1m units

    if bank_override_m is not None:
        bank_10 = max(0, int(round(float(bank_override_m) * 10)))

    team_short = _team_short_map(teams_df)
    players_idx = players_df.set_index("id", drop=False)

    start_gw = int(gw) + 1

    rows = []
    for p in picks:
        pid = int(p.get("element"))
        if pid not in players_idx.index:
            continue
        pr = players_idx.loc[pid]

        team_id = int(_to_float(pr.get("team"), 0))
        fdr = compute_fdr_for_team(team_id, fixtures_df, start_gw, int(fixtures_ahead))
        sc = player_score(pr, fdr)

        sell_10 = p.get("selling_price")
        if sell_10 is None or pd.isna(sell_10):
            sell_10 = pr.get("now_cost")
        sell_10 = int(_to_float(sell_10, 0))

        rows.append(
            {
                "Player": _name(pr),
                "Pos": _position_name(int(_to_float(pr.get("element_type"), 0))),
                "Team": team_short.get(team_id, str(team_id)),
                "Status": str(pr.get("status", "")),
                "Price": _to_float(pr.get("now_cost"), 0) / 10.0,
                "Sell": sell_10 / 10.0,
                "EP_next": _to_float(pr.get("ep_next"), 0.0),
                "Form": _to_float(pr.get("form"), 0.0),
                "FDR_avg": round(float(fdr), 2),
                "Score": round(float(sc), 3),
            }
        )

    squad_df = pd.DataFrame(rows)
    if not squad_df.empty:
        squad_df["_risk"] = (squad_df["Status"].astype(str).str.lower() != "a").astype(int)
        squad_df = squad_df.sort_values(["_risk", "Score"], ascending=[False, True]).drop(columns=["_risk"]).reset_index(
            drop=True
        )

    budget_info = {
        "GW": str(gw),
        "Bank (m)": f"{bank_10/10.0:.1f}" + (" (override)" if bank_override_m is not None else ""),
        "Squad value (m)": f"{value_10/10.0:.1f}",
        "FDR start GW": str(start_gw),
        "Fixtures window": str(int(fixtures_ahead)),
    }

    return squad_df, budget_info


# -------------------------
# Transfer suggestions
# -------------------------

def _team_counts(player_ids: Iterable[int], players_idx: pd.DataFrame) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for pid in player_ids:
        if pid not in players_idx.index:
            continue
        t = int(_to_float(players_idx.loc[pid].get("team"), 0))
        counts[t] = counts.get(t, 0) + 1
    return counts


def _priority(sell_status: str, score_gain: float) -> str:
    s = str(sell_status).lower().strip()
    if s and s != "a":
        return "P1"
    if score_gain >= 2.0:
        return "P1"
    if score_gain >= 1.0:
        return "P2"
    return "P3"


def suggest_transfer_moves(
    manager_id: int,
    gameweek: int,
    players_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
    fixtures_ahead: int = 5,
    bank_override_m: Optional[float] = None,
    transfers_wanted: int = 1,
    top_candidates: int = 80,
) -> List[Dict]:
    """Suggest 1..N transfers.

    Args:
        bank_override_m: free budget override in millions. If None, uses API bank.
        transfers_wanted: how many transfers to attempt (greedy)
        top_candidates: how many buy candidates to consider per position

    Returns:
        List of dict rows: priority, sell, buy, score_gain, budget_after, reason
    """
    gw = resolve_gameweek(int(manager_id), int(gameweek))
    data = load_entry_picks(int(manager_id), int(gw))
    picks = data.get("picks", []) or []
    entry_hist = data.get("entry_history", {}) or {}

    players_idx = players_df.set_index("id", drop=False)
    team_short = _team_short_map(teams_df)

    # Current cash
    bank_10 = int(entry_hist.get("bank", 0) or 0)
    if bank_override_m is not None:
        bank_10 = max(0, int(round(float(bank_override_m) * 10)))

    # Squad + sell prices
    squad_ids = [int(p.get("element")) for p in picks if p.get("element") is not None]
    sell_prices_10: Dict[int, int] = {}
    for p in picks:
        pid = int(p.get("element"))
        sp = p.get("selling_price")
        if sp is None or pd.isna(sp):
            if pid in players_idx.index:
                sp = players_idx.loc[pid].get("now_cost")
        sell_prices_10[pid] = int(_to_float(sp, 0))

    start_gw = int(gw) + 1
    fixtures_ahead = int(max(1, fixtures_ahead))

    # Precompute score/FDR for all players once
    all_players = players_df.copy()
    all_players["team_id"] = pd.to_numeric(all_players["team"], errors="coerce").fillna(0).astype(int)
    all_players["fdr"] = all_players["team_id"].apply(lambda t: compute_fdr_for_team(int(t), fixtures_df, start_gw, fixtures_ahead))
    all_players["score"] = all_players.apply(lambda r: player_score(r, float(r.get("fdr", 3.0))), axis=1)
    all_players["price_10"] = pd.to_numeric(all_players["now_cost"], errors="coerce").fillna(0).astype(int)

    # Available buys: exclude unavailable status
    all_players["status_s"] = all_players["status"].astype(str).str.lower().str.strip()
    buy_pool = all_players[all_players["status_s"] == "a"].copy()

    # Team counts (max 3)
    team_counts = _team_counts(squad_ids, players_idx)

    # Map owned for fast exclusion
    owned = set(squad_ids)

    chosen_moves: List[Dict] = []
    sold_ids: set = set()
    bought_ids: set = set()

    def _sell_rows() -> pd.DataFrame:
        rows = []
        for pid in squad_ids:
            if pid in sold_ids:
                continue
            if pid not in players_idx.index:
                continue
            pr = players_idx.loc[pid]
            team_id = int(_to_float(pr.get("team"), 0))
            fdr = compute_fdr_for_team(team_id, fixtures_df, start_gw, fixtures_ahead)
            sc = player_score(pr, fdr)
            status = str(pr.get("status", ""))
            pos = int(_to_float(pr.get("element_type"), 0))
            rows.append(
                {
                    "id": pid,
                    "pos": pos,
                    "team": team_id,
                    "status": status,
                    "score": sc,
                    "sell_10": sell_prices_10.get(pid, int(_to_float(pr.get("now_cost"), 0))),
                }
            )
        if not rows:
            return pd.DataFrame(columns=["id", "pos", "team", "status", "score", "sell_10"])
        df = pd.DataFrame(rows)
        df["risk"] = (df["status"].astype(str).str.lower().str.strip() != "a").astype(int)
        return df.sort_values(["risk", "score"], ascending=[False, True]).drop(columns=["risk"]).reset_index(drop=True)

    def _team_name(team_id: int) -> str:
        return team_short.get(int(team_id), str(team_id))

    def _player_label(pid: int) -> str:
        if pid not in players_idx.index:
            return str(pid)
        r = players_idx.loc[pid]
        t = int(_to_float(r.get("team"), 0))
        return f"{_name(r)} ({_team_name(t)})"

    # Greedy: pick best improvement each step
    for _ in range(max(1, int(transfers_wanted))):
        sell_df = _sell_rows()
        if sell_df.empty:
            break

        best = None

        for _, srow in sell_df.iterrows():
            sell_id = int(srow["id"])
            pos = int(srow["pos"])
            sell_team = int(srow["team"])
            sell_status = str(srow["status"])
            sell_score = float(srow["score"])
            sell_10 = int(srow["sell_10"])

            available_10 = bank_10 + sell_10

            # Predict team cap after selling
            team_counts_after_sell = dict(team_counts)
            team_counts_after_sell[sell_team] = max(0, team_counts_after_sell.get(sell_team, 0) - 1)

            candidates = buy_pool[buy_pool["element_type"].astype(int) == pos]
            # Basic pruning
            candidates = candidates[~candidates["id"].isin(owned)]
            candidates = candidates[~candidates["id"].isin(bought_ids)]
            candidates = candidates[candidates["price_10"] <= available_10]

            # Team cap pruning
            def _ok_team(team_id: int) -> bool:
                return team_counts_after_sell.get(int(team_id), 0) < 3

            candidates = candidates[candidates["team_id"].apply(_ok_team)]

            if candidates.empty:
                continue

            candidates = candidates.sort_values(["score", "total_points"], ascending=[False, False]).head(int(top_candidates))

            # Best candidate for this sell
            crow = candidates.iloc[0]
            buy_id = int(crow["id"])
            buy_score = float(crow["score"])
            buy_10 = int(crow["price_10"])

            gain = buy_score - sell_score
            if gain <= 0:
                continue

            budget_after_10 = bank_10 + sell_10 - buy_10

            # pick best move overall
            if (best is None) or (gain > best["score_gain"]):
                best = {
                    "sell_id": sell_id,
                    "buy_id": buy_id,
                    "sell": _player_label(sell_id),
                    "buy": f"{_name(crow)} ({_team_name(int(crow['team_id']))})",
                    "score_gain": round(float(gain), 3),
                    "delta_price_m": round((buy_10 - sell_10) / 10.0, 2),
                    "budget_after_m": round(budget_after_10 / 10.0, 2),
                    "priority": _priority(sell_status, float(gain)),
                    "reason": "" if str(sell_status).lower().strip() == "a" else "Replace flagged player",
                }

        if best is None:
            break

        # Apply chosen transfer
        sell_id = int(best["sell_id"])
        buy_id = int(best["buy_id"])

        # update state
        sold_ids.add(sell_id)
        bought_ids.add(buy_id)
        owned.discard(sell_id)
        owned.add(buy_id)

        # Update budget
        bank_10 = int(round(best["budget_after_m"] * 10))

        # Update team counts
        if sell_id in players_idx.index:
            sell_team = int(_to_float(players_idx.loc[sell_id].get("team"), 0))
            team_counts[sell_team] = max(0, team_counts.get(sell_team, 0) - 1)
        buy_team = int(_to_float(all_players.set_index('id').loc[buy_id].get("team"), 0))
        team_counts[buy_team] = team_counts.get(buy_team, 0) + 1

        # Better reason if it is a clear fixtures upgrade
        if not best["reason"]:
            best["reason"] = "Upgrade score (form/EP/FDR)"

        chosen_moves.append(
            {
                "Priority": best["priority"],
                "Sell": best["sell"],
                "Buy": best["buy"],
                "Score gain": best["score_gain"],
                "Delta price (m)": best["delta_price_m"],
                "Budget after (m)": best["budget_after_m"],
                "Reason": best["reason"],
            }
        )

    # Priority first, then gain
    chosen_moves.sort(key=lambda d: (d["Priority"], -float(d["Score gain"])))
    return chosen_moves


# Backwards compatible placeholder

def generate_transfer_suggestions(*args, **kwargs) -> pd.DataFrame:
    """Deprecated in current UI; kept for compatibility."""
    return pd.DataFrame()


def build_wildcard_team(*args, **kwargs) -> pd.DataFrame:
    """Simple placeholder; not used in the current UI."""
    return pd.DataFrame()


def suggest_chip_play(*args, **kwargs) -> Optional[str]:
    """Chip suggestion is out of scope here; return None."""
    return None
