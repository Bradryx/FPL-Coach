"""Fantasy Premier League helper functions for a Streamlit UI.

Public endpoints used:
- /bootstrap-static/ (players, teams, events)
- /fixtures/
- /entry/{id}/event/{gw}/picks/

Key behavior:
- FDR is computed over the next N *fixtures* (handles blanks/double GWs).
- By default we skip the current GW and start from (current_gw + 1) because the
  first "upcoming" fixture in the current GW is often already played.

Limitations (public API):
- We do not have each player's exact sell price (profit rules require auth). We
  approximate by using the player's current price (now_cost).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

BASE_URL = "https://fantasy.premierleague.com/api"


class FPLError(RuntimeError):
    pass


# -------------------------
# Low-level API helpers
# -------------------------


def _get_json(path: str, timeout: int = 20, retries: int = 2) -> dict:
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
    """If the selected GW is not available for an entry, walk backwards a few GWs."""
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
# Fixture difficulty helpers
# -------------------------


def _safe_float(x, default: float = 0.0) -> float:
    try:
        if pd.isna(x):
            return default
        if isinstance(x, str):
            x = x.strip().replace(",", ".")
        return float(x)
    except Exception:
        return default


def compute_fdr_for_team(
    team_id: int,
    fixtures: pd.DataFrame,
    start_gameweek: int,
    fixtures_ahead: int = 5,
) -> Optional[float]:
    """Average FDR for `team_id` over the next N fixtures (fixture-count based)."""
    if fixtures is None or fixtures.empty:
        return None

    n = max(1, int(fixtures_ahead))

    f = fixtures.copy()
    f = f[f["event"].notna()]
    f = f[f["event"] >= int(start_gameweek)]
    f = f[(f["team_h"] == int(team_id)) | (f["team_a"] == int(team_id))]
    if f.empty:
        return None

    sort_cols = ["event"] + (["kickoff_time"] if "kickoff_time" in f.columns else [])
    f = f.sort_values(sort_cols, na_position="last").head(n)

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
    """Compact string of upcoming opponents for a team (starts at current_gw + 1)."""
    if fixtures is None or fixtures.empty or teams_df is None or teams_df.empty:
        return ""

    teams_map = {
        int(row["id"]): str(row.get("short_name") or row.get("name") or row.get("id"))
        for _, row in teams_df.iterrows()
        if pd.notna(row.get("id"))
    }

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
# Scoring
# -------------------------


POSITION_NAME = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


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
    fixtures_ahead: int = 5,
    skip_current_gw: bool = True,
) -> pd.DataFrame:
    """Return players df with helper columns used by the UI and planners.

    - expected_points: points-like signal (NOT normalized by price)
    - score: value-like signal (normalized by price)
    """
    df = players_df.copy()

    fdr_start_gw = int(current_gw) + (1 if skip_current_gw else 0)

    team_map = _build_team_short_map(teams_df)
    df["name"] = df.apply(_player_display_name, axis=1)
    df["team_short"] = df["team"].apply(lambda t: team_map.get(int(t), str(t)) if pd.notna(t) else "")
    df["position"] = df["element_type"].apply(lambda p: POSITION_NAME.get(int(p), str(p)) if pd.notna(p) else "")
    df["price"] = df["now_cost"].apply(lambda c: _safe_float(c) / 10.0)

    def _fdr(team_id: int) -> float:
        v = compute_fdr_for_team(team_id, fixtures_df, int(fdr_start_gw), int(fixtures_ahead))
        return float(v) if v is not None else 3.0

    df["fdr"] = df["team"].apply(lambda t: _fdr(int(t)) if pd.notna(t) else 3.0)
    df["upcoming"] = df["team"].apply(
        lambda t: get_upcoming_fixtures(int(t), fixtures_df, teams_df, int(current_gw), min(6, int(fixtures_ahead)))
        if pd.notna(t)
        else ""
    )

    ep_next = df["ep_next"].apply(_safe_float)
    ppg = df["points_per_game"].apply(_safe_float)
    form = df["form"].apply(_safe_float)

    # Lower FDR is better; map 1..5 -> multiplier ~1.25..0.75
    fdr_mult = 1.25 - ((df["fdr"].clip(1, 5) - 1) * (0.5 / 4.0))

    raw = (ep_next * 0.60) + (form * 0.25) + (ppg * 0.15)
    df["expected_points"] = raw * fdr_mult

    # Value-like score (kept for the existing tables)
    df["score"] = df["expected_points"] / (df["price"].replace(0, pd.NA)).fillna(4.5)

    # Availability
    df["is_available"] = True
    if "status" in df.columns:
        df["is_available"] = df["status"].astype(str).isin(["a", "d"])  # a=available, d=doubtful

    return df


# -------------------------
# UI helpers
# -------------------------


def show_current_team(
    manager_id: int,
    gameweek: int,
    players_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
    fixtures_ahead: int = 5,
) -> pd.DataFrame:
    picks = load_entry_picks(int(manager_id), int(gameweek))
    element_ids = [p["element"] for p in picks.get("picks", [])]

    scored = _score_players(players_df, teams_df, fixtures_df, current_gw=int(gameweek), fixtures_ahead=int(fixtures_ahead))
    squad = scored[scored["id"].isin(element_ids)].copy()

    order = {eid: i for i, eid in enumerate(element_ids)}
    squad["_order"] = squad["id"].map(order)

    cols = ["name", "team_short", "position", "price", "total_points", "fdr", "upcoming", "score", "status"]
    cols = [c for c in cols if c in squad.columns]
    return squad.sort_values("_order")[cols].reset_index(drop=True)


def generate_transfer_suggestions(
    manager_id: int,
    gameweek: int,
    players_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
    fixtures_ahead: int = 5,
    top_n: int = 20,
) -> pd.DataFrame:
    """Top transfer targets (not in squad), ranked by our score."""
    picks = load_entry_picks(int(manager_id), int(gameweek))
    element_ids = [p["element"] for p in picks.get("picks", [])]

    scored = _score_players(players_df, teams_df, fixtures_df, current_gw=int(gameweek), fixtures_ahead=int(fixtures_ahead))
    targets = scored[~scored["id"].isin(element_ids)].copy()
    targets = targets[targets["is_available"]]

    cols = ["name", "team_short", "position", "price", "total_points", "fdr", "upcoming", "score"]
    cols = [c for c in cols if c in targets.columns]
    return targets.sort_values("score", ascending=False).head(int(top_n))[cols].reset_index(drop=True)


# -------------------------
# Transfer planning
# -------------------------


def _priority(sell: pd.Series, buy: pd.Series, score_gain: float, delta_m: float) -> Tuple[str, str]:
    """Return (priority, reason)."""
    chance = _safe_float(sell.get("chance_of_playing_next_round"), default=100.0)
    status = str(sell.get("status") or "").lower().strip()

    injured_flag = (status not in ["a", "d"]) or (chance and chance < 75)
    # score_gain is in "expected_points" units, typically ~0..3
    big_gain = score_gain >= 1.0
    frees_cash = delta_m <= -1.0

    if injured_flag:
        pr = "P1"
        reason = "Beschikbaarheid risico (injury/rotation)."
    elif big_gain:
        pr = "P1"
        reason = "Grote upgrade in score."
    elif frees_cash and score_gain >= 0.4:
        pr = "P2"
        reason = "Budget enabler die ook punten helpt."
    elif score_gain >= 0.5:
        pr = "P2"
        reason = "Solide upgrade in score."
    else:
        pr = "P3"
        reason = "Kleine upgrade / vooral optimalisatie."

    # Add a short quantitative note
    note = f"Gain {score_gain:+.2f}, budget {delta_m:+.1f}m"
    return pr, f"{reason} ({note})"


def _team_counts_from_ids(scored: pd.DataFrame, ids: List[int]) -> Dict[int, int]:
    sub = scored[scored["id"].isin(ids)].copy()
    if sub.empty:
        return {}
    return sub["team"].dropna().astype(int).value_counts().to_dict()


def suggest_transfer_plan(
    manager_id: int,
    gameweek: int,
    players_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
    fixtures_ahead: int = 5,
    num_transfers: int = 2,
    bank_override_m: Optional[float] = None,
    top_plans: int = 3,
    beam_width: int = 40,
    buy_pool_per_pos: int = 40,
) -> List[pd.DataFrame]:
    """Return up to top_plans transfer sequences.

    Important: we allow sequences where an early downgrade frees budget for later
    upgrades (e.g., selling a premium to fund two strong moves).
    """

    num_transfers = max(1, int(num_transfers))
    top_plans = max(1, int(top_plans))

    picks = load_entry_picks(int(manager_id), int(gameweek))
    squad_ids = [p["element"] for p in picks.get("picks", [])]
    bank_tenths = int(picks.get("entry_history", {}).get("bank", 0) or 0)
    bank_m = bank_tenths / 10.0
    if bank_override_m is not None:
        bank_m = float(bank_override_m)

    scored = _score_players(players_df, teams_df, fixtures_df, current_gw=int(gameweek), fixtures_ahead=int(fixtures_ahead))
    scored = scored[scored["is_available"]].copy()

    squad = scored[scored["id"].isin(squad_ids)].copy()
    pool = scored[~scored["id"].isin(squad_ids)].copy()
    if squad.empty or pool.empty:
        return []

    # Build a strong buy pool per position.
    # IMPORTANT: rank on expected_points (not value "score"), otherwise the planner
    # over-prefers cheap 4-5m players.
    pool = pool[pool["element_type"].notna() & pool["now_cost"].notna()].copy()
    pool["now_cost_t"] = pool["now_cost"].apply(lambda x: int(round(_safe_float(x, 0.0))))
    pool_by_pos: Dict[int, pd.DataFrame] = {}
    for pos in [1, 2, 3, 4]:
        pos_df = pool[pool["element_type"].astype(int) == pos].sort_values("expected_points", ascending=False)
        pool_by_pos[pos] = pos_df.head(int(buy_pool_per_pos)).copy()

    squad = squad[squad["element_type"].notna() & squad["now_cost"].notna()].copy()
    squad["now_cost_t"] = squad["now_cost"].apply(lambda x: int(round(_safe_float(x, 0.0))))

    # Candidate sell list: include both (a) worst scores and (b) highest prices
    worst = squad.sort_values("expected_points", ascending=True).head(8)
    expensive = squad.sort_values("now_cost_t", ascending=False).head(6)
    sell_candidates = pd.concat([worst, expensive]).drop_duplicates(subset=["id"])

    # Beam search state
    # state = (total_gain, bank_tenths, squad_ids_set, team_counts, moves_list)
    start_bank_t = int(round(bank_m * 10))
    start_team_counts = _team_counts_from_ids(scored, squad_ids)
    states = [(0.0, start_bank_t, set(squad_ids), start_team_counts, [])]

    def _team_ok(team_counts: Dict[int, int], buy_team: int) -> bool:
        return team_counts.get(buy_team, 0) + 1 <= 3

    for step in range(num_transfers):
        next_states = []
        for total_gain, bank_t, cur_ids, team_counts, moves in states:
            cur_squad = scored[scored["id"].isin(list(cur_ids))].copy()
            if cur_squad.empty:
                continue

            # Recompute sell candidates within current state
            cur_squad = cur_squad[cur_squad["element_type"].notna() & cur_squad["now_cost"].notna()].copy()
            cur_squad["now_cost_t"] = cur_squad["now_cost"].apply(lambda x: int(round(_safe_float(x, 0.0))))
            worst_s = cur_squad.sort_values("expected_points", ascending=True).head(8)
            expensive_s = cur_squad.sort_values("now_cost_t", ascending=False).head(6)
            sells = pd.concat([worst_s, expensive_s]).drop_duplicates(subset=["id"]).to_dict("records")

            for sell in sells:
                sell_id = int(sell["id"])
                sell_team = int(sell["team"]) if pd.notna(sell.get("team")) else -1
                sell_pos = int(sell["element_type"])
                sell_cost_t = int(sell.get("now_cost_t") or 0)

                # After selling, you free up sell_cost_t in budget terms
                max_buy_cost_t = bank_t + sell_cost_t

                counts_after_sell = dict(team_counts)
                counts_after_sell[sell_team] = max(0, counts_after_sell.get(sell_team, 0) - 1)

                candidates = pool_by_pos.get(sell_pos)
                if candidates is None or candidates.empty:
                    continue

                # Filter by budget and not already in squad
                cand = candidates[(candidates["now_cost_t"] <= max_buy_cost_t) & (~candidates["id"].isin(cur_ids))].copy()
                if cand.empty:
                    continue

                # Team limit
                def _ok_row(r) -> bool:
                    t = int(r["team"]) if pd.notna(r.get("team")) else -1
                    return _team_ok(counts_after_sell, t)

                cand = cand[cand.apply(_ok_row, axis=1)]
                if cand.empty:
                    continue

                # Take top few buys for branching
                cand = cand.sort_values("expected_points", ascending=False).head(8)
                for _, buy in cand.iterrows():
                    buy_id = int(buy["id"])
                    buy_team = int(buy["team"]) if pd.notna(buy.get("team")) else -1
                    buy_cost_t = int(buy["now_cost_t"])

                    new_bank_t = bank_t + sell_cost_t - buy_cost_t
                    score_gain = float(_safe_float(buy.get("expected_points")) - _safe_float(sell.get("expected_points")))
                    if score_gain <= 0:
                        continue

                    # Update team counts
                    new_counts = dict(counts_after_sell)
                    new_counts[buy_team] = new_counts.get(buy_team, 0) + 1

                    new_ids = set(cur_ids)
                    new_ids.remove(sell_id)
                    new_ids.add(buy_id)

                    delta_m = (buy_cost_t - sell_cost_t) / 10.0
                    pr, reason = _priority(pd.Series(sell), buy, score_gain, delta_m)
                    move = {
                        "Step": step + 1,
                        "Priority": pr,
                        "Sell": _player_display_name(pd.Series(sell)),
                        "Buy": _player_display_name(buy),
                        "Score gain": round(score_gain, 3),
                        "Delta (m)": round(delta_m, 1),
                        "Budget after (m)": round(new_bank_t / 10.0, 1),
                        "Reason": reason,
                    }

                    next_states.append((total_gain + score_gain, new_bank_t, new_ids, new_counts, moves + [move]))

        # Prune beam
        next_states.sort(key=lambda x: x[0], reverse=True)
        states = next_states[: max(5, int(beam_width))]
        if not states:
            break

    if not states:
        return []

    # Build top plans (unique by move signature)
    seen = set()
    plans: List[pd.DataFrame] = []
    for total_gain, _bank_t, _ids, _counts, moves in sorted(states, key=lambda x: x[0], reverse=True):
        sig = tuple((m["Sell"], m["Buy"]) for m in moves)
        if not moves or sig in seen:
            continue
        seen.add(sig)

        df = pd.DataFrame(moves)
        df.attrs["total_gain"] = total_gain
        # Put total gain in a visible column header-like way
        df.insert(0, "Total gain", [round(total_gain, 3)] + [""] * (len(df) - 1))
        plans.append(df)
        if len(plans) >= top_plans:
            break

    return plans


def build_wildcard_team(
    manager_id: int,
    gameweek: int,
    players_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
    budget_m: float = 100.0,
    weeks_ahead: int = 5,
) -> pd.DataFrame:
    """Build a simple 15-player wildcard squad (greedy)."""
    scored = _score_players(players_df, teams_df, fixtures_df, current_gw=int(gameweek), fixtures_ahead=int(weeks_ahead))
    scored = scored[scored["is_available"]].copy()

    budget_t = int(round(float(budget_m) * 10))
    need = {1: 2, 2: 5, 3: 5, 4: 3}

    picked_ids: List[int] = []
    team_counts: Dict[int, int] = {}
    spent_t = 0

    scored = scored.sort_values(["score", "total_points"], ascending=[False, False])
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

    for _, row in scored.iterrows():
        _try_pick(row)
        if all(picked_pos_counts[p] >= need[p] for p in need):
            break

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
    """Light chip suggestion based on double/blank gameweek signals."""
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
