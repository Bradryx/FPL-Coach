"""
Simple Fantasy Premier League (FPL) assistant.

This script shows how to connect to the public FPL API, download
player and fixture data and compute basic statistics.  It then
extracts the current gameweek picks for a specified manager and
returns a basic summary table to help with decision making.

The script does not require any authentication for the endpoints
used here.  However, more advanced endpoints (such as `my-team/{id}`)
do require a login cookie.  See the FPL API guide for details on
how to obtain an authentication token.

Usage:
    python fpl_assistant.py --manager-id 1548623 --gameweek 6

Note: running this script from within the OpenAI environment may
not succeed because direct HTTP requests to fantasy.premierleague.com
are restricted.  You should run it on your own machine with internet
access to the FPL API.  See the README in this repository for more
details.
"""

import argparse
import json
import sys
from collections import defaultdict

try:
    import pandas as pd  # type: ignore
    import requests  # type: ignore
except ImportError:
    sys.exit("This script requires the pandas and requests packages.")


def fetch_json(url: str) -> dict:
    """Fetch JSON from the given URL with a browser user‑agent.

    The FPL API may return a 403 if the request does not include a
    User‑Agent header.  Passing a common browser UA usually avoids
    this issue.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/118.0 Safari/537.36"
        )
    }
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    return resp.json()


def load_bootstrap() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load overall FPL data (players and teams) from bootstrap‑static.

    Returns a tuple (players_df, teams_df).
    """
    data = fetch_json("https://fantasy.premierleague.com/api/bootstrap-static/")
    players_df = pd.DataFrame(data["elements"])
    teams_df = pd.DataFrame(data["teams"])
    return players_df, teams_df


def load_fixtures() -> pd.DataFrame:
    """Load fixture list and convert to DataFrame."""
    fixtures = fetch_json("https://fantasy.premierleague.com/api/fixtures/")
    return pd.DataFrame(fixtures)


def load_picks(manager_id: int, gameweek: int) -> dict:
    """Load the picks for the given manager and gameweek."""
    url = (
        f"https://fantasy.premierleague.com/api/entry/{manager_id}/"
        f"event/{gameweek}/picks/"
    )
    return fetch_json(url)


def compute_fdr_for_team(
    team_id: int, fixtures: pd.DataFrame, weeks_ahead: int = 6
) -> float:
    """Compute the average fixture difficulty for a team over the next N weeks.

    The `fixtures` DataFrame should come from the fixtures endpoint and
    contain the columns `event`, `team_h`, `team_a` and `difficulty` (for both
    home and away teams).  Difficulty ratings are 1–5 where 5 is hardest
    according to the official FPL Fixture Difficulty Rating【271463546749381†L119-L136】.
    """
    current_gw = fixtures["event"].min()
    upcoming = fixtures[(fixtures["event"] >= current_gw) & (fixtures["event"] < current_gw + weeks_ahead)]
    difficulties = []
    for _, row in upcoming.iterrows():
        if row["team_h"] == team_id:
            difficulties.append(row["team_h_difficulty"])
        if row["team_a"] == team_id:
            difficulties.append(row["team_a_difficulty"])
    return sum(difficulties) / len(difficulties) if difficulties else 0.0


def generate_transfer_suggestions(
    manager_id: int,
    gameweek: int,
    players_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
    top_n: int = 5,
) -> pd.DataFrame:
    """Generate a simple list of potential transfer targets.

    This function compares the manager's current picks with the rest of
    the player pool and returns a list of high‑performing players that
    are not currently in the squad.  It does not take budget or position
    constraints into account; see :func:`suggest_transfer_moves` for a
    budget‑aware replacement strategy.
    """
    picks_data = load_picks(manager_id, gameweek)
    current_elements = {p["element"] for p in picks_data["picks"]}

    # Merge team data into players for lookup of team name/short name
    merged = players_df.merge(
        teams_df[["id", "name", "short_name"]].rename(columns={"id": "team"}),
        left_on="team",
        right_on="team",
        how="left",
    )
    # Compute FDR per team for the next six weeks
    fdr_map: dict[int, float] = {}
    for team_id in merged["team"].unique():
        fdr_map[team_id] = compute_fdr_for_team(team_id, fixtures_df, weeks_ahead=6)
    merged["fdr_next6"] = merged["team"].map(fdr_map)

    # Exclude players already owned and those unavailable (status != 'a')
    available = merged[(~merged["id"].isin(current_elements)) & (merged["status"] == "a")].copy()

    # Adjust scores by minutes played.  Players who have played more minutes are
    # considered more reliable.  We scale minutes by the maximum possible
    # minutes so far (gameweek × 90) and multiply the form score by this
    # availability ratio.
    max_minutes = max(gameweek, 1) * 90
    available["minutes_ratio"] = available["minutes"] / max_minutes
    available["minutes_ratio"].fillna(0.0, inplace=True)
    # Compute a score: points per game adjusted by minutes_ratio and divided by FDR
    available["transfer_score"] = (
        available["points_per_game"].astype(float) * available["minutes_ratio"]
    ) / (available["fdr_next6"] + 1e-3)

    # Sort by score and limit to top candidates
    top = available.sort_values("transfer_score", ascending=False).head(top_n).copy().reset_index(drop=True)
    # Add rank (1..n)
    top.insert(0, "Rank", top.index + 1)
    # Map element_type to position names
    position_map = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
    top["Position"] = top["element_type"].map(position_map)
    # Convert cost to millions for clarity
    top["Price"] = top["now_cost"] / 10.0
    # Rename columns for readability
    top.rename(
        columns={
            "web_name": "Name",
            "points_per_game": "Points_per_Game",
            "fdr_next6": "Avg_FDR_next6",
            "transfer_score": "Score",
        },
        inplace=True,
    )
    # Drop unnecessary columns (id, element_type, now_cost, minutes_ratio)
    drop_cols = ["id", "element_type", "now_cost", "minutes_ratio"]
    for col in drop_cols:
        if col in top.columns:
            top.drop(columns=col, inplace=True)
    # Add player's own team short name and upcoming fixtures string
    # Player's team short name
    team_lookup = teams_df.set_index("id")["short_name"].to_dict()
    top["Team"] = top["team"].map(team_lookup)
    # Determine current gameweek from fixtures to align upcoming fixtures
    current_gw = fixtures_df["event"].min()
    top["Fixtures"] = top.apply(
        lambda row: get_upcoming_fixtures(int(row["team"]), fixtures_df, teams_df, current_gw=current_gw, num_games=5),
        axis=1,
    )
    # Drop internal 'team' column as it is now represented by 'Team'
    if "team" in top.columns:
        top.drop(columns=["team"], inplace=True)
    # Reorder columns
    ordered_cols = [
        "Rank",
        "Name",
        "Team",
        "Position",
        "Price",
        "Points_per_Game",
        "Avg_FDR_next6",
        "Score",
        "Fixtures",
    ]
    return top[ordered_cols]


def suggest_transfer_moves(
    manager_id: int,
    gameweek: int,
    players_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
    max_transfers: int = 2,
) -> list[tuple[str, str, float]]:
    """Suggest specific transfer moves based on budget and team composition.

    This function analyses the manager's current squad and identifies a small
    number of underperforming players to replace.  For each such player it
    recommends a replacement in the same position who has a higher expected
    return (points per game relative to upcoming fixture difficulty) and fits
    within the available budget.  The result is a list of tuples of the
    form (sell_player_name, buy_player_name, price_difference).

    :param manager_id: FPL manager entry ID.
    :param gameweek: current gameweek number.
    :param players_df: full player data from bootstrap‑static.
    :param teams_df: team information from bootstrap‑static.
    :param fixtures_df: full fixture list.
    :param max_transfers: maximum number of transfer moves to suggest.
    :returns: a list of suggested transfers.
    """
    picks_data = load_picks(manager_id, gameweek)
    picks = picks_data["picks"]
    bank = picks_data["entry_history"]["bank"] / 10.0

    # Merge team data into players for FDR mapping
    merged = players_df.merge(
        teams_df[["id", "name", "short_name"]].rename(columns={"id": "team"}),
        left_on="team",
        right_on="team",
        how="left",
    )
    # Compute FDR per team for next six weeks
    fdr_map: dict[int, float] = {}
    for team_id in merged["team"].unique():
        fdr_map[team_id] = compute_fdr_for_team(team_id, fixtures_df, weeks_ahead=6)
    merged["fdr_next6"] = merged["team"].map(fdr_map)
    merged["points_per_game"] = merged["points_per_game"].astype(float)
    # Minutes ratio to penalise low‑minute players.  Use gameweek to compute max
    max_minutes = max(gameweek, 1) * 90
    merged["minutes_ratio"] = merged["minutes"] / max_minutes
    merged["minutes_ratio"].fillna(0.0, inplace=True)

    # Build lookup of current squad with cost and position
    current_ids = {p["element"] for p in picks}
    current_df = merged[merged["id"].isin(current_ids)].copy()
    # Include minutes_ratio in the current player score
    current_df["current_score"] = (
        current_df["points_per_game"] * current_df["minutes_ratio"]
    ) / (current_df["fdr_next6"] + 1e-3)

    # Identify underperforming players (lowest scores)
    worst_players = current_df.sort_values("current_score").head(max_transfers)

    suggestions: list[tuple[str, str, float]] = []
    # Build a team count mapping to enforce max 3 per team in the new squad
    # Count current players per team
    team_count = defaultdict(int)
    for _, p_row in current_df.iterrows():
        team_count[p_row["team"]] += 1

    # For each player to be replaced
    for _, row in worst_players.iterrows():
        sell_id = int(row["id"])
        sell_name = row["web_name"]
        sell_cost = row["now_cost"] / 10.0
        position = row["element_type"]
        available_budget = bank + sell_cost

        # Candidates: same position, not currently owned, available, cost within budget, team limit not exceeded
        candidates = merged[
            (merged["element_type"] == position)
            & (~merged["id"].isin(current_ids))
            & (merged["status"] == "a")
            & ((merged["now_cost"] / 10.0) <= available_budget)
        ].copy()

        if candidates.empty:
            continue
        # Compute candidate score
        candidates["candidate_score"] = (
            candidates["points_per_game"] * candidates["minutes_ratio"]
        ) / (candidates["fdr_next6"] + 1e-3)
        # Sort by score descending
        candidates.sort_values("candidate_score", ascending=False, inplace=True)
        # Find the first candidate who does not break the 3‑per‑team rule
        selected_candidate = None
        for _, cand in candidates.iterrows():
            team_id = cand["team"]
            # After replacing sell_id, we will remove from team_count of sell player's team
            # but we haven't removed yet; we evaluate candidate separately for each move
            # Only ensure candidate's team count does not exceed 3
            if team_count[team_id] >= 3:
                continue
            selected_candidate = cand
            break
        if selected_candidate is None:
            continue

        buy_name = selected_candidate["web_name"]
        buy_cost = selected_candidate["now_cost"] / 10.0
        suggestions.append((sell_name, buy_name, buy_cost - sell_cost))

        # Update budget and current squads counts for subsequent suggestions
        bank += sell_cost - buy_cost
        current_ids.remove(sell_id)
        current_ids.add(int(selected_candidate["id"]))
        # Update team counts: remove sell and add buy
        team_count[row["team"]] -= 1
        team_count[selected_candidate["team"]] += 1

    return suggestions


def suggest_chip_play(
    manager_id: int,
    gameweek: int,
    players_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
    fdr_threshold: float = 3.5,
    injury_threshold: int = 3,
) -> str | None:
    """Heuristically determine whether a Wildcard chip might be warranted.

    The logic is intentionally simple: if your team has a high average fixture
    difficulty for the coming weeks or multiple unavailable players (injured or
    suspended), the function recommends a Wildcard.  Free Hit and other chips
    are not automatically suggested because they depend heavily on blank/double
    gameweeks which this script cannot detect in this environment.

    :returns: the name of the recommended chip ('wildcard') or None if no chip
    is recommended.
    """
    picks_data = load_picks(manager_id, gameweek)
    picks = picks_data["picks"]
    # Determine which players are unavailable (status != 'a')
    merged = players_df.merge(
        teams_df[["id", "name", "short_name"]].rename(columns={"id": "team"}),
        left_on="team",
        right_on="team",
        how="left",
    )
    # Map player id to status and team for FDR
    current_ids = {p["element"] for p in picks}
    current_df = merged[merged["id"].isin(current_ids)].copy()
    # Compute FDR per team
    fdr_map: dict[int, float] = {}
    for team_id in merged["team"].unique():
        fdr_map[team_id] = compute_fdr_for_team(team_id, fixtures_df, weeks_ahead=6)
    current_df["fdr_next6"] = current_df["team"].map(fdr_map)
    unavailable = current_df[current_df["status"] != "a"]
    avg_fdr = current_df["fdr_next6"].mean() if not current_df.empty else 0.0
    # Recommend a wildcard if many players are injured/unavailable or the average
    # upcoming fixture difficulty is high
    if len(unavailable) >= injury_threshold or avg_fdr >= fdr_threshold:
        return "wildcard"
    return None


def show_current_team(
    manager_id: int,
    gameweek: int,
    players_df: pd.DataFrame,
    teams_df: pd.DataFrame,
) -> None:
    """Display the user's current squad with names, positions and basic stats.

    This function retrieves the manager's picks for the specified gameweek
    and matches the element IDs to player names.  It prints a table with
    position, name, cost (in £m), points per game and total minutes.
    """
    picks_data = load_picks(manager_id, gameweek)
    ids = [p["element"] for p in picks_data["picks"]]
    subset = players_df[players_df["id"].isin(ids)].copy()
    position_map = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
    subset["position"] = subset["element_type"].map(position_map)
    subset["cost"] = subset["now_cost"] / 10.0
    cols = ["position", "web_name", "cost", "points_per_game", "minutes"]
    subset = subset[cols].sort_values(["position", "web_name"])
    print(subset.to_string(index=False))


def build_wildcard_team(
    manager_id: int,
    gameweek: int,
    players_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
    weeks_ahead: int = 6,
    verbose: bool = True,
) -> pd.DataFrame:
    """Construct a fresh 15‑man squad ignoring current picks using the manager's total budget.

    This helper is intended for use when the Wildcard chip is active.  It ignores
    the user's existing squad and instead assembles an optimal team based on
    form (points per game), fixture difficulty (FDR) and minutes played.  It
    enforces standard FPL squad rules: 2 goalkeepers, 5 defenders, 5
    midfielders and 3 forwards; a maximum of three players from any one
    Premier League team; and total cost not exceeding the manager's current
    squad value plus available bank.  The function returns a DataFrame
    containing the selected players sorted by position.

    Note: This is a greedy approximation and may not always find the absolute
    optimal team, but it generally yields a strong squad within the budget.

    :param manager_id: FPL manager entry ID.
    :param gameweek: current gameweek number (used to scale minutes).
    :param players_df: full player data from bootstrap‑static.
    :param teams_df: team information from bootstrap‑static.
    :param fixtures_df: fixture list.
    :param weeks_ahead: how many upcoming fixtures to consider when computing FDR.
    :param verbose: whether to print diagnostic information about budget usage.
    :returns: DataFrame with selected players and their attributes.
    """
    # Determine available budget: team value plus bank
    picks_data = load_picks(manager_id, gameweek)
    entry = picks_data.get("entry_history", {})
    # value and bank are stored as integers in tenths of a million (e.g. 1003 => £100.3m)
    squad_value = entry.get("value", 0) / 10.0
    bank = entry.get("bank", 0) / 10.0
    total_budget = squad_value + bank

    # Merge team info to compute FDR per team
    merged = players_df.merge(
        teams_df[["id", "name", "short_name"]].rename(columns={"id": "team"}),
        left_on="team",
        right_on="team",
        how="left",
    )
    # Compute FDR map
    fdr_map: dict[int, float] = {}
    for team_id in merged["team"].unique():
        fdr_map[team_id] = compute_fdr_for_team(team_id, fixtures_df, weeks_ahead)
    merged["fdr_next"] = merged["team"].map(fdr_map)
    merged["points_per_game"] = merged["points_per_game"].astype(float)
    # Minutes ratio penalises players with limited playing time
    max_minutes = max(gameweek, 1) * 90
    merged["minutes_ratio"] = merged["minutes"] / max_minutes
    merged["minutes_ratio"].fillna(0.0, inplace=True)
    # Only consider players who are available (status == 'a')
    available = merged[merged["status"] == "a"].copy()
    # Score: combine points per game and minutes, scaled by fixture difficulty
    available["score"] = (
        available["points_per_game"] * available["minutes_ratio"]
    ) / (available["fdr_next"] + 1e-3)

    # Define roster requirements per position (element_type)
    position_requirements = {1: 2, 2: 5, 3: 5, 4: 3}
    # Container to track selected players and team counts
    selected_rows: list[dict] = []
    team_counts: defaultdict[int, int] = defaultdict(int)
    remaining_budget = total_budget

    # Iterate through each position group
    for position, needed in position_requirements.items():
        # Sort available players within this position by score descending
        pool = available[available["element_type"] == position].copy()
        pool.sort_values(["score"], ascending=False, inplace=True)

        count = 0
        for _, player in pool.iterrows():
            if count >= needed:
                break
            # Price in millions
            price = player["now_cost"] / 10.0
            # Check budget and team limit
            if price > remaining_budget:
                continue
            if team_counts[player["team"]] >= 3:
                continue
            # Select player
            selected_rows.append(player.to_dict())
            remaining_budget -= price
            team_counts[player["team"]] += 1
            count += 1
            # Break if requirement met
            if count >= needed:
                break
        # If we could not fill requirement due to budget/team constraints, we stop
        # Additional complex optimisation is beyond this script's scope

    # Create DataFrame from selected players and compute derived fields
    selected_df = pd.DataFrame(selected_rows)
    if selected_df.empty:
        return selected_df
    position_map = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
    selected_df["Position"] = selected_df["element_type"].map(position_map)
    selected_df["Price"] = selected_df["now_cost"] / 10.0
    # Map team id to short name
    team_lookup = teams_df.set_index("id")["short_name"].to_dict()
    selected_df["Team"] = selected_df["team"].map(team_lookup)
    # Compute upcoming fixtures string for each selected player
    # Determine current gameweek from fixtures
    current_gw = fixtures_df["event"].min()
    selected_df["Fixtures"] = selected_df.apply(
        lambda row: get_upcoming_fixtures(int(row["team"]), fixtures_df, teams_df, current_gw=current_gw, num_games=5),
        axis=1,
    )
    # Rename stats columns for readability
    selected_df.rename(
        columns={
            "web_name": "Name",
            "points_per_game": "Points_per_Game",
            "minutes": "Minutes",
            "fdr_next": "Avg_FDR_next6",
        },
        inplace=True,
    )
    # Select and order columns
    cols = ["Position", "Name", "Team", "Price", "Points_per_Game", "Minutes", "Avg_FDR_next6", "Fixtures"]
    selected_df = selected_df[cols].sort_values(["Position", "Name"])

    if verbose:
        spent = total_budget - remaining_budget
        print(f"\nWildcard selection built using £{spent:.1f}m of £{total_budget:.1f}m budget; £{remaining_budget:.1f}m remaining.")

    return selected_df


def get_upcoming_fixtures(
    team_id: int,
    fixtures: pd.DataFrame,
    teams_df: pd.DataFrame,
    current_gw: int,
    num_games: int = 5,
) -> str:
    """Return a string summarising the next `num_games` fixtures for the given team.

    Each fixture is formatted as "OPP (H/A,difficulty)", where OPP is the
    opponent's short name, H/A indicates home or away, and difficulty is the
    official FDR rating.  Fixtures are drawn from events on or after the
    current gameweek and sorted by event.
    """
    # Filter for upcoming fixtures for this team
    upcoming = fixtures[(fixtures["event"] >= current_gw) & (
        (fixtures["team_h"] == team_id) | (fixtures["team_a"] == team_id)
    )].sort_values("event")
    summaries: list[str] = []
    # Build a lookup for team short names
    team_lookup = teams_df.set_index("id")["short_name"].to_dict()
    for _, row in upcoming.head(num_games).iterrows():
        if row["team_h"] == team_id:
            opp_id = row["team_a"]
            difficulty = row["team_h_difficulty"]
            loc = "H"
        else:
            opp_id = row["team_h"]
            difficulty = row["team_a_difficulty"]
            loc = "A"
        opp_name = team_lookup.get(opp_id, str(opp_id))
        summaries.append(f"{opp_name} ({loc},{difficulty})")
    return "; ".join(summaries)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fantasy Premier League assistant")
    parser.add_argument("--manager-id", type=int, required=True, help="Your FPL manager entry ID")
    parser.add_argument("--gameweek", type=int, required=True, help="Current gameweek number")
    parser.add_argument("--top-n", type=int, default=5, help="Number of transfer suggestions to display")
    parser.add_argument(
        "--wildcard",
        action="store_true",
        help="If set, build a new 15‑man squad using your total budget instead of suggesting transfers for the current squad.",
    )
    args = parser.parse_args()

    print(f"Loading FPL data for manager {args.manager_id} (GW{args.gameweek})…")
    players_df, teams_df = load_bootstrap()
    fixtures_df = load_fixtures()

    if args.wildcard:
        # Construct a completely new squad within the available budget
        new_team = build_wildcard_team(
            manager_id=args.manager_id,
            gameweek=args.gameweek,
            players_df=players_df,
            teams_df=teams_df,
            fixtures_df=fixtures_df,
            weeks_ahead=6,
            verbose=True,
        )
        if new_team.empty:
            print("\nUnable to construct a wildcard squad within your budget and constraints.")
        else:
            print("\nProposed Wildcard squad (sorted by position):")
            print(new_team.to_string(index=False))
    else:
        # Generate generic transfer suggestions
        suggestions = generate_transfer_suggestions(
            manager_id=args.manager_id,
            gameweek=args.gameweek,
            players_df=players_df,
            teams_df=teams_df,
            fixtures_df=fixtures_df,
            top_n=args.top_n,
        )
        print("\nTop transfer targets based on form and upcoming fixtures:")
        print(suggestions.to_string(index=False))

        # Display the current squad with names and stats
        print("\nYour current squad (sorted by position):")
        show_current_team(
            manager_id=args.manager_id,
            gameweek=args.gameweek,
            players_df=players_df,
            teams_df=teams_df,
        )

        # Provide budget‑aware transfer moves
        moves = suggest_transfer_moves(
            manager_id=args.manager_id,
            gameweek=args.gameweek,
            players_df=players_df,
            teams_df=teams_df,
            fixtures_df=fixtures_df,
            max_transfers=2,
        )
        if moves:
            print("\nSuggested transfer moves (sell → buy, budget impact):")
            for sell, buy, delta in moves:
                direction = "(cost neutral)" if abs(delta) < 1e-6 else (
                    f"(+£{delta:.1f}m)" if delta > 0 else f"(–£{abs(delta):.1f}m)"
                )
                print(f" - {sell} → {buy} {direction}")
        else:
            print("\nNo sensible transfer moves found within your budget.")

        # Suggest whether to play a chip such as the wildcard
        chip_suggestion = suggest_chip_play(
            manager_id=args.manager_id,
            gameweek=args.gameweek,
            players_df=players_df,
            teams_df=teams_df,
            fixtures_df=fixtures_df,
        )
        if chip_suggestion:
            print(f"\nChip recommendation: consider playing your {chip_suggestion} chip.")


if __name__ == "__main__":
    main()