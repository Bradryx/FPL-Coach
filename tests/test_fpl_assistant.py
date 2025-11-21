import pandas as pd

from fpl_assistant import (
    compute_fdr_for_team,
    get_upcoming_fixtures,
    infer_current_gameweek,
)


def test_compute_fdr_for_team_respects_gameweek_window():
    fixtures = pd.DataFrame(
        [
            {
                "event": 5,
                "team_h": 1,
                "team_a": 2,
                "team_h_difficulty": 2,
                "team_a_difficulty": 4,
                "kickoff_time": "2024-08-01T15:00:00Z",
            },
            {
                "event": 6,
                "team_h": 3,
                "team_a": 1,
                "team_h_difficulty": 3,
                "team_a_difficulty": 2,
                "kickoff_time": "2024-08-08T15:00:00Z",
            },
            {
                "event": 7,
                "team_h": 1,
                "team_a": 4,
                "team_h_difficulty": 5,
                "team_a_difficulty": 1,
                "kickoff_time": "2024-08-15T15:00:00Z",
            },
            {
                "event": None,
                "team_h": 1,
                "team_a": 5,
                "team_h_difficulty": 4,
                "team_a_difficulty": 2,
                "kickoff_time": "2024-08-22T15:00:00Z",
            },
        ]
    )

    difficulty = compute_fdr_for_team(
        team_id=1, fixtures=fixtures, start_gameweek=6, weeks_ahead=2
    )

    # Only GW6 and GW7 fixtures should be considered: (2 + 5) / 2 = 3.5
    assert difficulty == 3.5


def test_get_upcoming_fixtures_formats_and_limits_results():
    fixtures = pd.DataFrame(
        [
            {
                "event": 6,
                "team_h": 1,
                "team_a": 2,
                "team_h_difficulty": 3,
                "team_a_difficulty": 2,
                "kickoff_time": "2024-08-08T15:00:00Z",
            },
            {
                "event": 8,
                "team_h": 4,
                "team_a": 1,
                "team_h_difficulty": 4,
                "team_a_difficulty": 2,
                "kickoff_time": "2024-09-05T18:00:00Z",
            },
            {
                "event": 7,
                "team_h": 1,
                "team_a": 3,
                "team_h_difficulty": 2,
                "team_a_difficulty": 4,
                "kickoff_time": "2024-08-20T12:30:00Z",
            },
        ]
    )
    teams_df = pd.DataFrame(
        [
            {"id": 1, "short_name": "ABC"},
            {"id": 2, "short_name": "DEF"},
            {"id": 3, "short_name": "GHI"},
            {"id": 4, "short_name": "JKL"},
        ]
    )

    fixtures_str = get_upcoming_fixtures(
        team_id=1, fixtures=fixtures, teams_df=teams_df, current_gw=6, num_games=2
    )

    # Sorted by event (and kickoff_time within the same event) and limited to two entries
    assert fixtures_str == "DEF (H,3); GHI (H,2)"


def test_infer_current_gameweek_prefers_unfinished_fixtures():
    fixtures = pd.DataFrame(
        [
            {"event": 1, "finished": True},
            {"event": 2, "finished": False},
            {"event": 3, "finished": False},
        ]
    )

    assert infer_current_gameweek(fixtures) == 2


def test_infer_current_gameweek_falls_back_to_latest_when_finished():
    fixtures = pd.DataFrame(
        [
            {"event": 1, "finished": True},
            {"event": 2, "finished": True},
        ]
    )

    assert infer_current_gameweek(fixtures) == 2
