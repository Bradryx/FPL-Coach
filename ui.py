import streamlit as st
import pandas as pd

from fpl_assistant import (
    load_bootstrap,
    load_fixtures,
    get_current_gameweek,
    generate_transfer_suggestions,
    build_wildcard_team,
    suggest_transfer_moves,
    suggest_chip_play,
    show_current_team,
)

st.set_page_config(page_title="FPL Assistant", layout="wide")
st.title("Fantasy Premier League Assistant")

# --- Data loading with friendly errors (otherwise Streamlit 'stops')
try:
    players_df, teams_df = load_bootstrap()
    fixtures_df = load_fixtures()
except Exception as e:
    st.error(
        "Kon FPL data niet laden. Check of je internet/toegang tot fantasy.premierleague.com werkt.\n\n"
        f"Details: {type(e).__name__}: {e}"
    )
    st.stop()

# bootstrap-static contains 'events' but we didn't return it; quick re-fetch is wasteful.
# Instead: let user choose, with a sensible default.

MANAGERS = {
    "Custom": None,
    "Brandon": 1548623,
    "Elwin": 3979149,
    "Abdel": 4023757,
    "Bart": 2111015,
    "Nick": 3977511,
}

manager_choice = st.selectbox("Manager", list(MANAGERS.keys()), index=1)
default_manager_id = MANAGERS.get(manager_choice) or 1548623
manager_id = st.number_input(
    "Manager ID",
    min_value=1,
    value=int(default_manager_id),
    step=1,
    disabled=(manager_choice != "Custom"),
)

default_gw = int(get_current_gameweek())
gameweek = st.number_input("Gameweek", min_value=1, value=default_gw, step=1)

fixtures_ahead = st.slider(
    "Aantal upcoming fixtures voor FDR/scoring",
    min_value=1,
    max_value=10,
    value=5,
    step=1,
)

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    if st.button("Show Current Team"):
        try:
            team_df = show_current_team(
                manager_id,
                gameweek,
                players_df,
                teams_df,
                fixtures_df,
                fixtures_ahead=int(fixtures_ahead),
            )
            
            st.subheader("Current Team")
            st.dataframe(team_df, use_container_width=True)
        except Exception as e:
            st.error(f"Fout bij laden team: {type(e).__name__}: {e}")

with col2:
    if st.button("Show Transfer Suggestions"):
        try:
            suggestions = generate_transfer_suggestions(
                manager_id,
                gameweek,
                players_df,
                teams_df,
                fixtures_df,
                fixtures_ahead=int(fixtures_ahead),
                top_n=10,
            )
            st.subheader("Transfer Targets")
            st.dataframe(suggestions, use_container_width=True)
        except Exception as e:
            st.error(f"Fout bij transfer targets: {type(e).__name__}: {e}")

with col3:
    if st.button("Suggest Transfer Moves"):
        try:
            moves = suggest_transfer_moves(
                manager_id,
                gameweek,
                players_df,
                teams_df,
                fixtures_df,
                weeks_ahead=int(fixtures_ahead),
            )
            st.subheader("Transfer Moves")
            if moves:
                for sell, buy, delta in moves:
                    st.write(f"{sell} → {buy} ({delta:+.1f}m)")
            else:
                st.write("Geen duidelijke move gevonden.")
        except Exception as e:
            st.error(f"Fout bij transfer moves: {type(e).__name__}: {e}")

with col4:
    if st.button("Build Wildcard Squad"):
        try:
            squad = build_wildcard_team(
                manager_id,
                gameweek,
                players_df,
                teams_df,
                fixtures_df,
                weeks_ahead=int(fixtures_ahead),
            )
            st.subheader("Wildcard Squad")
            st.caption(
                f"Budget spent: {squad.attrs.get('budget_spent_m', 0):.1f}m | "
                f"Left: {squad.attrs.get('budget_left_m', 0):.1f}m"
            )
            st.dataframe(squad, use_container_width=True)
        except Exception as e:
            st.error(f"Fout bij wildcard squad: {type(e).__name__}: {e}")

with col5:
    if st.button("Chip Suggestion"):
        try:
            chip = suggest_chip_play(manager_id, gameweek, players_df, teams_df, fixtures_df)
            st.subheader("Chip")
            st.write(chip or "Geen chip suggestie.")
        except Exception as e:
            st.error(f"Fout bij chip suggestie: {type(e).__name__}: {e}")
