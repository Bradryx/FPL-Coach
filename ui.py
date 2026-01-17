import streamlit as st
import pandas as pd
from fpl_assistant import (
    load_bootstrap, load_fixtures,
    generate_transfer_suggestions, build_wildcard_team,
    suggest_transfer_moves, suggest_chip_play, show_current_team
)

st.title("Fantasy Premier League Assistant")

manager_id = st.number_input("Manager ID", min_value=1, value=1548623)
gameweek = st.number_input("Gameweek", min_value=1, value=6)

players_df, teams_df = load_bootstrap()
fixtures_df = load_fixtures()

if st.button("Show Transfer Suggestions"):
    suggestions = generate_transfer_suggestions(
        manager_id, gameweek, players_df, teams_df, fixtures_df, top_n=5
    )
    st.write("### Transfer Targets")
    st.dataframe(suggestions)

if st.button("Build Wildcard Squad"):
    squad = build_wildcard_team(
        manager_id, gameweek, players_df, teams_df, fixtures_df
    )
    st.write("### Wildcard Squad")
    st.dataframe(squad)

if st.button("Suggest Transfer Moves"):
    moves = suggest_transfer_moves(manager_id, gameweek, players_df, teams_df, fixtures_df)
    st.write("### Transfer Moves")
    if moves:
        for sell, buy, delta in moves:
            st.write(f"{sell} → {buy} ({delta:+.1f}m)")
    else:
        st.write("No sensible moves found.")

if st.button("Chip Suggestion"):
    chip = suggest_chip_play(manager_id, gameweek, players_df, teams_df, fixtures_df)
    if chip:
        st.write(f"Recommended chip: {chip}")
    else:
        st.write("No chip suggestion at this time.")
