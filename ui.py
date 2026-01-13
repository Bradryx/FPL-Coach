import streamlit as st
import pandas as pd
from fpl_assistant import (
    load_bootstrap, load_fixtures,
    generate_transfer_suggestions, build_wildcard_team,
    suggest_transfer_moves, suggest_chip_play
)

# --- Manager mapping ---
manager_map = {
    "Brandon": 1548623,
    "Elwin" : 3979149,
    "Abdel": 4023757,
    "Bart": 2111015,
    "Nick": 3977511
}

# --- Page setup ---
st.set_page_config(page_title="FPL Assistant", layout="wide")

# --- Custom CSS for PL style ---
def load_css(dark=False):
    if dark:
        bg = "#0d1117"
        text = "#e6edf3"
        card_bg = "#161b22"
    else:
        bg = "#f5f7fa"
        text = "#1a1a1a"
        card_bg = "#ffffff"

    st.markdown(
        f"""
        <style>
        body, .main, .block-container {{
            background: {bg} !important;
            color: {text} !important;
        }}
        .stTabs [role="tablist"] button[role="tab"] {{
            border-radius: 6px;
            padding: 8px 16px;
            margin: 0 4px;
            background: linear-gradient(90deg, #7b2ff7, #00c6ff);
            color: white !important;
            font-weight: bold;
            border: none;
        }}
        .stTabs [role="tablist"] {{
            background: transparent !important;
        }}
        .card {{
            background: {card_bg};
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 20px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# --- Sidebar ---
st.sidebar.header("⚙️ Instellingen")
theme = st.sidebar.selectbox("Thema", ["🌞 Licht", "🌙 Donker"])
dark_mode = theme == "🌙 Donker"
load_css(dark=dark_mode)

selected_manager = st.sidebar.selectbox("Manager", list(manager_map.keys()))
manager_id = manager_map[selected_manager]

gameweek = st.sidebar.number_input("Gameweek", min_value=1, value=6, step=1)

# --- Title ---
st.markdown(
    """
    <h1 style="background: linear-gradient(90deg, #7b2ff7, #00c6ff);
               -webkit-background-clip: text;
               -webkit-text-fill-color: transparent;
               font-weight: 900; text-align:center;">
    ⚽ Fantasy Premier League Assistant
    </h1>
    """,
    unsafe_allow_html=True,
)

# --- Fixtures formatter ---
def color_fixtures(fixtures_str: str) -> str:
    parts = fixtures_str.split("; ")
    styled = []
    for p in parts:
        try:
            diff = int(p.split(",")[-1].replace(")", ""))
        except:
            diff = 3
        if diff <= 2:
            color = "green"
        elif diff == 3:
            color = "orange"
        else:
            color = "red"
        styled.append(f"<span style='color:{color}; font-weight:bold'>{p}</span>")
    return " | ".join(styled)

# --- Load data ---
with st.spinner("FPL data ophalen..."):
    players_df, teams_df = load_bootstrap()
    fixtures_df = load_fixtures()

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(
    ["🔄 Transfer Targets", "🃏 Wildcard Squad", "📋 Transfer Moves", "🎲 Chip Suggestion"]
)

with tab1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Top Transfer Targets")
    suggestions = generate_transfer_suggestions(
        manager_id, gameweek, players_df, teams_df, fixtures_df, top_n=5
    )
    # NL kolomnamen
    suggestions.rename(columns={
        "Name": "Speler",
        "Team": "Club",
        "Position": "Positie",
        "Price": "Prijs (M)",
        "Points_per_Game": "Punten/Gem",
        "Avg_FDR_next6": "Gem Moeilijkheid (6)",
        "Score": "Score",
        "Fixtures": "Programma"
    }, inplace=True)
    # Fixtures kleuren
    suggestions["Programma"] = suggestions["Programma"].apply(color_fixtures)
    st.write(suggestions.to_html(escape=False), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Wildcard Squad")
    squad = build_wildcard_team(manager_id, gameweek, players_df, teams_df, fixtures_df)
    if squad.empty:
        st.warning("Geen geldige wildcard squad binnen budget.")
    else:
        squad.rename(columns={
            "Name": "Speler",
            "Team": "Club",
            "Position": "Positie",
            "Price": "Prijs (M)",
            "Points_per_Game": "Punten/Gem",
            "Minutes": "Minuten",
            "Avg_FDR_next6": "Gem Moeilijkheid (6)",
            "Fixtures": "Programma"
        }, inplace=True)
        squad["Programma"] = squad["Programma"].apply(color_fixtures)
        st.write(squad.to_html(escape=False), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Budget-aware Transfer Moves")
    moves = suggest_transfer_moves(manager_id, gameweek, players_df, teams_df, fixtures_df)
    if moves:
        for sell, buy, delta in moves:
            sign = f"{delta:+.1f}m"
            color = "🟢" if delta > 0 else "🔴" if delta < 0 else "🔵"
            st.markdown(f"{color} **{sell} → {buy}** ({sign})")
    else:
        st.info("Geen verstandige transfer moves gevonden.")
    st.markdown('</div>', unsafe_allow_html=True)

with tab4:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Chip Suggestion")
    chip = suggest_chip_play(manager_id, gameweek, players_df, teams_df, fixtures_df)
    if chip:
        st.success(f"Gebruik je **{chip.upper()}** chip!")
    else:
        st.info("Geen chip nodig deze week.")
    st.markdown('</div>', unsafe_allow_html=True)
