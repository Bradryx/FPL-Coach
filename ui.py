import streamlit as st
import pandas as pd

from fpl_assistant import (
    FPLError,
    get_current_gameweek,
    load_bootstrap,
    load_fixtures,
    resolve_gameweek,
    show_current_team,
    generate_transfer_suggestions,
    suggest_transfer_plans,
)


MANAGERS = {
    "Custom": None,
    "Brandon": 1548623,
    "Elwin": 3979149,
    "Abdel": 4023757,
    "Bart": 2111015,
    "Nick": 3977511,
}


st.set_page_config(page_title="FPL Coach", layout="wide")
st.title("FPL Coach")

# ----- Sidebar controls -----
with st.sidebar:
    st.header("Inputs")

    page = st.selectbox("Page", ["Home", "Coach"], index=1)

    manager_name = st.selectbox("Manager", list(MANAGERS.keys()), index=1)
    default_mid = MANAGERS.get(manager_name) or 1548623
    manager_id = st.number_input("Manager ID", min_value=1, value=int(default_mid))

    current_gw = get_current_gameweek()
    requested_gw = st.number_input("Gameweek", min_value=1, value=int(current_gw))

    fixtures_ahead = st.slider("Aantal upcoming fixtures voor FDR", min_value=1, max_value=10, value=5)

    st.divider()
    st.subheader("Minutes factor")
    minutes_lookback = st.slider("Laatste wedstrijden (minutes)", min_value=0, max_value=5, value=2)
    minutes_weight = st.slider("Minutes impact (0=uit, 1=hard)", min_value=0.0, max_value=1.0, value=0.7, step=0.05)

    num_transfers = st.slider("Aantal transfers", min_value=1, max_value=5, value=2)

    use_bank_override = st.checkbox("Vrij budget override gebruiken", value=False)
    bank_override_m = None
    if use_bank_override:
        bank_override_m = st.number_input("Vrij budget (m)", min_value=0.0, max_value=50.0, value=0.0, step=0.1)

    top_plans = st.slider("Aantal plannen tonen", min_value=1, max_value=5, value=3)


# ----- Home page -----
if page == "Home":
    st.subheader("Wat doet dit?")
    st.markdown(
        """
**FPL Coach** helpt je met:
- Je huidige team tonen (incl. FDR en availability)
- Top targets buiten je squad
- Multi-transfer plannen (verkoop van premium kan upgrades financieren)

**Belangrijke keuzes:**
- *FDR*: kijkt naar de volgende **N fixtures**, start vanaf **GW+1** (current GW wordt overgeslagen)
- *Minutes factor*: verlaagt de score van spelers met weinig minuten in de laatste wedstrijden

---

### Patch notes (eindgebruikers)
**Nieuw / verbeterd**
- Homepagina met uitleg en patch notes
- FDR op basis van **aantal fixtures** (niet alleen gameweeks) en start vanaf **GW+1**
- Minutes factor: neem **gespeelde minuten** van de laatste wedstrijden mee (bijv. speler met 50/180 min zakt in ranking)
- Multi-transfer plannen: verkoop van duurdere speler kan later extra upgrades mogelijk maken
- Manager presets + Custom ID
- Gameweek fallback als picks voor de gekozen GW nog niet beschikbaar zijn

**Bekende beperking**
- Verkoopprijs is een benadering (publieke data = huidige prijs). Exacte sell-price kan afwijken zonder login.
"""
    )
    st.stop()


# ----- Data loading -----
@st.cache_data(show_spinner=False)
def _load_data():
    players_df, teams_df = load_bootstrap()
    fixtures_df = load_fixtures()
    return players_df, teams_df, fixtures_df


try:
    players_df, teams_df, fixtures_df = _load_data()
except Exception as e:
    st.error(f"Kon FPL data niet laden. Details: {type(e).__name__}: {e}")
    st.stop()


# ----- Resolve GW for entry picks -----
try:
    gw = resolve_gameweek(int(manager_id), int(requested_gw))
    if gw != int(requested_gw):
        st.info(f"Gameweek aangepast naar {gw} (picks voor GW {requested_gw} niet beschikbaar).")
except Exception as e:
    st.error(f"Kon gameweek niet resolven voor manager {manager_id}. Details: {type(e).__name__}: {e}")
    st.stop()


col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("Huidige team")
    try:
        squad_df = show_current_team(
            int(manager_id),
            int(gw),
            players_df,
            teams_df,
            fixtures_df,
            fixtures_ahead=int(fixtures_ahead),
            minutes_lookback=int(minutes_lookback),
            minutes_weight=float(minutes_weight),
        )
        st.dataframe(squad_df, use_container_width=True)
    except Exception as e:
        st.error(f"Kon team niet laden. Details: {type(e).__name__}: {e}")

with col_right:
    st.subheader("Top targets (excl. squad)")
    try:
        targets_df = generate_transfer_suggestions(
            int(manager_id),
            int(gw),
            players_df,
            teams_df,
            fixtures_df,
            fixtures_ahead=int(fixtures_ahead),
            top_n=10,
            minutes_lookback=int(minutes_lookback),
            minutes_weight=float(minutes_weight),
        )
        st.dataframe(targets_df, use_container_width=True)
    except Exception as e:
        st.error(f"Kon targets niet genereren. Details: {type(e).__name__}: {e}")

st.divider()

st.subheader("Transfer plannen")

try:
    plans_df = suggest_transfer_plans(
        manager_id=int(manager_id),
        gameweek=int(gw),
        players_df=players_df,
        teams_df=teams_df,
        fixtures_df=fixtures_df,
        fixtures_ahead=int(fixtures_ahead),
        num_transfers=int(num_transfers),
        free_budget_m=bank_override_m,
        minutes_lookback=int(minutes_lookback),
        minutes_weight=float(minutes_weight),
        top_plans=int(top_plans),
    )

    if plans_df is None or plans_df.empty:
        st.warning("Geen plannen gevonden (budget/posities/team-limiet kan alles blokkeren).")
    else:
        st.dataframe(plans_df, use_container_width=True)

except FPLError as e:
    st.error(str(e))
except Exception as e:
    st.error(f"Plan generatie faalde. Details: {type(e).__name__}: {e}")
