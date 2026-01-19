import streamlit as st

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

PATCH_NOTES_MD = """
### Patch notes (v0.4)
- Homepagina met uitleg + patch notes.
- Manager presets (Brandon/Elwin/Abdel/Bart/Nick) + Custom ID.
- Gameweek auto-fallback als picks voor een GW nog niet beschikbaar zijn.
- FDR op basis van **N komende fixtures** (start vanaf GW+1).
- Minutes-impact: neem gespeelde minuten uit de laatste wedstrijden mee in scoring.
- Multi-transfer plannen: dure verkoop kan latere upgrades financieren (P1/P2/P3 prioriteit).
"""

HOME_MD = """
## Home

Deze app gebruikt alleen de **publieke** FPL API en doet:

- Je team inladen via Manager ID
- FDR/fixture-swing meenemen op basis van een gekozen aantal fixtures
- Transfer targets tonen (excl. je squad)
- Multi-transfer plannen maken (zodat een premium sale meerdere upgrades kan unlocken)
- Optioneel: minutes-impact (laatste N wedstrijden)

**Tip:** zet "Minutes impact" aan als je spelers wilt vermijden die weinig minuten maken (bv. 50/180).
"""


st.set_page_config(page_title="FPL Coach", layout="wide")

with st.sidebar:
    st.header("Navigatie")
    page = st.radio("Pagina", ["Home", "Coach"], index=1)

    st.divider()
    st.header("Inputs")

    manager_name = st.selectbox("Manager", list(MANAGERS.keys()), index=1)
    default_mid = MANAGERS.get(manager_name) or 1548623
    manager_id = st.number_input("Manager ID", min_value=1, value=int(default_mid))

    # Only load current GW when you are on the coach page (avoid API calls on Home)
    current_gw = None
    if page == "Coach":
        try:
            current_gw = get_current_gameweek()
        except Exception:
            current_gw = 1

    requested_gw = st.number_input("Gameweek", min_value=1, value=int(current_gw or 1))

    fixtures_ahead = st.slider("Aantal upcoming fixtures voor FDR", min_value=1, max_value=10, value=5)

    st.subheader("Minutes impact")
    minutes_lookback = st.slider("Laatste wedstrijden", min_value=0, max_value=5, value=2)
    minutes_weight = st.slider("Impact (0=uit, 1=hard)", min_value=0.0, max_value=1.0, value=0.7, step=0.05)

    st.subheader("Transfers")
    num_transfers = st.slider("Aantal transfers", min_value=1, max_value=5, value=2)

    use_bank_override = st.checkbox("Vrij budget override gebruiken", value=False)
    bank_override_m = None
    if use_bank_override:
        bank_override_m = st.number_input("Vrij budget (m)", min_value=0.0, max_value=50.0, value=0.0, step=0.1)

    top_plans = st.slider("Aantal plannen tonen", min_value=1, max_value=5, value=3)


st.title("FPL Coach")

if page == "Home":
    st.markdown(HOME_MD)
    with st.expander("Patch notes"):
        st.markdown(PATCH_NOTES_MD)
    st.stop()


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
            manager_id=int(manager_id),
            gameweek=int(gw),
            players_df=players_df,
            teams_df=teams_df,
            fixtures_df=fixtures_df,
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
            manager_id=int(manager_id),
            gameweek=int(gw),
            players_df=players_df,
            teams_df=teams_df,
            fixtures_df=fixtures_df,
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
        top_plans=int(top_plans),
        minutes_lookback=int(minutes_lookback),
        minutes_weight=float(minutes_weight),
    )

    if plans_df is None or plans_df.empty:
        st.warning("Geen plannen gevonden (budget/posities/team-limiet kan alles blokkeren).")
    else:
        st.dataframe(plans_df, use_container_width=True)

except FPLError as e:
    st.error(str(e))
except Exception as e:
    st.error(f"Plan generatie faalde. Details: {type(e).__name__}: {e}")
