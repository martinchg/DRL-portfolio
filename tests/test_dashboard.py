# Smoke test du dashboard Streamlit : le script doit s'exécuter sans exception
# avec les valeurs par défaut des widgets (télécharge AAPL via yfinance).
import os

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DASHBOARD = os.path.join(PROJECT_ROOT, "dashboard.py")


@pytest.mark.network
@pytest.mark.slow
def test_dashboard_runs_without_exception():
    at = AppTest.from_file(DASHBOARD, default_timeout=300)
    at.run()

    assert not at.exception, f"Exception dans le dashboard : {at.exception}"
