from pathlib import Path
import runpy

CORE = Path(__file__).resolve().parent.parent / "src" / "fraudshield_core.py"
runpy.run_path(str(CORE), init_globals={"page": '🔮 Fraud Forecasting'})
