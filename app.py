
import runpy
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CORE_FILE = BASE_DIR / "src" / "fraudshield_core.py"

runpy.run_path(
    str(CORE_FILE),
    init_globals={"page": "🏠 app"}
)

