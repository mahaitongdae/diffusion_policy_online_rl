from pathlib import Path
import tempfile


PROJECT_ROOT = Path(__file__).parent.parent.parent
PACKAGE_ROOT = Path(__file__).parent.parent

TEMP_ROOT = Path(tempfile.gettempdir())
