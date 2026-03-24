from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "robotic_warehouse" / "__init__.py"
SPEC = spec_from_file_location("robotic_warehouse_root", MODULE_PATH)
MODULE = module_from_spec(SPEC)
sys.modules.setdefault("robotic_warehouse_root", MODULE)
SPEC.loader.exec_module(MODULE)

RwareLegacyGymWrapper = MODULE.RwareLegacyGymWrapper
