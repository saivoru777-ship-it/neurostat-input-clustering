"""pytest conftest: ensure neurostat and project src are on the path."""
import sys
from pathlib import Path

# Add neurostat to path so `import neurostat` works
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "neurostat"))
# Add project root so `import src` works
sys.path.insert(0, str(Path(__file__).resolve().parent))
