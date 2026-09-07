"""Makes ``app`` resolve to ``servicio_ia/app``, not the root business backend's ``app``.

Both apps in this monorepo are literally named ``app``. Production runs the AI
service with ``uvicorn app.main:app --app-dir servicio_ia`` (see servicio_ia/README.md);
inserting the same ``servicio_ia`` directory onto ``sys.path`` here mirrors that
at collection time, so these tests import ``servicio_ia/app`` and never collide
with the root ``app`` package used by the root test suite.
"""

import sys
from pathlib import Path

SERVICIO_IA_DIR = Path(__file__).resolve().parent.parent

if str(SERVICIO_IA_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICIO_IA_DIR))
