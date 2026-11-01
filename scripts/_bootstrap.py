"""Imported first by every script.

1. Force Python's UTF-8 mode on Windows.
   Windows Python defaults text I/O to the ANSI code page (cp1252), so any library
   that opens a file without `encoding=` (ir_datasets does, for MS MARCO) crashes
   on the first non-ASCII byte. UTF-8 mode (PEP 540) makes UTF-8 the default.
   It can only be switched on at interpreter start, so if it's off we re-run the
   same command in a child interpreter with `-X utf8` and pass its exit code on.

2. Make `import searchengine` work even without `pip install -e .`.
"""

import os
import subprocess
import sys
from pathlib import Path

if os.name == "nt" and not sys.flags.utf8_mode:
    env = dict(os.environ, PYTHONUTF8="1")
    sys.exit(subprocess.call([sys.executable, "-X", "utf8", *sys.argv], env=env))

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
