"""Site plumbing for formula-a-day. Two lines of glue, nothing more.

Each day file starts with

    import fad; fad.day(__file__)

which points jaxvis at docs/<this file's slug>/. Everything after that is
plain jaxvis -- `@jaxvis.draw` does the rendering, the caching and the naming,
so the code on the published page is the code anyone would write.
"""
from pathlib import Path

import jaxvis

ROOT = Path(__file__).resolve().parent


def day(path):
    "Send this day file's renders into docs/<slug>/."
    return jaxvis.output_to(ROOT / "docs" / Path(path).stem)
