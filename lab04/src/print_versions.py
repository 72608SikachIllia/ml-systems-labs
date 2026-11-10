import platform
import sys
from importlib import metadata

from common import LAB_DIR, RESULTS

REQ = LAB_DIR.parent / "requirements.txt"


def main():
    lines = [
        f"Python {sys.version.split()[0]} ({platform.python_implementation()})",
        f"Platform {platform.platform()}",
        "",
    ]
    for raw in REQ.read_text().splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        name, pinned = raw.split("==", 1) if "==" in raw else (raw, "")
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            installed = "NOT INSTALLED"
        flag = "" if installed == pinned else f"   <-- pinned {pinned}"
        lines.append(f"{name}=={installed}{flag}")
    text = "\n".join(lines)
    print(text)
    (RESULTS / "versions.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
