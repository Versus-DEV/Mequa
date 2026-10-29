import json
import os
import sys

import numpy as np

PREFIX = "@@MEQUA@@ "


def _load_backend():
    choice = os.environ.get("MEQUA_WORKER_BACKEND")
    if choice == "numpy":
        from mequa import numpy_backend as be
    elif choice == "rust":
        from mequa import rust_backend as be
    else:
        from mequa import julia_bridge as be
    return be


def main():
    be = _load_backend()
    out = sys.stdout

    def send(obj):
        out.write(PREFIX + json.dumps(obj) + "\n")
        out.flush()

    send({"ready": True})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            if req["op"] == "quit":
                send({"ok": True})
                break
            if req["op"] == "analyze":
                data = np.load(req["points"])
                res = be.analyze(data, k=req["k"], knn=req["knn"], load=req["load"])
                np.savez(req["out"], **res)
                send({"ok": True})
            elif req["op"] == "synthetic":
                pts = be.synthetic_cells(req["n"], seed=req["seed"], clusters=req["clusters"])
                np.save(req["out"], pts)
                send({"ok": True})
            else:
                send({"ok": False, "error": "unknown op"})
        except Exception as exc:
            send({"ok": False, "error": str(exc)})


if __name__ == "__main__":
    main()
