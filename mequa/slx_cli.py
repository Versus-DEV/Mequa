import argparse
import csv
from pathlib import Path

from mequa import backend, couple, simulink, slx_demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--set", dest="assign", nargs="*", default=[])
    parser.add_argument("--t-end", type=float, default=6.0)
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--signal", default=None)
    parser.add_argument("--couple", action="store_true")
    parser.add_argument("--backend", choices=["julia", "rust", "numpy"], default="julia")
    parser.add_argument("--samples", type=int, default=20)
    args = parser.parse_args()

    variables = simulink.parse_assignments(" ".join(args.assign))
    if args.demo:
        path = slx_demo.make_demo_slx("data/demo_windkessel.slx")
        variables = {**slx_demo.DEMO_VARIABLES, **variables}
    elif args.model:
        path = args.model
    else:
        parser.error("give a .slx path or --demo")
    try:
        model = simulink.load_slx(path)
        print(f"model: {model.name}, {len(model.blocks)} blocks, {len(model.wires)} connections")
        result = simulink.simulate(model, t_end=args.t_end, dt=args.dt, variables=variables)
    except simulink.ModelError as exc:
        print(f"Import failed: {exc}")
        return
    name = args.signal or next(iter(result.signals))
    signal = result.signals[name]
    print(f"signal '{name}': min {signal.min():.4f}, max {signal.max():.4f}, mean {signal.mean():.4f}")
    Path("results").mkdir(exist_ok=True)
    with open("results/simulink_signal.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["time", name])
        writer.writerows(zip(result.time, signal))
    if args.couple:
        be = backend.load(args.backend)
        points = be.synthetic_cells(2000)
        rows = couple.drive_strain(result.time, signal, points, be, samples=args.samples)
        with open("results/simulink_coupled.csv", "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"backend: {be.name}")
        for r in rows:
            print(f"t={r['time']:.2f}  signal={r['signal']:.3f}  load={r['load']:.3f}  SPHI={r['sphi']:.4f}  peak strain={r['peak_strain']:.4f}")


if __name__ == "__main__":
    main()
