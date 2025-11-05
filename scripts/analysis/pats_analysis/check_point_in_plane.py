import argparse, json, sys
from typing import List, Dict, Tuple
import numpy as np

# ------------------------------ default planes ------------------------------
DEFAULT_PLANES: List[Dict] = [
    {
        "n": [0.6408563820557884, 0.7071067811865476, 0.29883623873011983],
        "p0": [0.0, 0.0, 0.0],
    },
    {
        "n": [0.6408563820557884, -0.7071067811865476, 0.29883623873011983],
        "p0": [0.0, 0.0, 0.0],
    },
    {"n": [0.8191520442889917, 0.0, -0.573576436351046], "p0": [0.0, 0.0, 0.0]},
    {"n": [0.08715574274765808, 0.0, 0.9961946980917455], "p0": [0.0, 0.0, 0.0]},
    {
        "n": [0.9063077870366499, 0.0, 0.42261826174069944],
        "p0": [0.36252311481466, 0.0, 0.1690473046962798],
    },
    {
        "n": [-0.9063077870366499, -0.0, -0.42261826174069944],
        "p0": [2.7189233611099497, 0.0, 1.2678547852220983],
    },
    {
        "n": [0.0, 0.0, -1.0],
        "p0": [0.0, 0.0, 1.5],
    },  # roof (inward normal pointing downward)
    {
        "n": [0.0, 0.0, 1.0],
        "p0": [0.0, 0.0, 0.0],
    },  # floor (inward normal pointing upward)
    {
        "n": [-1.0, 0.0, 0.0],
        "p0": [2.5, 0.0, 0.0],
    },  # back/front wall depending on your N axis
]


# ------------------------------ core logic ----------------------------------
def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v if n == 0 else v / n


def check_point_against_planes(
    point: np.ndarray, planes: List[Dict], margin: float = 0.0
) -> Tuple[bool, List[Dict]]:
    """
    Returns:
      inside: True if point is inside all planes with margin
      details: list of dicts per plane with signed distance & pass/fail
    Inside condition (your convention):
        s_i = n_i_hat · (x - p0_i)  >=  margin
    """
    details = []
    inside_all = True

    for idx, pl in enumerate(planes):
        n = np.asarray(pl["n"], dtype=float)
        p0 = np.asarray(pl["p0"], dtype=float)
        n_hat = normalize(n)
        s = float(np.dot(n_hat, point - p0))  # signed distance in meters
        ok = s >= margin
        if not ok:
            inside_all = False
        details.append(
            {
                "index": idx,
                "n_hat": n_hat.tolist(),
                "p0": p0.tolist(),
                "signed_distance": s,
                "required_margin": margin,
                "passes": ok,
                "name": pl.get("name", f"plane_{idx}"),
            }
        )

    return inside_all, details


def pretty_print_result(point: np.ndarray, inside: bool, details: List[Dict]):
    print(f"\nPoint x = [{point[0]:.6f}, {point[1]:.6f}, {point[2]:.6f}]")
    print(f"Result : {'INSIDE' if inside else 'OUT OF BOUNDS'}\n")

    # Table header
    print(
        f"{'idx':>3}  {'name':<16} {'s = n·(x-p0) [m]':>18}  {'>= margin?':>10}  {'n_hat':>24}  {'p0':>24}"
    )
    print("-" * 100)

    for d in details:
        print(
            f"{d['index']:>3}  {d['name']:<16} {d['signed_distance']:>18.6f}  "
            f"{'OK' if d['passes'] else 'FAIL':>10}  "
            f"{str(np.array(d['n_hat'])):>24}  {str(np.array(d['p0'])):>24}"
        )

    # Explicit list of violations
    bad = [d for d in details if not d["passes"]]
    if bad:
        print("\nViolated planes:")
        for d in bad:
            deficit = d["required_margin"] - d["signed_distance"]
            print(
                f"  - {d['name']} (idx {d['index']}): s={d['signed_distance']:.6f} m "
                f"< margin={d['required_margin']:.6f} m "
                f"(short by {deficit:.6f} m)"
            )
    else:
        print("\nAll planes satisfied.")


# ------------------------------ CLI ----------------------------------------
def load_planes(path: str | None) -> List[Dict]:
    if path is None:
        return DEFAULT_PLANES
    with open(path, "r") as f:
        data = json.load(f)
    # Allow either {"PLANES":[...]} or just [...]
    if isinstance(data, dict) and "PLANES" in data:
        return data["PLANES"]
    if isinstance(data, list):
        return data
    raise ValueError(
        "Unsupported planes file format. Use a list of {n,p0} or a dict with key 'PLANES'."
    )


def main():
    ap = argparse.ArgumentParser(
        description="Check if a 3D point is inside a convex region defined by planes (n·(x-p0) >= margin)."
    )
    ap.add_argument("x", type=float, help="x coordinate")
    ap.add_argument("y", type=float, help="y coordinate")
    ap.add_argument("z", type=float, help="z coordinate")
    ap.add_argument(
        "--margin",
        type=float,
        default=0.0,
        help="required safety margin (meters) inside each plane",
    )
    ap.add_argument(
        "--planes-file",
        type=str,
        default=None,
        help="JSON file with planes (list of {n:[..], p0:[..]}) or dict with key 'PLANES'",
    )
    args = ap.parse_args()

    point = np.array([args.x, args.y, args.z], dtype=float)
    planes = load_planes(args.planes_file)

    inside, details = check_point_against_planes(point, planes, margin=args.margin)
    pretty_print_result(point, inside, details)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
