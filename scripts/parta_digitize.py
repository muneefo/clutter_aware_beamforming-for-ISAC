"""Digitize the plotted vectors of arXiv:2602.10537v2 Fig. 12(b)."""
import argparse
import csv
import hashlib
import re
import subprocess
import tempfile
from pathlib import Path
from xml.etree import ElementTree


COLORS = {"rgb(0%, 44.700623%, 74.021912%)": 4,
          "rgb(46.702576%, 67.576599%, 18.797302%)": 8,
          "rgb(49.388123%, 18.406677%, 55.664062%)": 12}
DIGEST = "e7f1e9b262e73deb721bbe6439c09c886eaf2777e6617aa9ed79aa5576b234b6"


def extract(pdf):
    with pdf.open("rb") as inp:
        assert hashlib.file_digest(inp, "sha256").hexdigest() == DIGEST
    with tempfile.TemporaryDirectory() as tmp:
        svg = Path(tmp) / "page.svg"
        subprocess.run(["pdftocairo", "-f", "29", "-l", "29", "-svg", pdf, svg], check=True)
        root = ElementTree.parse(svg).getroot()
    curves = {}
    for path in root.iter("{http://www.w3.org/2000/svg}path"):
        data = path.get("d", "")
        color = path.get("stroke")
        if color not in COLORS:
            continue
        xy = [(float(x), float(y)) for x, y in re.findall(r"[ML] ([\d.]+) ([\d.]+)", data)]
        if len(xy) == 6 and 909 < xy[0][0] < 911 and 3950 < xy[0][1] < 5390:
            label = "estimated" if path.get("stroke-dasharray") else "perfect"
            curves[COLORS[color], label] = xy
    assert len(curves) == 6, f"Unexpected set of paths: {curves.keys()}"
    # Coordinates of the grid lines adjacent to the 14.0 and 15.5 dB labels.
    lower, upper = 4193.203125, 4786.210938
    for (users, label), points in sorted(curves.items()):
        for gamma, (x, y) in zip(range(0, 21, 4), points):
            yield (users, label, gamma, x, y, lower, upper, round(14 + (y - lower) * 1.5 / (upper - lower), 2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("csv", type=Path)
    args = parser.parse_args()
    with args.csv.open("x", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(("users", "design_kernel", "gamma_db", "x", "y", "y_14db", "y_15_5db", "digitized_scnr_db"))
        writer.writerows(extract(args.pdf))


if __name__ == "__main__":
    main()
