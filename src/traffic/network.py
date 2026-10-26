"""Network topology, generated from (rows, cols) so any grid size works.

Link model: every intersection has 4 incoming approach links, indexed
    link = intersection * 4 + heading,     heading in {N=0, E=1, S=2, W=3}
where heading is the direction of TRAVEL (a car heading N arrives from the south).
A link runs from the upstream intersection's stop line (pos 0) to this intersection's
stop line (pos = link_length). Links with no upstream intersection are ENTRY links
(sensor + demand source); a car leaving through a side with no neighbour EXITS.

Phases: 0 = N/S headings green, 1 = E/W headings green.
Turns:  0 = straight, 1 = left, 2 = right.
"""
from dataclasses import dataclass

import numpy as np

N, E, S, W = 0, 1, 2, 3
HEADINGS = ("N", "E", "S", "W")
DELTA = ((-1, 0), (0, 1), (1, 0), (0, -1))     # (d_row, d_col) for each heading; row 0 = top
TURN_DHEADING = (0, -1, 1)                     # straight, left, right
PHASE_OF_HEADING = (0, 1, 0, 1)


@dataclass(eq=False)
class Network:
    """Grid topology. Link id = intersection*4 + heading (N,E,S,W); entry/exit links are the grid boundary."""
    rows: int
    cols: int
    n_int: int
    n_links: int
    link_dest: np.ndarray       # (n_links,) intersection this link feeds
    link_heading: np.ndarray    # (n_links,)
    link_phase: np.ndarray      # (n_links,) phase during which this link is green
    next_link: np.ndarray       # (n_links, 3) link after a straight/left/right turn, -1 = exit
    entry_links: np.ndarray     # (n_entries,) links fed by demand
    int_rc: np.ndarray          # (n_int, 2) row, col of each intersection


def build_grid(rows: int = 2, cols: int = 2) -> Network:
    n_int = rows * cols
    n_links = 4 * n_int
    link_dest = np.repeat(np.arange(n_int), 4)
    link_heading = np.tile(np.arange(4), n_int)
    link_phase = np.array(PHASE_OF_HEADING)[link_heading]
    int_rc = np.array([(i // cols, i % cols) for i in range(n_int)])

    def idx(r, c):
        return r * cols + c if (0 <= r < rows and 0 <= c < cols) else -1

    next_link = -np.ones((n_links, 3), dtype=np.int64)
    entry = []
    for i in range(n_int):
        r, c = int_rc[i]
        for h in range(4):
            link = i * 4 + h
            for t in range(3):
                h2 = (h + TURN_DHEADING[t]) % 4
                j = idx(r + DELTA[h2][0], c + DELTA[h2][1])
                next_link[link, t] = j * 4 + h2 if j >= 0 else -1
            if idx(r - DELTA[h][0], c - DELTA[h][1]) < 0:
                entry.append(link)
    return Network(rows, cols, n_int, n_links, link_dest, link_heading, link_phase,
                   next_link, np.array(entry, dtype=np.int64), int_rc)
