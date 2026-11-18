"""Read permitted development files directly from ZIP; never execute archive content."""
import argparse
import hashlib
import json
import zipfile
from collections import defaultdict
from pathlib import Path


def inspect(archive):
    results = []
    with zipfile.ZipFile(archive) as source:
        groups = defaultdict(dict)
        for name in source.namelist():
            parts = name.split('/')
            if len(parts) != 4 or parts[1] not in {'train', 'student_test'}:
                continue
            extension = parts[-1].rsplit('.', 1)[-1]
            if extension in {'track', 'nodes', 'arcs', 'route'}:
                groups[(parts[1], parts[2])][extension] = name
        for (partition, trip), files in sorted(groups.items()):
            if set(files) != {'track', 'nodes', 'arcs', 'route'}:
                raise ValueError(f'Incomplete trip: {partition}/{trip}')
            raw = {ext: source.read(name) for ext, name in files.items()}
            rows = {ext: [line.split() for line in data.decode().splitlines() if line.strip()]
                    for ext, data in raw.items()}
            arcs = [tuple(map(int, row)) for row in rows['arcs']]
            route = [int(row[0]) for row in rows['route']]
            times = [float(row[2]) for row in rows['track']]
            arc_set = set(arcs)
            node_count = len(rows['nodes'])
            results.append({
                'partition': partition, 'trip': trip,
                'rows': {ext: len(value) for ext, value in rows.items()},
                'widths': {ext: sorted(set(map(len, value))) for ext, value in rows.items()},
                'network_hash': hashlib.sha256(raw['nodes'] + raw['arcs']).hexdigest(),
                'invalid_arc_nodes': sum(u < 0 or v < 0 or u >= node_count or v >= node_count
                                         for u, v in arcs),
                'invalid_route_ids': sum(i < 0 or i >= len(arcs) for i in route),
                'nonincreasing_timestamps': sum(b <= a for a, b in zip(times, times[1:])),
                'max_gap_s': max((b - a for a, b in zip(times, times[1:])), default=0),
                'reverse_arc_fraction': sum((v, u) in arc_set for u, v in arcs) / len(arcs),
                'directed_route_breaks': sum(arcs[a][1] != arcs[b][0]
                    for a, b in zip(route, route[1:]) if 0 <= a < len(arcs) and 0 <= b < len(arcs)),
            })
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive')
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('structural_inventory.json'))
    args = parser.parse_args()
    result = inspect(args.archive)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f'Inspected {len(result)} development trips; saved {args.output}')
