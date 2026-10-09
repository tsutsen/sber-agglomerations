#!/usr/bin/env python3
"""Собирает map.html из template.html, вшивая GeoJSON-данные.

Запуск из корня проекта:  python build.py
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent

def compact(gj_path: Path) -> str:
    try:
        d = json.loads(gj_path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise SystemExit(f"ошибка: не удалось прочитать {gj_path}: {e}") from e
    return json.dumps(d, ensure_ascii=False, separators=(",", ":"))

def find(name: str) -> Path:
    """GeoJSON ищем в ../data, потом рядом с build.py — папку landing можно пересылать."""
    for p in (HERE.parent / "data" / name, HERE / name):
        if p.exists():
            return p
    raise SystemExit(f"ошибка: {name} не найден (ни в data/, ни в landing/)")


def main() -> int:
    tpl = (HERE / "template.html").read_text()
    outlines = compact(find("cluster_outlines.geojson"))
    mo = compact(find("clusters.geojson"))
    official = compact(find("official.geojson"))
    for marker, data in (("__OUTLINES__", outlines), ("__GJ__", mo), ("__OFFICIAL__", official)):
        pat = rf"/\*{marker}\*/null;"
        if not re.search(pat, tpl):
            print(f"ошибка: маркер {marker} не найден в template.html", file=sys.stderr)
            return 1
        tpl = re.sub(pat, lambda m, d=data, mk=marker: "/*" + mk + "*/" + d + ";", tpl, count=1)
    (HERE / "map.html").write_text(tpl)
    print(f"map.html собран: {len(tpl)//1024} KB "
          f"(outlines {len(outlines)//1024} KB + mo {len(mo)//1024} KB + official {len(official)//1024} KB)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
