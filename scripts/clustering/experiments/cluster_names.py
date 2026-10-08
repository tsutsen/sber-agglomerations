"""Чистка названий МО (clean) — используется в features_exp (mo_names).

Генерация имён кластеров (из топ-3 МО по экономической активности) была
разовой визуальной задачей: её результат зафиксирован в
`data/experiments/inputs/derived/cluster_names.csv`, здесь остались только константы и
`clean`.
"""
from __future__ import annotations


def fmt(v: float) -> str:
    return f"{v / 1e12:.1f} трлн ₽/мес" if v >= 1e12 else f"{v / 1e9:.1f} млрд ₽/мес"
PREFIXES = [
    "внутригородская территория города федерального значения",
    "городской округ", "муниципальный район", "муниципальный округ",
    "рабочий поселок", "поселок", "посёлок", "поселение",
    "город-герой", "город-курорт", "город",
]
SUFFIXES = ["муниципальный район", "муниципальный округ", "городской округ"]


def clean(s: str) -> str:
    s = str(s).strip()
    changed = True
    while changed:
        changed = False
        low = s.lower()
        for p in PREFIXES:
            if low.startswith(p) and len(s) > len(p):
                s = s[len(p):].strip()
                changed = True
                break  # префиксы отсортированы от длинных к коротким; дальше режет баг
        low = s.lower()
        for sfx in SUFFIXES:
            if low.endswith(sfx) and len(s) > len(sfx):
                s = s[: -len(sfx)].strip()
                changed = True
                break
    return s





if __name__ == "__main__":
    main()
