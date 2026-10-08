"""Скачивание XLSX индекса качества жизни (развивай.рф) для всех городов.

API (JSON:API, без аутентификации):
  POST /api/export_files {city_id, index_type: "quality_of_life"} -> id
  GET  /api/export_files/{id} -> status, file_url
  GET  file_url -> xlsx (листы по годам 2015-2026)

Список городов (267): qol_cities_full.json (name, nameEng, lat, lon, region)
— извлечён из RSC-пейлода https://развивай.рф/index-quality-of-life.

Запуск: PYTHONPATH=src .venv/bin/python src/data/qol_download.py
Результат: data/raw/qol_cities/<nameEng>.xlsx + qol_cities_full.json
"""
from __future__ import annotations

import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

API = "https://xn--80aafaxhj3c.xn--p1ai/api"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "data", "raw", "qol_cities")
CITIES = os.path.join(ROOT, "data", "raw", "qol_cities_full.json")
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json",
           "Content-Type": "application/json"}


def _req(url: str, data: bytes | None = None, tries: int = 3) -> bytes:
    for k in range(tries):
        try:
            r = urllib.request.Request(url, data=data, headers=HEADERS)
            with urllib.request.urlopen(r, timeout=60) as resp:
                return resp.read()
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(2 * (k + 1))


def fetch_city(slug: str) -> str:
    dst = os.path.join(OUT, f"{slug}.xlsx")
    if os.path.exists(dst) and os.path.getsize(dst) > 5000:
        return f"{slug}: cached"
    body = json.dumps({"data": {"type": "export_files", "attributes": {
        "export_type": "city_xlsx",
        "parameters": {"city_id": slug, "index_type": "quality_of_life"}}}}).encode()
    j = json.loads(_req(f"{API}/export_files", body))
    eid = j["data"]["id"]
    file_url = None
    for _ in range(30):
        time.sleep(2)
        j = json.loads(_req(f"{API}/export_files/{eid}"))
        a = j["data"]["attributes"]
        if a["status"] == "completed" and a["file_url"]:
            file_url = a["file_url"]
            break
        if a["status"] in ("failed", "expired"):
            return f"{slug}: {a['status']}"
    if not file_url:
        return f"{slug}: timeout"
    blob = _req(file_url)
    tmp = dst + ".part"
    with open(tmp, "wb") as f:
        f.write(blob)
    os.replace(tmp, dst)
    return f"{slug}: {len(blob) // 1024} KB"


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    cities = json.load(open(CITIES, encoding="utf-8"))
    print(f"{len(cities)} cities -> {OUT}")
    done = fail = 0
    with ThreadPoolExecutor(max_workers=5) as ex:
        futs = {ex.submit(fetch_city, c["nameEng"]): c["nameEng"] for c in cities}
        for i, f in enumerate(as_completed(futs), 1):
            msg = f.result()
            if msg.endswith(" KB") or "cached" in msg:
                done += 1
            else:
                fail += 1
                print("FAIL", msg)
            if i % 25 == 0 or i == len(futs):
                print(f"  {i}/{len(futs)} done={done} fail={fail}", flush=True)
    print(f"total: done={done} fail={fail}")


if __name__ == "__main__":
    main()
