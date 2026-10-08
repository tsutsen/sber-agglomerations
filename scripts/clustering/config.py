"""Гиперпараметры пайплайна: единственный источник — config.yaml рядом с этим файлом."""
from __future__ import annotations

import os

import yaml

_CFG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
with open(_CFG, encoding="utf-8") as _f:
    CFG = yaml.safe_load(_f)
