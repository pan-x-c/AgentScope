# -*- coding: utf-8 -*-
"""Competition package"""
from .competition import Competition
from .knockout import Knockout
from .ucb import LUCB
from .league import League

__all__ = ["Competition", "Knockout", "LUCB", "League"]
