"""Статистика сравнений: парный бутстрэп по объектам (вопросам, запросам,
последовательностям) и по парам «сид × объект». Только numpy, без MLX."""
from .paired import (Interval, paired_bootstrap, item_means, reciprocal_ranks,
                     load_per_row, compare_table)

__all__ = ["Interval", "paired_bootstrap", "item_means", "reciprocal_ranks",
           "load_per_row", "compare_table"]
