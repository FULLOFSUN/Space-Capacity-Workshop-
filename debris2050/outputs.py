"""Write results in the layout of the workshop output templates:

  output_N_objs.xlsx       sheets Mean, MC 1..MC n
  output_collisions.xlsx   sheets MC 1..MC n, IDs
  output_pc_bins.xlsx      sheets Mean, MC 1..MC n
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo

from .population import CATEGORIES

HEADER_FILL = PatternFill("solid", fgColor="DDEBF7")
BOLD = Font(bold=True)
WRAP = Alignment(wrap_text=True, vertical="top")


def _autosize(ws, min_width: int = 10, max_width: int = 40) -> None:
    for col in ws.columns:
        letter = get_column_letter(col[0].column)
        width = max((len(str(c.value)) for c in col if c.value is not None), default=0)
        ws.column_dimensions[letter].width = max(min_width, min(max_width, width + 2))


def _add_table(ws, ref: str, name: str) -> None:
    t = Table(displayName=name, ref=ref)
    t.tableStyleInfo = TableStyleInfo(name="TableStyleMedium2", showRowStripes=True)
    ws.add_table(t)


def _note(ws, cell: str, merge: str, text: str) -> None:
    ws[cell] = text
    ws[cell].alignment = WRAP
    ws.merge_cells(merge)


# --------------------------------------------------------------------------
def write_n_objs(runs: list[pd.DataFrame], path: Path, note: str) -> None:
    wb = Workbook()
    wb.remove(wb.active)
    mean = pd.concat(runs).groupby("Year", as_index=False)[CATEGORIES].mean()
    sheets = [("Mean", mean)] + [(f"MC {i + 1}", r) for i, r in enumerate(runs)]
    for t_i, (name, df) in enumerate(sheets):
        ws = wb.create_sheet(name)
        ws.append(["Year"] + CATEGORIES + ["Total"])
        for r, (_, row) in enumerate(df.iterrows(), start=2):
            ws.cell(r, 1, int(row["Year"]))
            for c_i, cat in enumerate(CATEGORIES, start=2):
                v = float(row[cat])
                ws.cell(r, c_i, round(v, 2) if name == "Mean" else int(v))
            ws.cell(r, 8, round(float(row[CATEGORIES].sum()), 2))
        _add_table(ws, f"A1:H{len(df) + 1}", f"Table{t_i + 1}")
        _autosize(ws)
        if name == "Mean":
            _note(ws, "J1", "J1:Q12", note)
    wb.save(path)


def write_collisions(runs: list[pd.DataFrame], path: Path, note: str) -> None:
    wb = Workbook()
    wb.remove(wb.active)
    cols = ["Id", "Epoch", "Altitude [km]", "Type of collision", "Objects involved",
            "Constallations involved", "Fragments generated", "Categories involved"]
    for i, df in enumerate(runs):
        ws = wb.create_sheet(f"MC {i + 1}")
        ws.append(cols)
        for row in df[cols].itertuples(index=False):
            ws.append(list(row))
        if len(df):
            _add_table(ws, f"A1:H{len(df) + 1}", f"Collisions{i + 1}")
        else:
            for c in ws[1]:
                c.font, c.fill = BOLD, HEADER_FILL
        _autosize(ws)
        if i == 0:
            _note(ws, "J1", "J1:Q12", note)
    ids = wb.create_sheet("IDs")
    for r, (a, c) in enumerate([("Catastrophic", "Intact-intact"),
                                ("Non Catastrophic", "Intact-fragment"),
                                (None, "Fragment-fragment")], start=1):
        ids.cell(r, 1, a)
        ids.cell(r, 3, c)
    _autosize(ids)
    wb.save(path)


def write_pc_bins(runs: list[dict[int, pd.DataFrame]], path: Path, note: str) -> None:
    years = sorted(runs[0])
    bins = runs[0][years[0]]["Altitude (>)"].tolist()
    mean = {y: pd.concat([r[y] for r in runs]).groupby("Altitude (>)", sort=False)
            [["Intact objects", "Fragments", "Cumulative Pc"]].mean().reset_index()
            for y in years}
    wb = Workbook()
    wb.remove(wb.active)
    for name, data in [("Mean", mean)] + [(f"MC {i + 1}", r) for i, r in enumerate(runs)]:
        ws = wb.create_sheet(name)
        ws.cell(2, 1, "Altitude (>)").font = BOLD
        for r, b in enumerate(bins, start=3):
            ws.cell(r, 1, b)
        for y_i, y in enumerate(years):
            c0 = 2 + 3 * y_i
            ws.cell(1, c0, y).font = BOLD
            ws.cell(1, c0).alignment = Alignment(horizontal="center")
            ws.merge_cells(start_row=1, start_column=c0, end_row=1, end_column=c0 + 2)
            for k, lab in enumerate(["Intact objects", "Fragments", "Cumulative Pc"]):
                h = ws.cell(2, c0 + k, f"{lab} {y}")
                h.font, h.fill = BOLD, HEADER_FILL
                for r, v in enumerate(data[y][lab], start=3):
                    ws.cell(r, c0 + k, float(v) if lab == "Cumulative Pc" else
                            (round(float(v), 2) if name == "Mean" else int(v)))
                    if lab == "Cumulative Pc":
                        ws.cell(r, c0 + k).number_format = "0.000E+00"
        _autosize(ws, min_width=12)
        last = 3 + len(bins) + 1
        _note(ws, f"A{last}", f"A{last}:H{last + 7}", note)
    wb.save(path)
