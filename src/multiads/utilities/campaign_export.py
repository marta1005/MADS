"""Small CSV/XLSX/JSON writers for MADS campaign outputs."""

from __future__ import annotations

import csv
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import numpy as np


@dataclass(frozen=True, slots=True)
class CampaignExportPaths:
    """Paths written for one tabular MADS campaign export."""

    dataset_csv: Path
    flat_dataset_csv: Path
    design_space_csv: Path
    workbook_xlsx: Path
    manifest_json: Path


def campaign_export_paths(output_dir: str | Path, file_prefix: str) -> CampaignExportPaths:
    """Return the standard output paths for a campaign prefix."""

    output_path = Path(output_dir)
    return CampaignExportPaths(
        dataset_csv=output_path / f"{file_prefix}_dataset.csv",
        flat_dataset_csv=output_path / f"{file_prefix}_dataset_flat.csv",
        design_space_csv=output_path / f"{file_prefix}_design_space.csv",
        workbook_xlsx=output_path / f"{file_prefix}_results.xlsx",
        manifest_json=output_path / f"{file_prefix}_manifest.json",
    )


def flatten_dataset_columns(dataset):  # noqa: ANN001, ANN201
    """Return a copy of a GEMSEO dataset with simple string column names."""

    flat_dataset = dataset.copy()
    flat_columns: list[str] = []
    for column in flat_dataset.columns:
        if isinstance(column, tuple):
            group, variable, component = column
            suffix = "" if str(component) in {"", "0"} else f".{component}"
            flat_columns.append(f"{group}.{variable}{suffix}")
        else:
            flat_columns.append(str(column))
    flat_dataset.columns = flat_columns
    return flat_dataset


def dataframe_rows(dataframe) -> list[list[object]]:  # noqa: ANN001
    """Convert a dataframe-like object to worksheet rows."""

    safe_values = dataframe.astype(object).where(dataframe.notna(), None)
    return [list(dataframe.columns), *safe_values.values.tolist()]


def write_design_space_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Write design-space rows to CSV."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["name", "baseline", "lower_bound", "upper_bound"],
        )
        writer.writeheader()
        writer.writerows(rows)


def _xlsx_column_name(index: int) -> str:
    name = ""
    index += 1
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def _xlsx_cell(value, row: int, column: int) -> str:  # noqa: ANN001
    ref = f"{_xlsx_column_name(column)}{row}"
    if value is None:
        return f'<c r="{ref}"/>'
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return f'<c r="{ref}" t="b"><v>{1 if value else 0}</v></c>'
    if isinstance(value, (int, float, np.integer, np.floating)):
        value_float = float(value)
        if not np.isfinite(value_float):
            return f'<c r="{ref}"/>'
        return f'<c r="{ref}"><v>{value_float:.15g}</v></c>'
    return f'<c r="{ref}" t="inlineStr"><is><t>{escape(str(value))}</t></is></c>'


def _xlsx_sheet_xml(rows: list[list[object]]) -> str:
    xml_rows: list[str] = []
    for row_index, row in enumerate(rows, start=1):
        cells = "".join(_xlsx_cell(value, row_index, column_index) for column_index, value in enumerate(row))
        xml_rows.append(f'<row r="{row_index}">{cells}</row>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f"<sheetData>{''.join(xml_rows)}</sheetData>"
        "</worksheet>"
    )


def write_xlsx_workbook(path: str | Path, sheets: dict[str, list[list[object]]]) -> None:
    """Write a minimal XLSX workbook without optional spreadsheet dependencies."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet_items = list(sheets.items())
    workbook_sheets = "".join(
        f'<sheet name="{escape(name[:31])}" sheetId="{idx}" r:id="rId{idx}"/>'
        for idx, (name, _) in enumerate(sheet_items, start=1)
    )
    workbook_rels = "".join(
        (
            f'<Relationship Id="rId{idx}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{idx}.xml"/>'
        )
        for idx in range(1, len(sheet_items) + 1)
    )
    workbook_rels += (
        f'<Relationship Id="rId{len(sheet_items) + 1}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
        'Target="styles.xml"/>'
    )
    overrides = "".join(
        (
            f'<Override PartName="/xl/worksheets/sheet{idx}.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        )
        for idx in range(1, len(sheet_items) + 1)
    )
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "[Content_Types].xml",
            (
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                '<Default Extension="xml" ContentType="application/xml"/>'
                '<Override PartName="/xl/workbook.xml" '
                'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
                '<Override PartName="/xl/styles.xml" '
                'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
                f"{overrides}</Types>"
            ),
        )
        archive.writestr(
            "_rels/.rels",
            (
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                '<Relationship Id="rId1" '
                'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
                'Target="xl/workbook.xml"/>'
                "</Relationships>"
            ),
        )
        archive.writestr(
            "xl/workbook.xml",
            (
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
                'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
                f"<sheets>{workbook_sheets}</sheets></workbook>"
            ),
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            (
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                f"{workbook_rels}</Relationships>"
            ),
        )
        archive.writestr(
            "xl/styles.xml",
            (
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
                '<fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>'
                '<fills count="1"><fill><patternFill patternType="none"/></fill></fills>'
                '<borders count="1"><border/></borders>'
                '<cellStyleXfs count="1"><xf/></cellStyleXfs>'
                '<cellXfs count="1"><xf xfId="0"/></cellXfs>'
                "</styleSheet>"
            ),
        )
        for idx, (_, rows) in enumerate(sheet_items, start=1):
            archive.writestr(f"xl/worksheets/sheet{idx}.xml", _xlsx_sheet_xml(rows))


def write_campaign_results(
    *,
    output_dir: str | Path,
    file_prefix: str,
    dataset,
    design_space_rows: list[dict[str, Any]],
    analysis_settings_rows: list[list[object]],
    manifest: dict[str, Any],
    paths: CampaignExportPaths | None = None,
) -> CampaignExportPaths:
    """Write the standard CSV/XLSX/JSON files for a campaign."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    export_paths = paths or campaign_export_paths(output_path, file_prefix)

    dataset.to_csv(path_or_buf=str(export_paths.dataset_csv))
    flat_dataset = flatten_dataset_columns(dataset)
    flat_dataset.to_csv(path_or_buf=str(export_paths.flat_dataset_csv), index=False)
    write_design_space_csv(export_paths.design_space_csv, design_space_rows)
    write_xlsx_workbook(
        export_paths.workbook_xlsx,
        {
            "results": dataframe_rows(flat_dataset),
            "design_space": [
                ["name", "baseline", "lower_bound", "upper_bound"],
                *[
                    [row["name"], row["baseline"], row["lower_bound"], row["upper_bound"]]
                    for row in design_space_rows
                ],
            ],
            "analysis_settings": analysis_settings_rows,
        },
    )
    export_paths.manifest_json.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return export_paths
