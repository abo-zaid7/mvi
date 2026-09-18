"""
Build the Chapter 5 GUI report as a Word document.

Run with:   python3 make_gui_docx.py

The styles, fonts, page size and margins are taken from "previous assignment.docx"
so the new report looks exactly like the earlier one - the template is opened,
its body is emptied, and the new content is written into it.  Anything Word
already knew about the document (Times New Roman body text, the blue Heading 1
and Heading 2, the Caption style, Table Grid) therefore carries over untouched.

The Markdown subset handled is the one INDIVIDUAL_REPORT.md actually uses:
headings, paragraphs, bullet lists, pipe tables and [FIG] picture lines.
"""

import os
import re

from docx import Document
from docx.oxml import OxmlElement
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt

TEMPLATE = "previous assignment.docx"
SOURCE = "GUI_REPORT.md"
TARGET = "Chapter 5 - Group Assignment GUI.docx"

TITLE = "Medical Image Segmentation using Image Processing and Deep Learning Techniques"
SUBTITLE = [
    "Chapter 5 - Group Assignment GUI",
    "EE001-3.5-3-MVI Machine Vision",
    "Salman Alshawaf (TP072045) \u00b7 Maryam Afnan (TP079945) \u00b7 "
    "Mohamed Aman (TP079948) \u00b7 Adit Mayen Angony (TP079918)",
]


def blank_template():
    """Open the old report and strip its body, keeping every style definition."""
    document = Document(TEMPLATE)
    body = document.element.body
    for child in list(body):
        # The section properties hold the page size and margins, so they stay.
        if not child.tag.endswith("}sectPr"):
            body.remove(child)
    return document


def add_runs(paragraph, text):
    """Split a line into **bold** and *italic* runs."""
    for piece in re.split(r"(\*\*.+?\*\*|\*[^*]+?\*)", text):
        if not piece:
            continue
        if piece.startswith("**") and piece.endswith("**"):
            paragraph.add_run(piece[2:-2]).bold = True
        elif piece.startswith("*") and piece.endswith("*"):
            paragraph.add_run(piece[1:-1]).italic = True
        else:
            paragraph.add_run(piece)


def apply_numbering(paragraph, num_id=3, level=0):
    """Point a List Paragraph at the numbering the template already defines.

    The old report's lists use numId 3, and numbering.xml survives the body
    being emptied, so reusing that id gives the new lists the same markers and
    the same indents as the ones in the previous report.
    """
    ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    properties = paragraph._p.get_or_add_pPr()
    num_pr = OxmlElement("w:numPr")

    ilvl = OxmlElement("w:ilvl")
    ilvl.set(ns + "val", str(level))
    num_pr.append(ilvl)

    num = OxmlElement("w:numId")
    num.set(ns + "val", str(num_id))
    num_pr.append(num)

    properties.append(num_pr)


def add_table(document, rows):
    header = [c.strip() for c in rows[0].strip().strip("|").split("|")]
    body = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows[2:]]

    table = document.add_table(rows=1, cols=len(header))
    table.style = "Table Grid"
    table.autofit = True

    for cell, name in zip(table.rows[0].cells, header):
        cell.paragraphs[0].text = ""
        run = cell.paragraphs[0].add_run(name)
        run.bold = True

    for line in body:
        cells = table.add_row().cells
        for cell, value in zip(cells, line):
            cell.paragraphs[0].text = ""
            add_runs(cell.paragraphs[0], value)

    # A little air under every table, otherwise the next paragraph sits on it.
    document.add_paragraph()
    return table


def add_figure(document, path, caption):
    if not os.path.exists(path):
        print("  missing figure, skipped:", path)
        return
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    wide = path.endswith(("fig_layout.png", "fig_arabic.jpg",
                          "fig_credits.jpg", "fig_validation.jpg",
                          "fig_views.png", "fig_views_afnan.png", "fig_views_amman.png", "fig_views_adit.png",
                          "fig_adit_task1.jpg", "fig_adit_task2.jpg",
                          "fig_afnan_task1.jpg", "fig_afnan_task2.jpg",
                          "fig_amman_task1.jpg", "fig_amman_task2.jpg"))
    paragraph.add_run().add_picture(path, width=Inches(6.0 if wide else 4.6))

    line = document.add_paragraph(caption, style="Caption")
    line.alignment = WD_ALIGN_PARAGRAPH.CENTER


def add_title_block(document):
    title = document.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run(TITLE)
    run.bold = True
    run.font.size = Pt(16)

    for text in SUBTITLE:
        line = document.add_paragraph()
        line.alignment = WD_ALIGN_PARAGRAPH.CENTER
        line.add_run(text).font.size = Pt(12)

    document.add_paragraph()


def convert():
    document = blank_template()
    add_title_block(document)

    with open(SOURCE, encoding="utf-8") as handle:
        lines = handle.read().split("\n")

    index = 0
    tables = figures = 0

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        if not stripped:
            index += 1
            continue

        # ---------------------------------------------------------- headings
        if stripped.startswith("## "):
            document.add_paragraph(stripped[3:], style="Heading 2")
            index += 1
            continue
        if stripped.startswith("# "):
            document.add_paragraph(stripped[2:], style="Heading 1")
            index += 1
            continue

        # ---------------------------------------------------------- figures
        if stripped.startswith("[FIG]"):
            path, caption = stripped[5:].split("|", 1)
            add_figure(document, path.strip(), caption.strip())
            figures += 1
            index += 1
            continue

        # ---------------------------------------------------------- tables
        if stripped.startswith("|"):
            rows = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                rows.append(lines[index])
                index += 1
            add_table(document, rows)
            tables += 1
            continue

        # ---------------------------------------------------------- bullets
        if stripped.startswith("- "):
            paragraph = document.add_paragraph(style="List Paragraph")
            apply_numbering(paragraph)
            add_runs(paragraph, stripped[2:])
            index += 1
            continue

        # ---------------------------------------------------------- ordinary text
        paragraph = document.add_paragraph()
        add_runs(paragraph, stripped)
        index += 1

    document.save(TARGET)
    print(f"wrote {TARGET}: {tables} tables, {figures} figures")


if __name__ == "__main__":
    convert()
