"""
Build the individual report (Task 1 and Task 2) as a Word document.

Run with:   python3 make_individual_docx.py

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
import struct

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt

TEMPLATE = "previous assignment.docx"
SOURCE = "INDIVIDUAL_REPORT.md"
TARGET = "Individual Report - Task 1 and Task 2.docx"

TITLE = "Medical Image Segmentation using Image Processing and Deep Learning Techniques"
SUBTITLE = [
    "Individual Components - Task 1 and Task 2",
    "Salman Alshawaf (TP072045)",
    "EE001-3.5-3-MVI Machine Vision",
    "Dataset: BRISC 2025 - Brain Tumour MRI",
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


def image_size(path):
    """Pixel width and height, read out of the file's own header."""
    with open(path, "rb") as handle:
        head = handle.read(32)
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        return struct.unpack(">II", head[16:24])
    # JPEG: walk the segments until the frame header gives the size
    with open(path, "rb") as handle:
        data = handle.read()
    index = 2
    while index < len(data) - 9:
        if data[index] != 0xFF:
            index += 1
            continue
        marker, length = data[index + 1], struct.unpack(">H", data[index + 2:index + 4])[0]
        if marker in (0xC0, 0xC1, 0xC2, 0xC3):
            height, width = struct.unpack(">HH", data[index + 5:index + 9])
            return width, height
        index += 2 + length
    return 1, 1


# A4 with the template's one inch margins leaves this much text width.
PAGE_WIDTH = 6.2
MAX_HEIGHT = 8.2


def add_figure(document, path, caption):
    if not os.path.exists(path):
        print("  missing figure, skipped:", path)
        return

    # A flowchart that is three times wider than it is tall is unreadable at the
    # width a square scan wants, so every figure is sized from its own shape:
    # wide ones take the full text width, tall ones are capped by page height.
    pixel_w, pixel_h = image_size(path)
    aspect = pixel_w / float(pixel_h or 1)
    width = PAGE_WIDTH if aspect >= 1.25 else 4.6
    if width / aspect > MAX_HEIGHT:
        width = MAX_HEIGHT * aspect

    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.add_run().add_picture(path, width=Inches(width))

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


def add_figure_index(document):
    """
    The List of Figures the earlier report opened with.

    Word builds it from the Caption paragraphs when the field is updated, which
    is why the page numbers cannot be written here - select all and press F9, or
    right-click the list and choose "Update field", once the document is open.
    """
    document.add_paragraph("List of Figures", style="Heading 1")

    paragraph = document.add_paragraph()
    run = paragraph.add_run()

    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instruction = OxmlElement("w:instrText")
    instruction.set(qn("xml:space"), "preserve")
    instruction.text = ' TOC \\h \\z \\c "Figure" '
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    placeholder = OxmlElement("w:t")
    placeholder.text = "Right-click here and choose Update field to build the list."
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")

    for node in (begin, instruction, separate, placeholder, end):
        run._r.append(node)

    document.add_paragraph()


def convert():
    document = blank_template()
    add_title_block(document)
    add_figure_index(document)

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
