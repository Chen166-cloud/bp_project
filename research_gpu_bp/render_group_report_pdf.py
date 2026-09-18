"""Render the BP/FFBP group-meeting Markdown report as a polished A4 PDF."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "research_gpu_bp" / "组会汇报_BP与FFBP_CPU_GPU对比.md"
DEFAULT_OUTPUT = ROOT / "output" / "pdf" / "组会汇报_BP与FFBP_CPU_GPU对比.pdf"
OUTPUT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_OUTPUT

FONT_REGULAR = Path(r"C:\Windows\Fonts\msyh.ttc")
FONT_BOLD = Path(r"C:\Windows\Fonts\msyhbd.ttc")
FONT_SERIF = Path(r"C:\Windows\Fonts\simsun.ttc")


def register_fonts() -> None:
    pdfmetrics.registerFont(TTFont("MicrosoftYaHei", str(FONT_REGULAR)))
    pdfmetrics.registerFont(TTFont("MicrosoftYaHei-Bold", str(FONT_BOLD)))
    pdfmetrics.registerFont(TTFont("SimSun", str(FONT_SERIF)))
    pdfmetrics.registerFontFamily(
        "MicrosoftYaHei",
        normal="MicrosoftYaHei",
        bold="MicrosoftYaHei-Bold",
        italic="MicrosoftYaHei",
        boldItalic="MicrosoftYaHei-Bold",
    )


def normalize_text(value: str) -> str:
    value = value.replace("\u200b", "").replace("\t", " ").strip()
    value = value.replace("—", "-").replace("–", "-").replace("−", "-")
    value = value.replace("\u2011", "-")
    return re.sub(r"\s+", " ", value)


def convert_math(value: str) -> str:
    value = value.replace(r"\times", "×")
    value = re.sub(r"\^\{([^}]+)\}", lambda match: "^" + match.group(1), value)
    value = value.replace("{", "").replace("}", "")
    return value


def inline_markup(value: str) -> str:
    value = normalize_text(value)
    value = re.sub(r"\$([^$]+)\$", lambda m: convert_math(m.group(1)), value)
    tokens: list[str] = []

    def token(markup: str) -> str:
        marker = f"@@TOKEN{len(tokens)}@@"
        tokens.append(markup)
        return marker

    value = re.sub(r"<br\s*/?>", lambda _: token("<br/>"), value, flags=re.IGNORECASE)
    value = re.sub(
        r"\[([^\]]+)\]\((https?://[^)]+)\)",
        lambda m: token(
            f'<link href="{escape(m.group(2), {"\"": "&quot;"})}" '
            f'color="#1F5A85">{escape(m.group(1))}</link>'
        ),
        value,
    )
    value = re.sub(
        r"<(https?://[^>]+)>",
        lambda m: token(
            f'<link href="{escape(m.group(1), {"\"": "&quot;"})}" '
            f'color="#1F5A85">{escape(m.group(1))}</link>'
        ),
        value,
    )
    value = re.sub(
        r"`([^`]+)`",
        lambda m: token(f'<font name="MicrosoftYaHei">{escape(m.group(1))}</font>'),
        value,
    )
    value = escape(value)
    value = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", value)
    value = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", value)
    for index, markup in enumerate(tokens):
        value = value.replace(f"@@TOKEN{index}@@", markup)
    return value


def make_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=base["Title"],
            fontName="MicrosoftYaHei-Bold",
            fontSize=20,
            leading=27,
            textColor=colors.black,
            alignment=TA_LEFT,
            spaceAfter=5 * mm,
        ),
        "h2": ParagraphStyle(
            "H2",
            parent=base["Heading2"],
            fontName="MicrosoftYaHei-Bold",
            fontSize=14,
            leading=19,
            textColor=colors.black,
            spaceBefore=3.5 * mm,
            spaceAfter=1.8 * mm,
            keepWithNext=True,
        ),
        "h3": ParagraphStyle(
            "H3",
            parent=base["Heading3"],
            fontName="MicrosoftYaHei-Bold",
            fontSize=11.5,
            leading=16,
            textColor=colors.black,
            spaceBefore=2.5 * mm,
            spaceAfter=1.4 * mm,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName="SimSun",
            fontSize=9.4,
            leading=13.2,
            alignment=TA_LEFT,
            textColor=colors.black,
            firstLineIndent=18.8,
            spaceAfter=1.6 * mm,
            splitLongWords=True,
            wordWrap="CJK",
        ),
        "caption": ParagraphStyle(
            "Caption",
            parent=base["BodyText"],
            fontName="MicrosoftYaHei",
            fontSize=8.2,
            leading=11.2,
            alignment=TA_LEFT,
            textColor=colors.black,
            spaceBefore=0.8 * mm,
            spaceAfter=2 * mm,
            wordWrap="CJK",
            splitLongWords=True,
        ),
        "cell": ParagraphStyle(
            "Cell",
            parent=base["BodyText"],
            fontName="MicrosoftYaHei",
            fontSize=8,
            leading=11.2,
            alignment=TA_LEFT,
            textColor=colors.black,
            splitLongWords=True,
            wordWrap="CJK",
        ),
        "cell_center": ParagraphStyle(
            "CellCenter",
            parent=base["BodyText"],
            fontName="MicrosoftYaHei",
            fontSize=8,
            leading=11.2,
            alignment=TA_CENTER,
            textColor=colors.black,
            splitLongWords=True,
            wordWrap="CJK",
        ),
        "cell_header": ParagraphStyle(
            "CellHeader",
            parent=base["BodyText"],
            fontName="MicrosoftYaHei-Bold",
            fontSize=8,
            leading=11.2,
            alignment=TA_CENTER,
            textColor=colors.black,
        ),
        "list": ParagraphStyle(
            "List",
            parent=base["BodyText"],
            fontName="SimSun",
            fontSize=9.2,
            leading=13.2,
            leftIndent=6 * mm,
            firstLineIndent=-5 * mm,
            spaceAfter=1.1 * mm,
            textColor=colors.black,
            wordWrap="CJK",
        ),
        "reference": ParagraphStyle(
            "Reference",
            parent=base["BodyText"],
            fontName="MicrosoftYaHei",
            fontSize=7.8,
            leading=10.8,
            leftIndent=5.5 * mm,
            firstLineIndent=-5.5 * mm,
            spaceAfter=1 * mm,
            textColor=colors.black,
            splitLongWords=True,
            wordWrap="CJK",
        ),
    }


def make_table(rows: list[list[str]], styles: dict[str, ParagraphStyle], doc_width: float) -> Table:
    columns = len(rows[0])
    formatted: list[list[Paragraph]] = []
    for row_index, row in enumerate(rows):
        formatted_row: list[Paragraph] = []
        for column_index, cell in enumerate(row):
            if row_index == 0:
                style = styles["cell_header"]
            elif (columns == 2 and column_index == 0) or (
                columns == 6 and column_index in {1, 3, 4, 5}
            ):
                style = styles["cell_center"]
            else:
                style = styles["cell"]
            formatted_row.append(Paragraph(inline_markup(cell), style))
        formatted.append(formatted_row)

    if columns == 2:
        widths = [18 * mm, doc_width - 18 * mm]
    elif columns == 6:
        ratios = [0.205, 0.16, 0.17, 0.18, 0.135, 0.15]
        widths = [doc_width * ratio for ratio in ratios]
    else:
        widths = [doc_width / columns] * columns

    table = Table(formatted, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.white),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.45, colors.black),
                ("LEFTPADDING", (0, 0), (-1, -1), 3.4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3.4),
                ("TOPPADDING", (0, 0), (-1, -1), 3.2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3.2),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.white]),
            ]
        )
    )
    return table


def make_image(path: Path, doc_width: float) -> Image:
    with PILImage.open(path) as source_image:
        pixel_width, pixel_height = source_image.size
    if path.name == "bp_ffbp_cpu_gpu_timing.png":
        max_width = min(doc_width - 28 * mm, 130 * mm)
        max_height = 60 * mm
    elif path.name == "bp_ffbp_cpu_gpu_fourway.png":
        max_width = doc_width - 20 * mm
        max_height = 96 * mm
    else:
        max_width = doc_width - 24 * mm
        max_height = 96 * mm
    scale = min(max_width / pixel_width, max_height / pixel_height)
    return Image(str(path), width=pixel_width * scale, height=pixel_height * scale, hAlign="CENTER")


def parse_markdown(source: Path, styles: dict[str, ParagraphStyle], doc_width: float) -> list:
    lines = source.read_text(encoding="utf-8").splitlines()
    story: list = []
    paragraph_buffer: list[str] = []
    pending_image: Image | None = None
    in_references = False
    index = 0

    def flush_paragraph() -> None:
        nonlocal paragraph_buffer, pending_image
        if not paragraph_buffer:
            return
        text = normalize_text(" ".join(paragraph_buffer))
        style = styles["caption"] if re.match(r"^\*\*图\s*\d+", text) else styles["body"]
        paragraph = Paragraph(inline_markup(text), style)
        if pending_image is not None:
            story.append(KeepTogether([pending_image, paragraph]))
            pending_image = None
        else:
            story.append(paragraph)
        paragraph_buffer = []

    while index < len(lines):
        raw = lines[index]
        line = normalize_text(raw)

        if not line:
            flush_paragraph()
            index += 1
            continue

        image_match = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)$", line)
        if image_match:
            flush_paragraph()
            if pending_image is not None:
                story.append(pending_image)
            image_path = (source.parent / image_match.group(2)).resolve()
            if not image_path.exists():
                raise FileNotFoundError(f"Missing report image: {image_path}")
            pending_image = make_image(image_path, doc_width)
            index += 1
            continue

        if line.startswith("|"):
            flush_paragraph()
            if pending_image is not None:
                story.append(pending_image)
                pending_image = None
            table_lines: list[str] = []
            while index < len(lines) and normalize_text(lines[index]).startswith("|"):
                table_lines.append(normalize_text(lines[index]))
                index += 1
            rows: list[list[str]] = []
            for table_line in table_lines:
                cells = [cell.strip() for cell in table_line.strip("|").split("|")]
                if all(re.fullmatch(r":?-+:?", cell) for cell in cells):
                    continue
                rows.append(cells)
            story.append(make_table(rows, styles, doc_width))
            story.append(Spacer(1, 2.2 * mm))
            continue

        heading = re.match(r"^(#{1,3})\s+(.+)$", line)
        if heading:
            flush_paragraph()
            if pending_image is not None:
                story.append(pending_image)
                pending_image = None
            level = len(heading.group(1))
            heading_text = heading.group(2)
            if heading_text == "参考文献":
                in_references = True
            style = styles["title"] if level == 1 else styles[f"h{level}"]
            story.append(Paragraph(inline_markup(heading_text), style))
            index += 1
            continue

        numbered = re.match(r"^(\d+)\.\s+(.+)$", line)
        if numbered:
            flush_paragraph()
            if pending_image is not None:
                story.append(pending_image)
                pending_image = None
            style = styles["reference"] if in_references else styles["list"]
            story.append(
                Paragraph(
                    f"{numbered.group(1)}. {inline_markup(numbered.group(2))}",
                    style,
                )
            )
            index += 1
            continue

        paragraph_buffer.append(line)
        index += 1

    flush_paragraph()
    if pending_image is not None:
        story.append(pending_image)
    return story


def draw_page(canvas, document) -> None:
    canvas.saveState()
    page_width, _ = A4
    canvas.setFont("MicrosoftYaHei", 7.3)
    canvas.setFillColor(colors.black)
    canvas.drawRightString(page_width - document.rightMargin, 9 * mm, f"第 {document.page} 页")
    canvas.restoreState()


def main() -> None:
    register_fonts()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        rightMargin=14.5 * mm,
        leftMargin=14.5 * mm,
        topMargin=14 * mm,
        bottomMargin=13.5 * mm,
        title="2026-9-19 组会汇报：BP 与 FFBP 的 CPU/GPU 成像基线",
        author="SAR imaging project",
        subject="CPU/GPU BP and FFBP baseline comparison",
    )
    styles = make_styles()
    story = parse_markdown(SOURCE, styles, document.width)
    document.build(story, onFirstPage=draw_page, onLaterPages=draw_page)
    print(OUTPUT)


if __name__ == "__main__":
    main()
