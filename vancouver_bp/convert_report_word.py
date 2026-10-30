"""Convert the approved Markdown using the user's retained Word typography.

Run with the bundled workspace Python. Original report text and PNG bytes are
kept; the retained DOCX controls page geometry, paragraph roles and footer.
"""
from __future__ import annotations

from copy import deepcopy
from io import BytesIO
import hashlib
import json
from pathlib import Path
import re
from zipfile import ZipFile, ZIP_DEFLATED

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.shared import Pt, Twips, RGBColor
from lxml import etree as ET


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "vancouver_bp/results"
BUILD = RESULTS / "word_build"
SOURCE = RESULTS / "REPORT.md"
REFERENCE = Path(r"D:\Users\Chen\Desktop\10.17 chy.docx")
OUTPUT = RESULTS / "BP与FFBP成像对比实验.docx"
REFERENCE_SHA = "e9b389cab4e5ca2df12f3525738e22eba0fabe50a67673f0c2980bb3138bb629"
NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
WIDTH = 9978


def sha(data):
    return hashlib.sha256(data).hexdigest()


def node(tag, **attrs):
    e = OxmlElement(tag)
    for k, v in attrs.items():
        e.set(qn("w:" + k), str(v))
    return e


def format_run(run, size=10.5, bold=False):
    run.font.name = "宋体"
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = False
    run.font.color.rgb = RGBColor(0, 0, 0)
    rp = run._r.get_or_add_rPr()
    rf = rp.find(qn("w:rFonts"))
    for key in list(rf.attrib):
        if "Theme" in key or "theme" in key:
            del rf.attrib[key]
    for name in ("ascii", "hAnsi", "eastAsia", "cs"):
        rf.set(qn("w:" + name), "宋体")
    color = rp.find(qn("w:color"))
    for key in list(color.attrib):
        if key != qn("w:val"):
            del color.attrib[key]
    # Avoid inheriting the source's character style accents/underlines.
    for name in ("rStyle", "u", "highlight", "shd"):
        for e in list(rp.findall(qn("w:" + name))):
            rp.remove(e)
    return run


def format_paragraph(p, *, size=10.5, align=WD_ALIGN_PARAGRAPH.JUSTIFY,
                     indent=True, before=0, after=5, line=1.25,
                     keep_next=False, keep=True):
    f = p.paragraph_format
    f.alignment = align
    f.first_line_indent = Pt(21 if indent else 0)
    f.left_indent = Pt(0)
    f.right_indent = Pt(0)
    f.space_before = Pt(before)
    f.space_after = Pt(after)
    f.line_spacing = line
    f.keep_together = keep
    f.keep_with_next = keep_next
    f.widow_control = True
    for r in p.runs:
        format_run(r, size)
    return p


def append_text(p, text, size=10.5, bold=False):
    """Create black hyperlinks while retaining their original destinations."""
    pattern = re.compile(r"\[([^\]]+)\]\(<?([^>)]+)>?\)")
    start = 0
    for m in pattern.finditer(text):
        if m.start() > start:
            format_run(p.add_run(text[start:m.start()]), size, bold)
        label, target = m.groups()
        if not re.match(r"^[a-z]+://", target, re.I):
            target = (SOURCE.parent / target).resolve().as_uri()
        h = OxmlElement("w:hyperlink")
        h.set(qn("r:id"), p.part.relate_to(target, RT.HYPERLINK, is_external=True))
        run = p.add_run(label)
        format_run(run, size, bold)
        p._p.remove(run._r)
        h.append(run._r)
        p._p.append(h)
        start = m.end()
    if start < len(text):
        format_run(p.add_run(text[start:]), size, bold)


def blocks(text):
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith("# "):
            yield "title", line[2:]
        elif line.startswith("## "):
            yield "heading", line[3:]
        elif line.startswith("!["):
            m = re.fullmatch(r"!\[(.*)\]\(<?([^>)]+)>?\)", line)
            if not m:
                raise ValueError("Unrecognized figure: " + line)
            yield "image", (m.group(2), m.group(1))
        elif line.startswith("|"):
            table = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                row = [x.strip() for x in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", x) for x in row):
                    table.append(row)
                i += 1
            yield "table", table
            continue
        elif line.startswith("- "):
            yield "reference", line[2:]
        elif re.match(r"^图\d+", line):
            yield "caption", line
        else:
            paragraph = [line]
            while i + 1 < len(lines) and lines[i+1].strip() and not re.match(
                    r"^(#|\||!\[|- |图\d)", lines[i+1].strip()):
                i += 1
                paragraph.append(lines[i].strip())
            yield "p", " ".join(paragraph)
        i += 1


def add_table(doc, data):
    n = len(data[0])
    if n == 2:
        widths = [2400, 7578]
    elif n == 6:
        widths = [900, 1250, 1100, 2100, 2314, 2314]
    elif data[0][0] == "目标":
        widths = [850, 2200, 2300, 4628]
    else:
        widths = [1900, 2478, 2800, 2800]
    assert sum(widths) == WIDTH
    t = doc.add_table(rows=0, cols=n)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    pr = t._tbl.tblPr
    for e in list(pr):
        pr.remove(e)
    pr.append(node("w:tblW", w=WIDTH, type="dxa"))
    pr.append(node("w:jc", val="center"))
    borders = node("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        borders.append(node("w:" + edge, val="single", sz="4", color="000000"))
    pr.append(borders)
    pr.append(node("w:tblLayout", type="fixed"))
    margins = node("w:tblCellMar")
    for edge, value in (("top", 65), ("bottom", 65), ("left", 95), ("right", 95)):
        margins.append(node("w:" + edge, w=value, type="dxa"))
    pr.append(margins)
    for col, width in zip(t.columns, widths):
        col.width = Twips(width)
    for ri, row in enumerate(data):
        cells = t.add_row().cells
        trpr = t.rows[-1]._tr.get_or_add_trPr()
        trpr.append(node("w:cantSplit"))
        if ri == 0:
            trpr.append(node("w:tblHeader"))
        for ci, (cell, text) in enumerate(zip(cells, row)):
            cell.width = Twips(widths[ci])
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            cell._tc.get_or_add_tcPr().append(node("w:shd", val="clear", fill="FFFFFF"))
            p = cell.paragraphs[0]
            left = (n == 2 and ri > 0) or (n == 4 and data[0][0] == "目标" and ci == 3 and ri > 0)
            format_paragraph(p, size=8.5 if ri == 0 else 8,
                             align=WD_ALIGN_PARAGRAPH.LEFT if left else WD_ALIGN_PARAGRAPH.CENTER,
                             indent=False, after=0, line=269/240,
                             keep_next=ri < len(data)-1)
            append_text(p, text, size=8.5 if ri == 0 else 8, bold=ri == 0)
    # The source uses a modest gap after tables.
    p = doc.add_paragraph()
    format_paragraph(p, indent=False, after=3, line=1, keep=False)
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.line_spacing = Pt(3)
    format_run(p.add_run(""), 3)
    return t


def build():
    BUILD.mkdir(parents=True, exist_ok=True)
    if sha(REFERENCE.read_bytes()) != REFERENCE_SHA:
        raise RuntimeError("Reference changed; inspect the new reference before converting")
    if not (BUILD / "artifact.md").exists():
        raise RuntimeError("Missing retained-reference formatting contract")
    source_bytes = SOURCE.read_bytes()
    doc = Document(REFERENCE)
    doc._body.clear_content()
    for rid, rel in list(doc.part.rels.items()):
        if rel.reltype == RT.IMAGE:
            del doc.part.rels[rid]

    image_sources = []
    seen_title = False
    subtitle_done = False
    page_start_titles = {"二 两个强散射点的局部响应", "三 距离向和方位向质量指标", "四 整场与局部的数值对照"}
    for kind, value in blocks(source_bytes.decode("utf-8-sig")):
        if kind == "title":
            p = doc.add_paragraph(style="Title")
            format_paragraph(p, indent=False, align=WD_ALIGN_PARAGRAPH.CENTER,
                             before=12, after=6, keep_next=True)
            append_text(p, value, size=22, bold=True)
            seen_title = True
        elif kind == "heading":
            p = doc.add_paragraph(style="Heading 2")
            format_paragraph(p, indent=False, align=WD_ALIGN_PARAGRAPH.LEFT,
                             before=9, after=5, keep_next=True)
            append_text(p, value, size=12.5, bold=True)
            if value in page_start_titles:
                p.paragraph_format.page_break_before = True
        elif kind == "p":
            p = doc.add_paragraph(style="Normal")
            subtitle = seen_title and not subtitle_done
            format_paragraph(p, indent=not subtitle,
                             align=WD_ALIGN_PARAGRAPH.CENTER if subtitle else WD_ALIGN_PARAGRAPH.JUSTIFY,
                             after=5, keep_next=subtitle)
            append_text(p, value)
            subtitle_done = True
        elif kind == "image":
            path, alt = value
            image = (SOURCE.parent / path).resolve()
            image_sources.append((image, sha(image.read_bytes())))
            p = doc.add_paragraph()
            format_paragraph(p, align=WD_ALIGN_PARAGRAPH.CENTER, indent=False,
                             before=2, after=3, line=1, keep_next=True)
            run = p.add_run()
            inline = run.add_picture(str(image), width=Twips(WIDTH))._inline
            inline.docPr.set("descr", alt)
            inline.docPr.set("title", image.stem)
        elif kind == "caption":
            p = doc.add_paragraph(style="Caption")
            format_paragraph(p, size=9, align=WD_ALIGN_PARAGRAPH.CENTER,
                             indent=False, after=7, line=1.15)
            append_text(p, value, size=9)
        elif kind == "table":
            add_table(doc, value)
        elif kind == "reference":
            p = doc.add_paragraph(style="Normal")
            format_paragraph(p, size=9, align=WD_ALIGN_PARAGRAPH.LEFT,
                             indent=False, after=3, line=1.15)
            p.paragraph_format.left_indent = Twips(397)
            p.paragraph_format.first_line_indent = Twips(-397)
            append_text(p, value, size=9)

    doc.core_properties.title = "BP与FFBP成像对比实验"
    doc.core_properties.subject = "CDdata1片段的BP与FFBP成像及强散射点质量评价"
    doc.core_properties.author = ""
    doc.core_properties.last_modified_by = ""
    settings = doc.settings.element
    update = settings.find(qn("w:updateFields"))
    if update is None:
        update = node("w:updateFields", val="true")
        settings.append(update)
    else:
        update.set(qn("w:val"), "true")

    buf = BytesIO()
    doc.save(buf)
    # Keep all untouched template package parts byte-for-byte. Replace only
    # the report body, its relationships, metadata and field-refresh setting.
    generated = ZipFile(BytesIO(buf.getvalue()))
    template = ZipFile(REFERENCE)
    replace_parts = {
        "word/document.xml", "word/_rels/document.xml.rels",
        "word/settings.xml", "docProps/core.xml", "[Content_Types].xml",
    }
    original_images = {x for x in template.namelist() if x.startswith("word/media/")}
    added_images = {x for x in generated.namelist() if x.startswith("word/media/")}
    final_parts = {}
    for info in template.infolist():
        name = info.filename
        if info.is_dir() or name in original_images:
            continue
        final_parts[name] = generated.read(name) if name in replace_parts else template.read(name)
    for name in added_images:
        final_parts[name] = generated.read(name)
    # The old thumbnail depicted the reference's content, so omit it and its
    # root relationship; leave the remaining relationships untouched.
    final_parts.pop("docProps/thumbnail.wmf", None)
    rels = ET.fromstring(final_parts["_rels/.rels"])
    for rel in list(rels):
        if rel.get("Type", "").endswith("/thumbnail"):
            rels.remove(rel)
    final_parts["_rels/.rels"] = ET.tostring(rels, xml_declaration=True, encoding="UTF-8", standalone=True)
    ct = ET.fromstring(final_parts["[Content_Types].xml"])
    for item in list(ct):
        if item.get("PartName") == "/docProps/thumbnail.wmf":
            ct.remove(item)
    final_parts["[Content_Types].xml"] = ET.tostring(ct, xml_declaration=True, encoding="UTF-8", standalone=True)
    # App metadata from the old content must not advertise its old page/word counts.
    app = ET.fromstring(final_parts["docProps/app.xml"])
    for item in list(app):
        if ET.QName(item).localname in {"Pages", "Words", "Characters", "CharactersWithSpaces", "TotalTime", "Lines", "Paragraphs"}:
            app.remove(item)
    final_parts["docProps/app.xml"] = ET.tostring(app, xml_declaration=True, encoding="UTF-8", standalone=True)
    with ZipFile(OUTPUT, "w", ZIP_DEFLATED) as out:
        for name, data in final_parts.items():
            out.writestr(name, data)

    # Fidelity evidence covers exact source text, images, geometry and the
    # template components deliberately retained in the new report.
    retained = [name for name in template.namelist()
                if not name.endswith("/") and name not in replace_parts
                and name not in original_images
                and name not in {"docProps/thumbnail.wmf", "_rels/.rels", "docProps/app.xml"}]
    changed_retained = [name for name in retained if final_parts[name] != template.read(name)]
    assert not changed_retained, changed_retained
    final_doc = ET.fromstring(final_parts["word/document.xml"])
    orig_doc = ET.fromstring(template.read("word/document.xml"))
    assert ET.tostring(final_doc.find("w:body/w:sectPr", NS)) == ET.tostring(orig_doc.find("w:body/w:sectPr", NS))
    media_hashes = {sha(final_parts[n]) for n in added_images}
    assert all(digest in media_hashes for _, digest in image_sources), "Image data changed"
    assert sha(SOURCE.read_bytes()) == sha(source_bytes), "Markdown changed"
    assert sha(REFERENCE.read_bytes()) == REFERENCE_SHA, "Reference changed"
    evidence = {
        "output": str(OUTPUT), "source_markdown_sha256": sha(source_bytes),
        "reference_sha256": REFERENCE_SHA, "original_images_preserved": True,
        "image_sources": [{"path": str(p), "sha256": h} for p, h in image_sources],
        "preserved_parts": retained, "changed_preserve_only_parts": changed_retained,
        "section_geometry_unchanged": True,
        "reference_inventory": [{"path": i.filename, "size": i.file_size, "sha256": sha(template.read(i.filename))}
                                for i in template.infolist() if not i.is_dir()],
    }
    (BUILD / "conversion-evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "figures": len(image_sources),
                      "preserved_package_parts": len(retained), "source_images_unchanged": True}, ensure_ascii=False))


if __name__ == "__main__":
    build()
