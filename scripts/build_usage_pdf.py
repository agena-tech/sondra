from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, StyleSheet1, getSampleStyleSheet
from reportlab.lib.units import cm, inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    Image as RLImage,
    KeepTogether,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents


ROOT = Path(__file__).resolve().parents[1]
INPUT_MDX = ROOT / "docs" / "usage.mdx"
OUTPUT_PDF = ROOT / "docs" / "usage.pdf"
TEMP_DIR = ROOT / "tmp" / "pdfs"
LOGO_IMAGE = ROOT / "logo_agena.jpg"

FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
FONT_SANS = FONT_DIR / "DejaVuSans.ttf"
FONT_SANS_BOLD = FONT_DIR / "DejaVuSans-Bold.ttf"
FONT_MONO = FONT_DIR / "DejaVuSansMono.ttf"

@dataclass
class Block:
    kind: str
    level: int = 0
    text: str = ""
    index: int = 0


class UsageDocTemplate(BaseDocTemplate):
    def __init__(self, filename: str, **kwargs) -> None:
        super().__init__(filename, **kwargs)
        self._heading_seq = 0

    def beforeDocument(self) -> None:
        self._heading_seq = 0

    def afterFlowable(self, flowable) -> None:  # noqa: ANN001
        style_name = getattr(getattr(flowable, "style", None), "name", "")
        if style_name not in {"GuideHeading1", "GuideHeading2", "GuideHeading3"}:
            return

        level_map = {"GuideHeading1": 0, "GuideHeading2": 1, "GuideHeading3": 2}
        title = flowable.getPlainText()
        key = f"heading-{self._heading_seq}"
        self._heading_seq += 1

        self.canv.bookmarkPage(key)
        self.notify("TOCEntry", (level_map[style_name], title, self.page, key))


def register_fonts() -> None:
    pdfmetrics.registerFont(TTFont("DejaVuSans", str(FONT_SANS)))
    pdfmetrics.registerFont(TTFont("DejaVuSans-Bold", str(FONT_SANS_BOLD)))
    pdfmetrics.registerFont(TTFont("DejaVuSansMono", str(FONT_MONO)))


def build_styles() -> StyleSheet1:
    styles = getSampleStyleSheet()

    styles.add(
        ParagraphStyle(
            name="CoverTitle",
            parent=styles["Title"],
            fontName="DejaVuSans-Bold",
            fontSize=40,
            leading=44,
            alignment=TA_CENTER,
            textColor=colors.white,
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CoverSubtitle",
            parent=styles["Normal"],
            fontName="DejaVuSans",
            fontSize=11,
            leading=15,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#9ef3de"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="CoverEyebrow",
            parent=styles["Normal"],
            fontName="DejaVuSans-Bold",
            fontSize=10,
            leading=13,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#6ef0cf"),
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CoverGuideLabel",
            parent=styles["Normal"],
            fontName="DejaVuSans-Bold",
            fontSize=15,
            leading=18,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#71f0d0"),
            spaceAfter=7,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CoverTagline",
            parent=styles["Normal"],
            fontName="DejaVuSans",
            fontSize=11,
            leading=15,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#d7fff4"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="CoverMeta",
            parent=styles["Normal"],
            fontName="DejaVuSans-Bold",
            fontSize=10,
            leading=13,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#8df5dd"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodyTitle",
            parent=styles["Title"],
            fontName="DejaVuSans-Bold",
            fontSize=23,
            leading=29,
            alignment=TA_LEFT,
            textColor=colors.HexColor("#062d2c"),
            spaceAfter=14,
        )
    )
    styles.add(
        ParagraphStyle(
            name="GuideHeading1",
            parent=styles["Heading1"],
            fontName="DejaVuSans-Bold",
            fontSize=18,
            leading=23,
            textColor=colors.HexColor("#083f3d"),
            spaceBefore=16,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="GuideHeading2",
            parent=styles["Heading2"],
            fontName="DejaVuSans-Bold",
            fontSize=14,
            leading=18,
            textColor=colors.HexColor("#0b5e5a"),
            spaceBefore=12,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="GuideHeading3",
            parent=styles["Heading3"],
            fontName="DejaVuSans-Bold",
            fontSize=11.5,
            leading=15,
            textColor=colors.HexColor("#126f69"),
            spaceBefore=10,
            spaceAfter=5,
        )
    )
    styles.add(
        ParagraphStyle(
            name="GuideBody",
            parent=styles["BodyText"],
            fontName="DejaVuSans",
            fontSize=10.2,
            leading=15,
            textColor=colors.HexColor("#111827"),
            alignment=TA_JUSTIFY,
            spaceAfter=7,
        )
    )
    styles.add(
        ParagraphStyle(
            name="GuideBullet",
            parent=styles["BodyText"],
            fontName="DejaVuSans",
            fontSize=10.2,
            leading=15,
            textColor=colors.HexColor("#111827"),
            leftIndent=16,
            firstLineIndent=0,
            bulletIndent=4,
            spaceAfter=5,
        )
    )
    styles.add(
        ParagraphStyle(
            name="GuideNumbered",
            parent=styles["BodyText"],
            fontName="DejaVuSans",
            fontSize=10.2,
            leading=15,
            textColor=colors.HexColor("#111827"),
            leftIndent=18,
            firstLineIndent=0,
            spaceAfter=5,
        )
    )
    styles.add(
        ParagraphStyle(
            name="GuideQuote",
            parent=styles["BodyText"],
            fontName="DejaVuSans",
            fontSize=10.2,
            leading=15,
            textColor=colors.HexColor("#23413f"),
            leftIndent=18,
            borderPadding=6,
            italic=True,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="TOCHeading",
            parent=styles["Title"],
            fontName="DejaVuSans-Bold",
            fontSize=24,
            leading=28,
            textColor=colors.HexColor("#083f3d"),
            alignment=TA_LEFT,
            spaceAfter=16,
        )
    )
    styles.add(
        ParagraphStyle(
            name="TOCIntro",
            parent=styles["BodyText"],
            fontName="DejaVuSans",
            fontSize=10.5,
            leading=15,
            textColor=colors.HexColor("#324b49"),
            spaceAfter=12,
        )
    )
    styles.add(
        ParagraphStyle(
            name="GuideCode",
            parent=styles["Code"],
            fontName="DejaVuSansMono",
            fontSize=8.6,
            leading=11.2,
            textColor=colors.HexColor("#d7fff4"),
            leftIndent=0,
            rightIndent=0,
            spaceAfter=0,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Logo",
            parent=styles["Code"],
            fontName="DejaVuSansMono",
            fontSize=8.8,
            leading=10.2,
            textColor=colors.HexColor("#6ef0cf"),
            alignment=TA_CENTER,
        )
    )
    return styles


def sanitize_inline(text: str) -> str:
    escaped = html.escape(text.strip())
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(
        r"`([^`]+)`",
        lambda match: f'<font name="DejaVuSansMono">{match.group(1)}</font>',
        escaped,
    )
    return escaped


def starts_new_block(stripped: str) -> bool:
    return bool(
        not stripped
        or stripped == "---"
        or stripped.startswith("```")
        or re.match(r"^#{1,6}\s+", stripped)
        or stripped.startswith("- ")
        or stripped.startswith("> ")
        or re.match(r"^\d+\.\s+", stripped)
    )


def load_body_source() -> str:
    raw = INPUT_MDX.read_text(encoding="utf-8")
    if raw.startswith("---"):
        _, _, remainder = raw.partition("\n---")
        raw = remainder.lstrip("\n")
    raw = re.sub(
        r"<p\s+align=\"center\">\s*<b>Agena Memory Systems © 2026</b>\s*</p>",
        "",
        raw,
        flags=re.IGNORECASE | re.DOTALL,
    )
    lines = raw.splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.startswith("# Sondra Usage Guide"))
    except StopIteration:
        start = 0
    return "\n".join(lines[start:]).strip()


def parse_blocks(text: str) -> list[Block]:
    lines = text.splitlines()
    blocks: list[Block] = []
    i = 0
    block_index = 0

    while i < len(lines):
        raw = lines[i]
        stripped = raw.strip()

        if not stripped:
            i += 1
            continue

        if stripped == "---":
            blocks.append(Block(kind="hr", index=block_index))
            block_index += 1
            i += 1
            continue

        if stripped.startswith("```"):
            code_lines: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i].rstrip("\n"))
                i += 1
            i += 1
            blocks.append(Block(kind="code", text="\n".join(code_lines).rstrip(), index=block_index))
            block_index += 1
            continue

        heading_match = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if heading_match:
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()
            blocks.append(Block(kind="heading", level=level, text=title, index=block_index))
            block_index += 1
            i += 1
            continue

        if stripped.startswith("- "):
            item_lines = [stripped[2:].strip()]
            i += 1
            while i < len(lines):
                nxt = lines[i].strip()
                if starts_new_block(nxt):
                    break
                item_lines.append(nxt)
                i += 1
            blocks.append(Block(kind="bullet", text=" ".join(item_lines).strip(), index=block_index))
            block_index += 1
            continue

        number_match = re.match(r"^(\d+)\.\s+(.*)$", stripped)
        if number_match:
            item_lines = [number_match.group(2).strip()]
            label = number_match.group(1)
            i += 1
            while i < len(lines):
                nxt = lines[i].strip()
                if starts_new_block(nxt):
                    break
                item_lines.append(nxt)
                i += 1
            blocks.append(
                Block(kind="number", text=f"{label}. {' '.join(item_lines).strip()}", index=block_index)
            )
            block_index += 1
            continue

        if stripped.startswith("> "):
            quote_lines = [stripped[2:].strip()]
            i += 1
            while i < len(lines):
                nxt = lines[i].strip()
                if starts_new_block(nxt):
                    break
                quote_lines.append(nxt)
                i += 1
            blocks.append(Block(kind="quote", text=" ".join(quote_lines).strip(), index=block_index))
            block_index += 1
            continue

        para_lines = [stripped]
        i += 1
        while i < len(lines):
            nxt = lines[i].strip()
            if starts_new_block(nxt):
                break
            para_lines.append(nxt)
            i += 1
        blocks.append(Block(kind="paragraph", text=" ".join(para_lines).strip(), index=block_index))
        block_index += 1

    return blocks


def cover_page(canvas, _doc) -> None:  # noqa: ANN001
    width, height = A4
    canvas.saveState()
    canvas.setFillColor(colors.HexColor("#041a19"))
    canvas.rect(0, 0, width, height, fill=1, stroke=0)

    canvas.setStrokeColor(colors.HexColor("#00d9b2"))
    canvas.setLineWidth(1.2)
    canvas.rect(1.05 * cm, 1.05 * cm, width - 2.1 * cm, height - 2.1 * cm, fill=0, stroke=1)
    canvas.setLineWidth(2.4)
    canvas.line(1.8 * cm, height - 1.8 * cm, width - 1.8 * cm, height - 1.8 * cm)
    canvas.line(1.8 * cm, 1.8 * cm, width - 1.8 * cm, 1.8 * cm)

    canvas.setLineWidth(1)
    canvas.setStrokeColor(colors.HexColor("#0f5f58"))
    canvas.line(2.2 * cm, height - 3.0 * cm, 5.4 * cm, height - 3.0 * cm)
    canvas.line(width - 5.4 * cm, 3.0 * cm, width - 2.2 * cm, 3.0 * cm)
    canvas.line(width - 5.4 * cm, 2.65 * cm, width - 2.2 * cm, 2.65 * cm)
    canvas.line(width - 5.4 * cm, 2.3 * cm, width - 2.2 * cm, 2.3 * cm)

    canvas.restoreState()


def body_page(canvas, doc) -> None:  # noqa: ANN001
    width, height = A4
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#19c7a0"))
    canvas.setLineWidth(1)
    canvas.line(1.5 * cm, height - 1.3 * cm, width - 1.5 * cm, height - 1.3 * cm)
    canvas.line(1.5 * cm, 1.4 * cm, width - 1.5 * cm, 1.4 * cm)

    canvas.setFont("DejaVuSans-Bold", 9)
    canvas.setFillColor(colors.HexColor("#0a4d49"))
    canvas.drawString(1.7 * cm, height - 1.0 * cm, "SONDRA USER GUIDE")

    canvas.setFont("DejaVuSans", 8.5)
    canvas.setFillColor(colors.HexColor("#3a5a57"))
    canvas.drawRightString(width - 1.7 * cm, height - 1.0 * cm, "Agena Memory Systems")
    canvas.drawCentredString(width / 2, 0.95 * cm, str(doc.page))
    canvas.restoreState()


def code_block(code_text: str, styles: StyleSheet1):
    rendered = Preformatted(code_text or "", styles["GuideCode"])
    wrapper = Table([[rendered]], colWidths=[16.2 * cm])
    wrapper.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#082b29")),
                ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#14d7ae")),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    return wrapper


def cover_logo_card() -> Table:
    logo = RLImage(str(LOGO_IMAGE))
    logo.drawHeight = 1.58 * inch
    logo.drawWidth = 1.58 * inch
    logo.hAlign = "CENTER"

    card = Table([[logo]], colWidths=[2.18 * inch], rowHeights=[2.18 * inch])
    card.hAlign = "CENTER"
    card.setStyle(
        TableStyle(
            [
                ("INNERPADDING", (0, 0), (-1, -1), 0),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                ("TOPPADDING", (0, 0), (-1, -1), 12),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ]
        )
    )
    return card


def build_story(styles: StyleSheet1) -> list:
    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle(
            name="TOCLevel1",
            fontName="DejaVuSans-Bold",
            fontSize=10.5,
            leading=14,
            leftIndent=8,
            firstLineIndent=-2,
            textColor=colors.HexColor("#083f3d"),
        ),
        ParagraphStyle(
            name="TOCLevel2",
            fontName="DejaVuSans",
            fontSize=9.5,
            leading=13,
            leftIndent=22,
            firstLineIndent=-2,
            textColor=colors.HexColor("#23413f"),
        ),
        ParagraphStyle(
            name="TOCLevel3",
            fontName="DejaVuSans",
            fontSize=8.8,
            leading=12,
            leftIndent=34,
            firstLineIndent=-2,
            textColor=colors.HexColor("#476463"),
        ),
    ]

    story: list = []
    story.append(Spacer(1, 1.25 * inch))
    story.append(Paragraph("FOR ALL YOU NEEDS", styles["CoverEyebrow"]))
    story.append(Paragraph("SONDRA", styles["CoverTitle"]))
    story.append(Paragraph("USER GUIDE", styles["CoverGuideLabel"]))
    story.append(
        HRFlowable(
            width="34%",
            thickness=1.2,
            lineCap="round",
            color=colors.HexColor("#2fe6c0"),
            spaceBefore=4,
            spaceAfter=10,
            hAlign="CENTER",
        )
    )
    story.append(Paragraph("Prepared by Anezatra", styles["CoverSubtitle"]))
    story.append(Spacer(1, 0.78 * inch))
    cover_group: list = []
    if LOGO_IMAGE.exists():
        cover_group.append(cover_logo_card())
        cover_group.append(Spacer(1, 2.28 * inch))
    cover_group.append(
        Paragraph("An open-source AI agent for all-purpose tasks", styles["CoverTagline"])
    )
    cover_group.append(Spacer(1, 0.14 * inch))
    cover_group.append(Paragraph("2026 Edition", styles["CoverMeta"]))
    cover_group.append(Spacer(1, 0.08 * inch))
    cover_group.append(Paragraph("Copyright (C) 2026, Agena Memory Systems", styles["CoverSubtitle"]))
    story.append(KeepTogether(cover_group))
    story.append(NextPageTemplate("Body"))
    story.append(PageBreak())

    story.append(Paragraph("TABLE OF CONTENTS", styles["TOCHeading"]))
    story.append(
        Paragraph(
            "English contents page for the full guide, generated from the UTF-8 usage source.",
            styles["TOCIntro"],
        )
    )
    story.append(toc)
    story.append(PageBreak())

    blocks = parse_blocks(load_body_source())
    heading_styles = {1: "GuideHeading1", 2: "GuideHeading2", 3: "GuideHeading3"}
    first_h1 = True

    for block in blocks:
        if block.kind == "heading":
            level = min(3, max(1, block.level))
            if level == 1 and not first_h1:
                story.append(Spacer(1, 0.08 * inch))
            first_h1 = False
            story.append(Paragraph(sanitize_inline(block.text), styles[heading_styles[level]]))
        elif block.kind == "paragraph":
            story.append(Paragraph(sanitize_inline(block.text), styles["GuideBody"]))
        elif block.kind == "bullet":
            story.append(
                Paragraph(
                    sanitize_inline(block.text),
                    styles["GuideBullet"],
                    bulletText="\u2022",
                )
            )
        elif block.kind == "number":
            label, _, content = block.text.partition(" ")
            story.append(
                Paragraph(
                    sanitize_inline(content),
                    styles["GuideNumbered"],
                    bulletText=label,
                )
            )
        elif block.kind == "quote":
            story.append(Paragraph(sanitize_inline(block.text), styles["GuideQuote"]))
        elif block.kind == "code":
            story.append(code_block(block.text, styles))
            story.append(Spacer(1, 0.12 * inch))
        elif block.kind == "hr":
            story.append(
                HRFlowable(
                    width="100%",
                    thickness=0.8,
                    lineCap="round",
                    color=colors.HexColor("#8accc1"),
                    spaceBefore=8,
                    spaceAfter=8,
                )
            )

    return story


def build_pdf() -> None:
    register_fonts()
    styles = build_styles()
    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    doc = UsageDocTemplate(
        str(OUTPUT_PDF),
        pagesize=A4,
        leftMargin=1.65 * cm,
        rightMargin=1.65 * cm,
        topMargin=1.7 * cm,
        bottomMargin=1.8 * cm,
        title="SONDRA USER GUIDE",
        author="Anezatra",
    )

    cover_frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="cover")
    body_frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="body")
    doc.addPageTemplates(
        [
            PageTemplate(id="Cover", frames=[cover_frame], onPage=cover_page),
            PageTemplate(id="Body", frames=[body_frame], onPage=body_page),
        ]
    )

    story = build_story(styles)
    doc.multiBuild(story, maxPasses=20)


if __name__ == "__main__":
    build_pdf()
