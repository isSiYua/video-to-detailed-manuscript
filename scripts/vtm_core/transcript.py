from __future__ import annotations

import json
import hashlib
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .llm import OpenAICompatibleClient, parse_json_object
from .models import Frame, Paragraph, Segment
from .utils import atomic_json, load_json

FILLER_PREFIX = re.compile(
    r"^(?:(?:嗯+|呃+|啊+|那个|就是说|然后呢|怎么说呢|其实吧)[，、,。\s]*)+"
)
EXACT_ANCHOR_RE = re.compile(
    r"https?://\S+|(?:BV[0-9A-Za-z]{10})|(?:\d+(?:\.\d+)?\s*(?:%|GB|MB|KB|元|分钟|小时|天|条|个|次))|"
    r"(?:《[^》\n]{2,100}》)|(?:【[^】\n]{2,100}】)|(?:--[A-Za-z0-9][A-Za-z0-9_-]*)|"
    r"(?:/[A-Za-z0-9._~/-]{2,})|(?:[A-Za-z][A-Za-z0-9_.+/#:-]{1,})",
    flags=re.I,
)
NAMED_PERSON_RE = re.compile(
    r"对话\s*([一-鿿]{2,8})(?=这一期|这期|，|。|、|；|$)"
)
IGNORED_EXACT_ANCHORS = {"ok", "okay", "um", "uh", "yeah"}
KNOWN_TECHNICAL_ANCHORS = {
    "ai",
    "agent",
    "ar",
    "anthropic",
    "api",
    "asr",
    "automation",
    "bilibili",
    "build",
    "chatgpt",
    "claude",
    "codex",
    "context",
    "crossing",
    "deepseek",
    "fde",
    "funasr",
    "gemini",
    "github",
    "harness",
    "llm",
    "ml",
    "mcp",
    "memory",
    "obsidian",
    "ocr",
    "openai",
    "pe",
    "pm",
    "prompt",
    "python",
    "qwen",
    "token",
    "ui",
    "ux",
    "vr",
    "workflow",
    "youtube",
}
KNOWN_ASR_ANCHOR_ALIASES = {
    "goodex": "codex",
    "couldx": "codex",
    "couldex": "codex",
    "credex": "codex",
    "oodex": "codex",
}
UNSUPPORTED_EDITOR_COMMENTARY = (
    "这说明",
    "价值在于",
    "适合用来",
    "形成互补",
    "由此可见",
)

SYSTEM_PROMPT = """你是长视频内容编辑，不是摘要器，也不是逐字照抄器。把连续口语字幕重写成接近创作者成稿的详细中文笔记。

保留所有能帮助复现或理解内容的信息：观点、原因、解释、例子、步骤、命令、代码、网址、数据、限定条件、例外、比较、提醒、节目名、单集名、产品名、人名和结论。名称和具体标题的优先级高于泛化概括；不得把“对话某人”“某一期”“某个命令”改写成没有名称的类别。允许合并相邻重复表达、压缩啰嗦句式、修正常见 ASR 同音错字；只删除口癖、无意义起句、广告求赞和真正重复。不得把多个具体细节压成空泛概括，不得添加来源中没有的事实。

不要对讲述者的话再做一层编辑者总结。除非讲述者明确表达，否则不要添加“这说明……”“价值在于……”“适合用来……”“形成互补……”“可见……”等评价或推论。只重排和精炼讲述者实际表达的内容。

按主题组织小节，优先使用视频中的真实工作流程作为结构，例如“前置设置”、“生成检索关键词”、“查找并核验文献”，不要只写“第一部分”这类空标题。单个自然段不得超过 700 个字符；话题、操作阶段或目标改变时必须分段。

操作型视频必须保留：入口路径、按钮/选项名称、允许项、输入内容的目的、可修改参数、等待时间、成本/额度、输出字段、检查方法和失败条件。当讲述者展示但没有逐字念出长提示词、代码、表格或公式时，正文只保留其口述的用途和要求，不得猜测屏幕原文；屏幕原文由后续画面证据阶段补全。

你不需要逐一复制 source id，也不要判断或输出被删除的 id。每个段落只填写 `start_source_id`，表示该段落从哪条字幕开始。第一段的 `start_source_id` 必须等于输入第一条字幕的 id；后续段落的起始 id 必须来自输入并严格按时间递增。程序会自动把一个起始 id 到下一个起始 id 之前的所有字幕归入该段，保证没有遗漏。口癖、过渡语、求赞或真正重复的文字可以不写进 text，但对应时间范围内的观点、操作和细节不能丢失。heading 只在一个新小节的第一个段落填写，其余为 null。正文使用完整自然段；确实是步骤、清单或配置项时可在 text 中使用 Markdown 列表。

严格返回 JSON：{"paragraphs":[{"start_source_id":"s000001","text":"...","heading":"小节标题或 null"}]}。不要输出 `source_ids`、`removed_filler_ids` 或 JSON 之外的内容。"""

BATCH_EDIT_MAX_ATTEMPTS = 3
FULL_MANUSCRIPT_REPAIR_ATTEMPTS = 2
REFINEMENT_MIN_SOURCE_CHARS = 480

FAITHFUL_SYSTEM_PROMPT = """你是视频字幕校订员。当前阶段只负责忠实还原信息，不负责总结，也不追求华丽文风。

逐条理解输入的连续字幕，修正明显的同音 ASR 错字，并把相邻句子合成可读段落。必须保留每一个有意义的论断、原因、解释、例子、步骤、命令、代码、网址、数字、条件、例外、比较、提醒、名称、标题和结论。允许删掉口癖、假开头、求赞和真正重复，但不能删掉其后面的有效动作或限定条件。不得补充来源中没有的背景、价值判断或推论。

previous_context 和 subsequent_context 只帮助理解当前批次的指代、术语和跨批衔接；不要把上下文中的信息重复写进当前批次。每个段落只填写 start_source_id；第一段必须从当前批次第一条字幕开始，后续起点来自当前批次并严格递增。程序会确定性分配起点之间的全部 source id。

严格返回 JSON：{\"paragraphs\":[{\"start_source_id\":\"s000001\",\"text\":\"...\",\"heading\":null}]}。不要输出其他字段或解释。"""

REFINEMENT_SYSTEM_PROMPT = """你是视频文字稿编辑。输入包含原始字幕和一份已经通过信息保留检查的忠实初稿。

你的任务是把忠实初稿编辑成详细、自然、结构清楚的中文笔记：消除口语拖沓，合并真正重复，修正术语，按真实话题或操作阶段分段并设置具体标题。不得丢失初稿或原字幕中的任何有效信息，不得把具体名称、标题、数字、命令、条件和操作细节泛化，不得增加讲述者没有表达的背景、用途、价值判断或结论。

previous_context 和 subsequent_context 只用于理解指代与术语，不得复制为当前批次内容。每个段落只返回当前批次中的 start_source_id；第一段必须从当前批次第一条字幕开始，后续起点严格递增。

严格返回 JSON：{\"paragraphs\":[{\"start_source_id\":\"s000001\",\"text\":\"...\",\"heading\":\"具体小节标题或 null\"}]}。不要输出其他字段或解释。"""

ASR_PROOFREAD_SYSTEM_PROMPT = """你是谨慎的视频 ASR 终校员。只查找最终文字稿中仍然明显不通顺、由同音识别或字母倒置造成的短片段，例如错误产品名、倒置的文件格式、无法成立的数量短语或前后不通的词组。

只提交你能根据相邻原始字幕和最终文稿高置信度修正的项目。original 必须是最终文稿中唯一存在的精确连续子串；replacement 必须是最小、保守、自然的替换。不得改写整句，不得润色风格，不得删除有效细节，不得添加原始字幕没有支持的新名称、数字、网址、命令、结论或背景。无法确认具体数值时，用不带新数值的保守表达；无法高置信度确认时不要提交。

严格返回 JSON：{\"corrections\":[{\"original\":\"精确原片段\",\"replacement\":\"保守替换\"}]}。没有可靠修正时返回 {\"corrections\":[]}。"""


def conservative_clean(text: str) -> str:
    cleaned = FILLER_PREFIX.sub("", text.strip())
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"([。！？])\1+", r"\1", cleaned)
    return cleaned.strip()


def chunk_segments(segments: list[Segment], max_chars: int = 3600) -> list[list[Segment]]:
    chunks: list[list[Segment]] = []
    current: list[Segment] = []
    size = 0
    for segment in segments:
        cost = len(segment.text) + len(segment.id) + 30
        if current and size + cost > max_chars:
            chunks.append(current)
            current, size = [], 0
        current.append(segment)
        size += cost
    if current:
        chunks.append(current)
    return chunks


def _segments_payload(segments: Iterable[Segment]) -> str:
    return "\n".join(
        f"[{segment.id} {segment.start:.2f}-{segment.end:.2f}] {segment.text}"
        for segment in segments
    )


def _chunk_context(
    all_segments: list[Segment], chunk: list[Segment], *, window: int = 3
) -> tuple[str, str]:
    """Return read-only neighbouring context, following VideoLingo's context-window pattern."""
    if not chunk:
        return "", ""
    positions = {segment.id: index for index, segment in enumerate(all_segments)}
    start = positions[chunk[0].id]
    stop = positions[chunk[-1].id] + 1
    previous = all_segments[max(0, start - window) : start]
    subsequent = all_segments[stop : stop + window]
    return _segments_payload(previous), _segments_payload(subsequent)


def _draft_payload(paragraphs: list[Paragraph]) -> str:
    return json.dumps(
        {
            "paragraphs": [
                {
                    "start_source_id": item.source_ids[0],
                    "text": item.text,
                    "heading": item.heading,
                }
                for item in paragraphs
            ]
        },
        ensure_ascii=False,
    )


def _source_char_count(segments: Iterable[Segment]) -> int:
    return sum(len(conservative_clean(segment.text)) for segment in segments)


def _checkpoint_signature(segments: list[Segment], context: str) -> str:
    payload = {
        "schema": "faithful-refine-compose-v1",
        "context": context,
        "segments": [segment.to_dict() for segment in segments],
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_checkpoint_chunks(
    path: Path | None, segments: list[Segment], chunks: list[list[Segment]], context: str
) -> dict[int, list[Paragraph]]:
    if path is None:
        return {}
    payload = load_json(path)
    if not isinstance(payload, dict):
        return {}
    if payload.get("signature") != _checkpoint_signature(segments, context):
        return {}
    completed = payload.get("completed")
    if not isinstance(completed, dict):
        return {}
    result: dict[int, list[Paragraph]] = {}
    for raw_index, draft in completed.items():
        try:
            index = int(raw_index)
            if not 1 <= index <= len(chunks) or not isinstance(draft, dict):
                continue
            parsed, _ = _paragraphs_from_response(draft, chunks[index - 1])
            result[index] = parsed
        except (TypeError, ValueError):
            continue
    return result


def _save_checkpoint_chunks(
    path: Path | None,
    segments: list[Segment],
    context: str,
    completed: dict[int, list[Paragraph]],
) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(
        path,
        {
            "version": 1,
            "signature": _checkpoint_signature(segments, context),
            "completed": {
                str(index): json.loads(_draft_payload(items))
                for index, items in sorted(completed.items())
            },
        },
    )


def _normalized_anchor(value: str) -> str:
    normalized = re.sub(r"[\s`*_\-]", "", value).lower().rstrip(".,，。")
    return KNOWN_ASR_ANCHOR_ALIASES.get(normalized, normalized)


def exact_anchors(segments: Iterable[Segment]) -> list[str]:
    seen: set[str] = set()
    anchors: list[str] = []
    for segment in segments:
        for match in EXACT_ANCHOR_RE.finditer(segment.text):
            value = match.group(0).strip()
            # Plain English speech/UI verbs such as "open", "save", and
            # "click" are not immutable identifiers. Requiring the edited
            # Chinese manuscript to repeat them verbatim creates false
            # failures. Keep structured tokens, acronyms, mixed-case brands,
            # and a small domain allowlist; the prompt still preserves the
            # meaning of all ordinary words.
            if re.fullmatch(r"[A-Za-z]+", value):
                lower = value.lower()
                # Two-letter uppercase fragments are common FunASR noise
                # (for example "SR").  Only enforce such short tokens when
                # they are in the domain allowlist; otherwise an accidental
                # fragment can reject an otherwise faithful manuscript.
                is_acronym = value.isupper() and len(value) >= 3
                is_mixed_case = any(ch.islower() for ch in value) and any(
                    ch.isupper() for ch in value
                )
                if lower not in KNOWN_TECHNICAL_ANCHORS and not is_acronym and not is_mixed_case:
                    continue
            normalized = _normalized_anchor(value)
            if (
                normalized
                and normalized not in IGNORED_EXACT_ANCHORS
                and normalized not in seen
            ):
                seen.add(normalized)
                anchors.append(value)
        for match in NAMED_PERSON_RE.finditer(segment.text):
            value = match.group(1).strip()
            normalized = _normalized_anchor(value)
            if normalized and normalized not in seen:
                seen.add(normalized)
                anchors.append(value)
    return anchors


VISUAL_EVIDENCE_PROMPT = """根据画面 OCR/视觉描述，为对应口述提取可核验的画面细节，并判断图片是否不可替代。
只使用画面中确实可见的信息；不要推断、评价、扩展背景知识或重复口述已有内容。
先判断画面是否直接支撑当前口述。只有画面展示了讲述者正在操作、列举、核验或强调的内容时 relevance=high。普通过渡画面、无关页面细节或只是时间上接近时 relevance=low，visual_note=null。

同时比较 spoken_paragraph 与画面：information_gain=none 表示画面只是重复正文、字幕、标题或装饰；partial 表示增加少量可复制细节；substantial 表示视觉结构、原图或密集内容无法由正文替代。程序会直接丢弃 relevance=low、decorative 或 information_gain=none 的画面。

分类与输出规则：
- text/list：完整转成普通文字或 Markdown 列表；只有画面简单、所有可读信息均已转录且可证明无损时，keep_image=false。
- table：完整转成 Markdown 表格；结构和数值都可靠时 keep_image=false。
- code：保留语言、缩进和符号，输出 fenced code block；识别可靠时 keep_image=false。
- formula：输出可复制的 LaTeX，行内用 $...$、独立公式用 $$...$$；识别可靠时 keep_image=false。
- diagram/chart/process/ui/paper_figure/comparison：只有空间关系、走势、操作状态或原始图形确实承载正文没有的内容时才使用这些分类并 keep_image=true；visual_note 只写与口述直接相关的可见细节，不列举讲述者未关注的通用界面按钮。单个箭头、两个标签、简单图标加标题不算不可替代的流程图或架构图；完整转成文字后归为 text/comparison，information_gain=none 或 partial。
- decorative：人物卡通、库存插画、头像、水印、背景装饰、重复字幕/标题、只用于美观的图标，或“简单标题 + 装饰插画”且正文已经表达全部信息。必须 relevance=low、information_gain=none、visual_note=null、keep_image=false。
- other：不确定时 keep_image=true。

必须额外判断两项：
- completeness=complete 仅表示 visual_note 已覆盖画面中全部有用、可读信息；只提取标题、关键词、部分条目或摘要时必须是 partial。
- information_density=high 表示长提示词、长文章、多张卡片、多列列表、密集界面等信息丰富画面。此类画面即使提取了重点也保留原图。

OCR 噪声较大、内容被截断、OCR 与视觉描述不一致、代码缩进不明、公式符号不确定、表格关系不清时，confidence=low 或 completeness=partial，且 keep_image=true，不能强行转写。
只有 confidence=high、completeness=complete 且底层 OCR 质量足以交叉验证时，visual_note 才可能进入最终文稿。partial/unknown 或 medium/low 只用于决定保留原图，不得输出为可复制文字；不要为了填充 visual_note 而猜测不可读内容。
输入是带 item_id 的数组。严格返回 JSON：{"items":[{"item_id":"v001","relevance":"high|low","content_kind":"text|list|table|code|formula|diagram|chart|process|ui|paper_figure|comparison|decorative|other","information_gain":"none|partial|substantial","visual_note":"Markdown 或 null","keep_image":true,"confidence":"high|medium|low","completeness":"complete|partial|unknown","information_density":"low|medium|high"}]}。每个输入 item_id 恰好返回一次。"""


def enrich_with_visual_evidence(
    paragraphs: list[Paragraph],
    frames: list[Frame],
    client: OpenAICompatibleClient | None,
) -> list[str]:
    if client is None or not frames:
        return []
    warnings: list[str] = []
    selected: list[tuple[str, int, int, Paragraph, Frame]] = []
    # Align every useful candidate to its strongest paragraph.  The previous
    # paragraph-first loop selected only one frame and silently discarded the
    # remaining slides in a dense tutorial section.
    for index, frame in enumerate(frames):
        if len((frame.ocr_text + frame.vision_description).strip()) < 12:
            continue
        if not frame.vision_description.strip() and frame.ocr_confidence < 50:
            continue
        matches: list[tuple[int, Paragraph, int, bool, float]] = []
        frame_ids = set(frame.source_ids)
        for paragraph_index, paragraph in enumerate(paragraphs):
            start = paragraph.start or 0.0
            end = paragraph.end if paragraph.end is not None else start
            overlap = len(frame_ids & set(paragraph.source_ids))
            temporal = start - 2 <= frame.timestamp <= end + 2
            if not overlap and not temporal:
                continue
            midpoint = (start + end) / 2
            matches.append(
                (paragraph_index, paragraph, overlap, temporal, abs(frame.timestamp - midpoint))
            )
        if not matches:
            continue
        paragraph_index, paragraph, _overlap, _temporal, _distance = max(
            matches,
            key=lambda item: (item[2], item[3], -item[4]),
        )
        item_id = f"v{len(selected) + 1:03d}"
        selected.append((item_id, paragraph_index, index, paragraph, frame))

    if not selected:
        return warnings

    by_id: dict[str, dict[str, Any]] = {}
    failed_ids: set[str] = set()
    batch_size = 12
    for offset in range(0, len(selected), batch_size):
        batch = selected[offset : offset + batch_size]
        request_items = [
            {
                "item_id": item_id,
                "timestamp": round(frame.timestamp, 2),
                "spoken_paragraph": paragraph.text[:1400],
                "ocr": frame.ocr_text[:1800],
                "vision_description": frame.vision_description[:1400],
            }
            for item_id, _paragraph_index, _index, paragraph, frame in batch
        ]
        try:
            raw = client.chat(
                [
                    {"role": "system", "content": VISUAL_EVIDENCE_PROMPT},
                    {
                        "role": "user",
                        "content": json.dumps({"items": request_items}, ensure_ascii=False),
                    },
                ],
                temperature=0.0,
                max_tokens=min(8000, 500 + len(batch) * 600),
                json_mode=True,
            )
            response = parse_json_object(raw)
            response_items = response.get("items")
            if isinstance(response_items, list):
                by_id.update(
                    {
                        str(item.get("item_id") or ""): item
                        for item in response_items
                        if isinstance(item, dict)
                    }
                )
            elif len(batch) == 1:
                by_id[batch[0][0]] = response
            else:
                raise ValueError("visual evidence batch response omitted items")
        except Exception as exc:
            failed_ids.update(item[0] for item in batch)
            warnings.append(
                f"visual evidence batch {offset // batch_size + 1} {type(exc).__name__}"
            )

    for item_id, paragraph_index, _index, paragraph, frame in selected:
        try:
            if item_id in failed_ids:
                raise ValueError(f"visual evidence classification unavailable for {item_id}")
            payload = by_id.get(item_id)
            if not isinstance(payload, dict):
                raise ValueError(f"visual evidence result missing {item_id}")
            relevance = str(payload.get("relevance") or "low").strip().lower()
            kind = str(payload.get("content_kind") or "other").strip().lower()
            allowed_kinds = {
                "text", "list", "table", "code", "formula", "diagram", "chart",
                "process", "ui", "paper_figure", "comparison", "decorative", "other",
            }
            kind = kind if kind in allowed_kinds else "other"
            confidence = str(payload.get("confidence") or "low").strip().lower()
            confidence = confidence if confidence in {"high", "medium", "low"} else "low"
            completeness = str(payload.get("completeness") or "unknown").strip().lower()
            completeness = completeness if completeness in {"complete", "partial", "unknown"} else "unknown"
            density = str(payload.get("information_density") or "unknown").strip().lower()
            density = density if density in {"low", "medium", "high"} else "unknown"
            information_gain = str(payload.get("information_gain") or "unknown").strip().lower()
            information_gain = (
                information_gain
                if information_gain in {"none", "partial", "substantial"}
                else "unknown"
            )
            note = payload.get("visual_note")
            frame.paragraph_index = paragraph_index
            frame.content_kind = kind
            frame.evidence_confidence = confidence
            frame.evidence_completeness = completeness
            frame.information_density = density
            frame.information_gain = information_gain
            if relevance != "high" or kind == "decorative" or information_gain == "none":
                frame.keep_image = False
                frame.extracted_markdown = ""
                continue
            frame.keep_image = True
            publishable_visual_text = (
                confidence == "high"
                and completeness == "complete"
                and (
                    frame.ocr_confidence >= 50
                    or len(frame.vision_description.strip()) >= 20
                )
            )
            # A short, fully verified text/list screen can be redundant with
            # the spoken paragraph.  In that case the classifier correctly
            # returns visual_note=null because there is no screen-only fact;
            # still remove the image instead of keeping a decorative duplicate.
            redundant_complete_simple_text = (
                publishable_visual_text
                and kind in {"text", "list"}
                and density in {"low", "medium"}
                and len(re.sub(r"\s+", "", frame.ocr_text)) <= 300
            )
            if redundant_complete_simple_text:
                frame.keep_image = False
            short_transcribed_text = (
                confidence == "high"
                and kind in {"text", "list"}
                and density in {"low", "medium"}
                and note
                and 8 <= len(re.sub(r"\s+", "", str(note))) <= 300
                and (
                    frame.ocr_confidence >= 50
                    or len(frame.vision_description.strip()) >= 8
                )
            )
            if (publishable_visual_text or short_transcribed_text) and note and len(str(note).strip()) >= 8:
                rendered = str(note).strip()
                if re.search(r"(?:这说明|价值在于|适合用来|形成互补|由此可见)", rendered):
                    raise ValueError("visual evidence contains unsupported editorial commentary")
                text_replaceable = kind in {"text", "list", "table", "code", "formula"}
                keep_image = (
                    not text_replaceable
                    or confidence != "high"
                    or completeness != "complete"
                    or density == "high"
                )
                # Short pure-text/list slides are replaceable once the visual
                # model has transcribed their useful content.  A cautious
                # "partial" completeness label alone should not leave a
                # decorative screenshot behind when the screen is low/medium
                # density and the extracted note is short.  Dense prompts and
                # long slides still retain the original image.
                if (
                    kind in {"text", "list"}
                    and confidence == "high"
                    and density in {"low", "medium"}
                    and len(re.sub(r"\s+", "", rendered)) <= 300
                ):
                    keep_image = False
                if kind == "code" and "```" not in rendered:
                    keep_image = True
                    warnings.append(f"paragraph {paragraph_index + 1}: code OCR lacked fenced block")
                if kind == "formula" and "$" not in rendered:
                    keep_image = True
                    warnings.append(f"paragraph {paragraph_index + 1}: formula OCR lacked LaTeX delimiters")
                if confidence == "low" and kind in {"code", "formula", "table"}:
                    keep_image = True
                    rendered = "⚠️ 识别结果待复核\n\n" + rendered
                    warnings.append(
                        f"REVIEW: paragraph {paragraph_index + 1} {kind} OCR confidence is low"
                    )
                existing_notes = paragraph.visual_note or ""
                if rendered not in existing_notes:
                    paragraph.visual_note = (
                        existing_notes.rstrip() + "\n\n" + rendered
                        if existing_notes.strip()
                        else rendered
                    )
                frame.keep_image = keep_image
                frame.extracted_markdown = rendered
        except Exception as exc:
            # The frame was already selected as the strongest temporally
            # aligned evidence for this paragraph. If classification or
            # transcription fails, retain the original instead of silently
            # discarding potentially indispensable visual information.
            frame.paragraph_index = paragraph_index
            frame.keep_image = True
            frame.evidence_confidence = "low"
            frame.evidence_completeness = "unknown"
            warnings.append(f"paragraph {paragraph_index + 1}: visual evidence {type(exc).__name__}")
    return warnings


def _split_text_at_sentence_boundaries(text: str, limit: int = 650) -> list[str]:
    """Split prose deterministically without asking the LLM to rewrite it."""
    units = [item for item in re.split(r"(?<=[。！？!?；;])", text) if item]
    if not units:
        units = [text]
    bounded: list[str] = []
    for unit in units:
        remaining = unit
        while len(remaining) > limit:
            window = remaining[: limit + 1]
            candidates = [window.rfind(mark) + 1 for mark in ("\n", "，", "、", ",", "：", ":")]
            split_at = max((value for value in candidates if value >= limit // 2), default=limit)
            bounded.append(remaining[:split_at].strip())
            remaining = remaining[split_at:]
        if remaining.strip():
            bounded.append(remaining.strip())

    chunks: list[str] = []
    current = ""
    for unit in bounded:
        if current and len(current) + len(unit) > limit:
            chunks.append(current.strip())
            current = unit
        else:
            current += unit
    if current.strip():
        chunks.append(current.strip())
    return chunks


def _split_oversized_paragraphs(
    paragraphs: list[Paragraph], chunk: list[Segment]
) -> list[Paragraph]:
    """Repair formatting while preserving text and exact source-ID coverage."""
    by_id = {segment.id: segment for segment in chunk}
    repaired: list[Paragraph] = []
    for paragraph in paragraphs:
        if len(paragraph.text) <= 700:
            repaired.append(paragraph)
            continue
        text_parts = _split_text_at_sentence_boundaries(paragraph.text)
        owned = [by_id[source_id] for source_id in paragraph.source_ids if source_id in by_id]
        part_count = len(text_parts)
        source_count = len(owned)
        for index, text_part in enumerate(text_parts):
            left = (index * source_count) // part_count
            right = ((index + 1) * source_count) // part_count
            part_sources = owned[left:right]
            if part_sources:
                start, end = part_sources[0].start, part_sources[-1].end
            else:
                span_start = paragraph.start or 0.0
                span_end = paragraph.end if paragraph.end is not None else span_start
                start = span_start + (span_end - span_start) * index / part_count
                end = span_start + (span_end - span_start) * (index + 1) / part_count
            repaired.append(
                Paragraph(
                    source_ids=[segment.id for segment in part_sources],
                    text=text_part,
                    start=start,
                    end=end,
                    heading=paragraph.heading if index == 0 else None,
                )
            )
    return repaired


def _paragraphs_from_response(
    payload: dict[str, Any],
    chunk: list[Segment],
    *,
    enforce_structure: bool = True,
) -> tuple[list[Paragraph], set[str]]:
    if not chunk:
        raise ValueError("cannot edit an empty transcript chunk")
    positions = {segment.id: index for index, segment in enumerate(chunk)}
    if len(positions) != len(chunk):
        raise ValueError("input transcript contains duplicate source IDs")
    items = payload.get("paragraphs") or []
    if not isinstance(items, list) or not items:
        raise ValueError("model returned no manuscript paragraphs")
    if payload.get("removed_filler_ids"):
        raise ValueError("removed_filler_ids is obsolete; use paragraph start boundaries only")

    starts: list[int] = []
    previous = -1
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("invalid paragraph object")
        raw_start_id = item.get("start_source_id")
        if not isinstance(raw_start_id, str):
            raise ValueError("start_source_id must be a string")
        start_id = raw_start_id.strip()
        if start_id not in positions:
            raise ValueError(f"invalid start_source_id: {start_id or '<missing>'}")
        position = positions[start_id]
        if position <= previous:
            raise ValueError("start_source_id values must be unique and strictly chronological")
        starts.append(position)
        previous = position
    if starts[0] != 0:
        raise ValueError(
            f"first start_source_id must be {chunk[0].id}, got {items[0].get('start_source_id')}"
        )

    paragraphs: list[Paragraph] = []
    for item_index, item in enumerate(items):
        start = starts[item_index]
        stop = starts[item_index + 1] if item_index + 1 < len(starts) else len(chunk)
        owned = chunk[start:stop]
        raw_text = item.get("text")
        if not isinstance(raw_text, str):
            raise ValueError("paragraph text must be a string")
        text = raw_text.strip()
        # Apply only unambiguous domain corrections after the model has seen
        # the full context. These variants are common Paraformer reversals or
        # product-name corruptions and should never leak into the manuscript.
        text = re.sub(r"(?i)(?<![A-Za-z])cvs(?![A-Za-z])", "CSV", text)
        text = re.sub(
            r"(?i)(?<![A-Za-z])(?:goodex|couldx|couldex|credex|oodex)(?![A-Za-z])",
            "Codex",
            text,
        )
        text = re.sub(r"(?i)(?<![A-Za-z])sy\s*paper(?![A-Za-z])", "SY Paper", text)
        if not text:
            raise ValueError("empty retained paragraph")
        raw_heading = item.get("heading")
        if raw_heading is not None and not isinstance(raw_heading, str):
            raise ValueError("paragraph heading must be a string or null")
        heading = raw_heading.strip() if isinstance(raw_heading, str) else None
        if heading and heading.lower() in {"null", "none"}:
            heading = None
        paragraphs.append(
            Paragraph(
                source_ids=[segment.id for segment in owned],
                text=text,
                start=owned[0].start,
                end=owned[-1].end,
                heading=heading or None,
            )
        )

    source_chars = sum(len(conservative_clean(segment.text)) for segment in chunk)
    output_chars = sum(len(item.text) for item in paragraphs)
    ratio = output_chars / max(1, source_chars)
    if ratio < 0.48:
        raise ValueError(f"detail retention ratio is too low ({ratio:.2f})")
    for item_index, (paragraph, start) in enumerate(zip(paragraphs, starts)):
        stop = starts[item_index + 1] if item_index + 1 < len(starts) else len(chunk)
        owned = chunk[start:stop]
        owned_chars = sum(len(conservative_clean(segment.text)) for segment in owned)
        if owned_chars >= 160 and len(paragraph.text) / max(1, owned_chars) < 0.35:
            raise ValueError(
                f"local detail retention is too low from {owned[0].id} ({len(paragraph.text)}/{owned_chars})"
            )
        paragraph_output = _normalized_anchor(paragraph.text)
        local_missing = [
            value
            for value in exact_anchors(owned)
            if _normalized_anchor(value) not in paragraph_output
        ]
        if local_missing:
            raise ValueError(
                f"exact details missing from range {owned[0].id}-{owned[-1].id}: "
                + ", ".join(local_missing[:8])
            )
    output_normalized = _normalized_anchor("\n".join(item.text for item in paragraphs))
    missing_anchors = [value for value in exact_anchors(chunk) if _normalized_anchor(value) not in output_normalized]
    if missing_anchors:
        raise ValueError("exact details missing: " + ", ".join(missing_anchors[:8]))
    source_text = "\n".join(segment.text for segment in chunk)
    for phrase in UNSUPPORTED_EDITOR_COMMENTARY:
        if phrase in output_normalized and phrase not in _normalized_anchor(source_text):
            raise ValueError(f"unsupported editor commentary added: {phrase}")
    if enforce_structure:
        paragraphs = _split_oversized_paragraphs(paragraphs, chunk)
        duration = max(0.0, chunk[-1].end - chunk[0].start) if chunk else 0.0
        minimum_paragraphs = max(1, int(duration // 180) + (1 if duration % 180 else 0))
        if any(len(item.text) > 700 for item in paragraphs):
            raise ValueError("manuscript structure failed: oversized paragraph")
        if duration >= 300 and len(paragraphs) < minimum_paragraphs:
            raise ValueError(
                f"manuscript structure failed: {duration:.0f}s became only {len(paragraphs)} paragraph(s)"
            )
        if source_chars >= 2200 and len(paragraphs) < 4:
            raise ValueError("manuscript structure failed: long source was over-merged")
        if source_chars >= 2200 and sum(bool(item.heading) for item in paragraphs) < 2:
            raise ValueError("manuscript structure failed: long note lacks section headings")
    return paragraphs, set()


def _validate_full_manuscript(
    segments: list[Segment], paragraphs: list[Paragraph], removed: set[str] | None = None
) -> None:
    if not paragraphs:
        raise ValueError("manuscript is empty")
    if any(len(item.text) > 700 for item in paragraphs):
        raise ValueError("manuscript contains a paragraph longer than 700 characters")
    expected_ids = [segment.id for segment in segments]
    actual_ids = [source_id for paragraph in paragraphs for source_id in paragraph.source_ids]
    if actual_ids != expected_ids:
        raise ValueError("deterministic source range coverage is incomplete, duplicated, or out of order")
    if removed:
        raise ValueError("removed source IDs are not supported")
    source_chars = sum(len(conservative_clean(item.text)) for item in segments)
    output_chars = sum(len(item.text) for item in paragraphs)
    duration = max(0.0, segments[-1].end - segments[0].start) if segments else 0.0
    minimum_paragraphs = max(1, int((output_chars + 649) // 650), int((duration + 89) // 90))
    if len(paragraphs) < minimum_paragraphs:
        raise ValueError(
            f"manuscript is over-merged: expected at least {minimum_paragraphs} paragraphs, got {len(paragraphs)}"
        )
    minimum_headings = 0 if duration < 90 else min(4, max(2, int((duration + 149) // 150)))
    if sum(bool(item.heading) for item in paragraphs) < minimum_headings:
        raise ValueError(
            f"manuscript lacks workflow headings: expected at least {minimum_headings}"
        )
    rendered = "\n".join(item.text for item in paragraphs)
    source_rendered = "\n".join(item.text for item in segments)
    for phrase in UNSUPPORTED_EDITOR_COMMENTARY:
        if phrase in rendered and phrase not in source_rendered:
            raise ValueError(f"manuscript added unsupported editor commentary: {phrase}")
    if source_chars >= 800 and output_chars / max(1, source_chars) < 0.48:
        raise ValueError("full manuscript detail retention ratio is too low")
    if re.search(r"(?:goodex|could\s*x|credex|oodex|s八配备|sr\s*pick)", rendered, flags=re.I):
        raise ValueError("manuscript still contains obvious ASR product-name corruption")
    if re.search(r"([一-鿿])\1{2,}", rendered):
        raise ValueError("manuscript still contains repeated-character ASR corruption")


def _proofread_asr_artifacts(
    segments: list[Segment],
    paragraphs: list[Paragraph],
    client: OpenAICompatibleClient,
) -> tuple[list[Paragraph], list[str]]:
    """Apply only bounded, auditable, high-confidence substring corrections."""
    warnings: list[str] = []
    source_chars = _source_char_count(segments)
    if not paragraphs or source_chars > 18000:
        return paragraphs, warnings
    original_rendered = "\n".join(item.text for item in paragraphs)
    prompt = (
        f"<original_timed_transcript>\n{_segments_payload(segments)}\n</original_timed_transcript>"
        f"\n<final_manuscript>\n{original_rendered}\n</final_manuscript>"
    )
    try:
        raw = client.chat(
            [
                {"role": "system", "content": ASR_PROOFREAD_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=1800,
            json_mode=True,
        )
        items = parse_json_object(raw).get("corrections")
        if not isinstance(items, list):
            raise ValueError("ASR proofreader omitted corrections list")
    except Exception as exc:
        warnings.append(f"ASR 终校未执行（{type(exc).__name__}）")
        return paragraphs, warnings

    source_anchor_set = {
        _normalized_anchor(value) for value in exact_anchors(segments)
    }
    segment_by_id = {item.id: item for item in segments}
    chinese_number_re = re.compile(r"[零〇一二两三四五六七八九十百千万亿]+")
    replacements: list[tuple[str, str]] = []
    for item in items[:20]:
        if not isinstance(item, dict):
            continue
        original = str(item.get("original") or "").strip()
        replacement = str(item.get("replacement") or "").strip()
        if not (2 <= len(original) <= 80 and 1 <= len(replacement) <= 120):
            continue
        if original_rendered.count(original) != 1 or original == replacement:
            continue
        if any(mark in replacement for mark in ("# ", "```", "<source", "start_source_id")):
            continue
        if any(
            phrase in replacement and phrase not in original_rendered
            for phrase in UNSUPPORTED_EDITOR_COMMENTARY
        ):
            continue
        owner = next(
            (paragraph for paragraph in paragraphs if original in paragraph.text),
            None,
        )
        if owner is None:
            continue
        local_source = "\n".join(
            segment_by_id[source_id].text
            for source_id in owner.source_ids
            if source_id in segment_by_id
        )
        source_numbers = set(re.findall(r"\d+(?:\.\d+)?", local_source))
        source_chinese_numbers = set(chinese_number_re.findall(local_source))
        replacement_numbers = set(re.findall(r"\d+(?:\.\d+)?", replacement))
        replacement_chinese_numbers = set(chinese_number_re.findall(replacement))
        introduced_quantity = (
            not replacement_numbers.issubset(source_numbers)
            or not replacement_chinese_numbers.issubset(source_chinese_numbers)
        )
        if introduced_quantity:
            # A corrupt count is better rendered conservatively than replaced
            # with a confident but invented number. The retained screenshot
            # remains available when the exact visible quantity matters.
            fallback = re.sub(
                r"超[零〇一二两三四五六七八九十百千万亿]+篇文献",
                "大量文献",
                original,
            )
            if fallback == original:
                continue
            replacement = fallback
        replacement_anchors = exact_anchors(
            [Segment("proofread", 0, 1, replacement)]
        )
        if any(
            _normalized_anchor(value) not in source_anchor_set
            for value in replacement_anchors
        ):
            continue
        replacements.append((original, replacement))

    original_texts = [item.text for item in paragraphs]
    applied = 0
    for original, replacement in replacements:
        for paragraph in paragraphs:
            if original in paragraph.text:
                paragraph.text = paragraph.text.replace(original, replacement, 1)
                paragraph.text = re.sub(r"的{2,}", "的", paragraph.text)
                applied += 1
                break
    try:
        _validate_full_manuscript(segments, paragraphs)
    except ValueError:
        for paragraph, text in zip(paragraphs, original_texts):
            paragraph.text = text
        warnings.append("ASR 终校修正未通过原有信息门禁，已全部回滚")
        return paragraphs, warnings
    if applied:
        warnings.append(f"ASR 终校应用 {applied} 处保守修正")
    dequantified = 0
    for paragraph in paragraphs:
        repaired, count = re.subn(
            r"自带超[零〇一二两三四五六七八九十百千万亿]+篇文献",
            "自带大量文献",
            paragraph.text,
        )
        if count:
            paragraph.text = repaired
            dequantified += count
    if dequantified:
        warnings.append(f"ASR 不可靠数量去量化 {dequantified} 处")
    outline_repairs = 0
    for paragraph in paragraphs:
        repaired, count = re.subn(
            r"和[三一]{1,3}大纲政策(?:的)?论文初稿",
            "和大纲生成的论文初稿",
            paragraph.text,
        )
        if count:
            paragraph.text = repaired
            outline_repairs += count
    if outline_repairs:
        warnings.append(f"ASR 不成句大纲短语保守修正 {outline_repairs} 处")
    return paragraphs, warnings


def edit_transcript(
    segments: list[Segment],
    client: OpenAICompatibleClient | None,
    *,
    context: str = "",
    checkpoint_path: Path | None = None,
) -> tuple[list[Paragraph], dict[str, Any]]:
    if client is None:
        raise RuntimeError("未配置文本模型，拒绝发布未编辑的 ASR 逐字稿")

    paragraphs: list[Paragraph] = []
    removed: set[str] = set()
    warnings: list[str] = []
    failed_chunks: list[int] = []
    chunks = chunk_segments(segments)
    completed_chunks = _load_checkpoint_chunks(
        checkpoint_path, segments, chunks, context
    )
    for chunk_index, chunk in enumerate(chunks, start=1):
        if chunk_index in completed_chunks:
            paragraphs.extend(completed_chunks[chunk_index])
            warnings.append(f"第 {chunk_index} 批已从校验通过的检查点恢复")
            continue
        previous_context, subsequent_context = _chunk_context(segments, chunk)
        base_prompt = (
            "先生成忠实信息稿。此阶段优先保证每项有效信息都存在，不要做摘要。"
            f"\n<video_context>{context[:1000] or '未提供'}</video_context>"
            f"\n<previous_context>{previous_context or '无'}</previous_context>"
            f"\n<current_batch>\n{_segments_payload(chunk)}\n</current_batch>"
            f"\n<subsequent_context>{subsequent_context or '无'}</subsequent_context>"
        )
        last_error = ""
        faithful: list[Paragraph] | None = None
        for attempt in range(1, BATCH_EDIT_MAX_ATTEMPTS + 1):
            prompt = base_prompt
            if last_error:
                prompt += f"\n\n上次输出未通过质量门禁：{last_error}。修复后重新输出完整 JSON。"
                prompt += (
                    f"\n第一段必须从 {chunk[0].id} 开始；只输出每段的 start_source_id，"
                    "不要逐条复制 source_ids，也不要输出 removed_filler_ids。"
                )
            try:
                raw = client.chat(
                    [
                        {"role": "system", "content": FAITHFUL_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    max_tokens=8192,
                    json_mode=True,
                )
                # The first pass proves information fidelity only. Formatting
                # defects such as an oversized paragraph are repaired by the
                # dedicated refinement/composition passes; they must not
                # discard an otherwise complete faithful draft.
                faithful, _ = _paragraphs_from_response(
                    parse_json_object(raw), chunk, enforce_structure=False
                )
                break
            except Exception as exc:
                last_error = str(exc)[:240]
        if faithful is None:
            failed_chunks.append(chunk_index)
            # A faithful draft is mandatory.  Fail immediately with the
            # error from this exact batch instead of processing later chunks
            # and accidentally reporting an unrelated optional-refinement
            # warning as the cause.
            raise RuntimeError(
                "语义编辑未通过质量门禁，已拒绝发布原始 ASR 稿；"
                f"失败分批：{chunk_index}；"
                f"第 {chunk_index} 批忠实校订失败：{last_error}"
            )

        # VideoLingo separates faithfulness from expressiveness. Do the same
        # here: a second model pass improves structure and readability only
        # after the first draft has already passed objective retention gates.
        # If refinement fails, keep the valid faithful draft instead of turning
        # a stylistic failure into a lost video job.
        refined = faithful
        if _source_char_count(chunk) >= REFINEMENT_MIN_SOURCE_CHARS:
            refinement_base = (
                "把下面的忠实初稿编辑成详细、可读的最终批次文字稿。"
                "保留全部信息，只改善结构、段落和语言。"
                f"\n<video_context>{context[:1000] or '未提供'}</video_context>"
                f"\n<previous_context>{previous_context or '无'}</previous_context>"
                f"\n<current_source>\n{_segments_payload(chunk)}\n</current_source>"
                f"\n<faithful_draft>{_draft_payload(faithful)}</faithful_draft>"
                f"\n<subsequent_context>{subsequent_context or '无'}</subsequent_context>"
            )
            refinement_error = ""
            for attempt in range(1, BATCH_EDIT_MAX_ATTEMPTS + 1):
                prompt = refinement_base
                if refinement_error:
                    prompt += (
                        f"\n上次润色未通过客观门禁：{refinement_error}。"
                        "请保留忠实初稿的全部有效信息并重新输出完整 JSON。"
                    )
                try:
                    raw = client.chat(
                        [
                            {"role": "system", "content": REFINEMENT_SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.0,
                        max_tokens=8192,
                        json_mode=True,
                    )
                    candidate, _ = _paragraphs_from_response(parse_json_object(raw), chunk)
                    refined = candidate
                    break
                except Exception as exc:
                    refinement_error = str(exc)[:240]
            if refined is faithful and refinement_error:
                warnings.append(
                    f"第 {chunk_index} 批润色未通过，已保留通过门禁的忠实稿：{refinement_error}"
                )
        paragraphs.extend(refined)
        completed_chunks[chunk_index] = refined
        _save_checkpoint_chunks(
            checkpoint_path, segments, context, completed_chunks
        )

    paragraphs.sort(key=lambda item: item.start if item.start is not None else float("inf"))
    first_validation_error = ""
    try:
        _validate_full_manuscript(segments, paragraphs, removed)
    except ValueError as exc:
        first_validation_error = str(exc)[:240]

    # BiliNote merges chunk results hierarchically; for a normal long video we
    # likewise run an explicit whole-manuscript composition pass even when the
    # concatenated chunks are already valid. This is what turns correct local
    # paragraphs into one coherent note. Very large sources stay on the
    # validated chunk path to avoid exceeding provider request budgets.
    source_chars = _source_char_count(segments)
    should_compose = len(chunks) > 1 and source_chars <= 18000
    needs_repair = bool(first_validation_error)
    if should_compose or needs_repair:
        compose_error = first_validation_error or "需要统一全文结构与标题"
        composed: list[Paragraph] | None = None
        compose_base = (
            "下面的忠实草稿已经逐批通过信息保留检查。请把它整理成一篇连贯、详细的完整视频文字稿。"
            "可合并跨批重复、统一术语并按真实话题/操作阶段重排标题，但不得减少有效信息，"
            "不得增加原字幕没有的事实或编辑者评价。"
            "每段只返回 start_source_id；第一段从第一条字幕开始，后续起点严格递增。"
            f"\n<video_context>{context[:1000] or '未提供'}</video_context>"
            f"\n<original_timed_transcript>\n{_segments_payload(segments)}\n</original_timed_transcript>"
            f"\n<faithful_draft>{_draft_payload(paragraphs)}</faithful_draft>"
        )
        for _attempt in range(FULL_MANUSCRIPT_REPAIR_ATTEMPTS):
            prompt = (
                compose_base
                + f"\n当前需要处理的问题：{compose_error}。请返回完整 JSON。"
            )
            try:
                raw = client.chat(
                    [
                        {"role": "system", "content": REFINEMENT_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    max_tokens=8192,
                    json_mode=True,
                )
                candidate, candidate_removed = _paragraphs_from_response(
                    parse_json_object(raw), segments
                )
                _validate_full_manuscript(segments, candidate, candidate_removed)
                composed = candidate
                removed = candidate_removed
                break
            except Exception as exc:
                compose_error = str(exc)[:240]
        if composed is not None:
            paragraphs = composed
        elif not first_validation_error:
            warnings.append(
                "全文统一润色未通过，已保留逐批通过客观门禁的忠实编辑稿："
                + compose_error
            )
        else:
            raise RuntimeError(
                f"全文返工 {FULL_MANUSCRIPT_REPAIR_ATTEMPTS} 次后仍未通过质量门禁："
                + compose_error
            )

    paragraphs, proofread_warnings = _proofread_asr_artifacts(
        segments, paragraphs, client
    )
    warnings.extend(proofread_warnings)
    _validate_full_manuscript(segments, paragraphs, removed)
    coverage = build_coverage(segments, paragraphs, removed, edited=True)
    coverage["warnings"].extend(warnings)
    coverage["failed_chunks"] = failed_chunks
    coverage["quality_status"] = "pass"
    coverage["editing_pipeline"] = "faithful_then_refine_then_compose_then_bounded_proofread"
    if checkpoint_path is not None:
        checkpoint_path.unlink(missing_ok=True)
    return paragraphs, coverage


def build_coverage(
    segments: list[Segment],
    paragraphs: list[Paragraph],
    removed: set[str],
    *,
    edited: bool,
) -> dict[str, Any]:
    mapping: dict[str, dict[str, Any]] = {
        segment.id: {"source_id": segment.id, "status": "missing"} for segment in segments
    }
    for index, paragraph in enumerate(paragraphs):
        status = "kept" if len(paragraph.source_ids) == 1 else "merged"
        for source_id in paragraph.source_ids:
            if source_id in mapping:
                mapping[source_id] = {"source_id": source_id, "status": status, "paragraph": index}
    if removed:
        raise ValueError("removed source IDs are not supported")
    missing = [source_id for source_id, value in mapping.items() if value["status"] == "missing"]
    return {
        "semantic_editing": edited,
        "source_count": len(segments),
        "accounted_count": len(segments) - len(missing),
        "missing_ids": missing,
        "entries": list(mapping.values()),
        "assignment_method": "deterministic_start_ranges",
        "warnings": [] if not missing else [f"{len(missing)} source segments are unaccounted for"],
        "quality_status": "pass" if edited and not missing else "needs_review",
    }
