from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from . import PIPELINE_VERSION
from .llm import OpenAICompatibleClient, parse_json_object
from .models import InformationUnit, OutlineSection, Paragraph, Segment
from .utils import atomic_json, load_json


UNIT_ATTEMPTS = 3
OUTLINE_ATTEMPTS = 2
SECTION_ATTEMPTS = 3
AUDIT_REPAIR_ATTEMPTS = 3
AUDIT_FORMAT_ATTEMPTS = 3
TERMINOLOGY_ATTEMPTS = 2
LEGAL_DROP_REASONS = {"filler", "false_start", "promotional_request", "true_repetition"}
UNSUPPORTED_EDITOR_COMMENTARY = (
    "这说明", "价值在于", "适合用来", "形成互补", "由此可见", "可以帮助", "值得注意的是",
)
TRANSCRIPT_OPENING_RE = re.compile(
    r"^(?:然后|接着|现在|我们(?:来|下面|开始)|下面(?:我们)?|到这里|再来看|大家可以看到)[，、：:\s]*"
)
DIRECT_ADDRESS_RE = re.compile(
    r"今天手把手|同学们|咱们|大家可以|你可以|你只需要|我们(?:需要|开始|下面|接下来|再来|来看)"
)
TRANSCRIPT_CONNECTIVE_RE = re.compile(
    r"(?:^|[。！？]\s*)(?:好[，,、]?|然后|接着|再来|那么|那(?:我们)?|现在(?:我们)?)[，,、\s]"
)
LATIN_ENTITY_RE = re.compile(
    r"\b[A-Za-z][A-Za-z0-9.-]*(?:\s+[A-Za-z][A-Za-z0-9.-]*)?\b"
)

EDITORIAL_BRIEF = """黄金编辑准则：
- 成稿应像创作者事先写好的完整脚本，而不是按时间顺序改写字幕，也不是只保留结论的摘要。
- 先确定本节要讲清的中心，再把相邻信息单元综合成自然段；不要按“一条信息单元一句话”的方式机械拼接。
- 同一段中应明确动作与原因、步骤与结果、观点与例子之间的关系；只有来源真的列举步骤或名单时才使用列表。
- 删除口癖和重复表达，但保留入口、按钮、参数、数字、例子、限制、验证方式和失败条件。
- 使用中性文字稿语气。不要保留主持式称呼和流水连接词，也不要添加编辑者自己的价值判断。

风格示例：
差：好，我们先点设置。然后点击常规。接着往下看语言，这里有一个小细节。
好：在“设置→常规→语言”中切换界面语言；若该选项受网络环境限制，需先满足网络条件，切换后重新打开应用。
示例只说明编辑方式，不允许把示例中的事实写入不相关视频。"""

# Deliberately high precision. Ordinary English words and unknown acronyms are
# not hard anchors: that was the source of the old `open` / `SR` false failures.
HARD_ANCHOR_RE = re.compile(
    r"https?://[^\s，。；、]+|BV[0-9A-Za-z]{10}|"
    r"\d+(?:\.\d+)?\s*(?:%|GB|MB|KB|TB|元|美元|分钟|小时|天|条|个|次|篇|页|字|token)s?|"
    r"《[^》\n]{2,120}》|【[^】\n]{2,120}】|"
    r"--[A-Za-z0-9][A-Za-z0-9_-]*|/(?:[A-Za-z0-9._~-]+/)+[A-Za-z0-9._~-]*|"
    r"\b(?:Codex|Claude(?:\.md)?|ChatGPT|DeepSeek|OpenAI|Anthropic|Gemini|FunASR|Qwen|"
    r"YouTube|Bilibili|Obsidian|GitHub|FDE|Harness|MCP|OCR|ASR|LLM|API)\b",
    re.I,
)

UNIT_SYSTEM_PROMPT = """你是视频内容取证编辑。先把连续口语字幕转换成“信息单元”，不要直接写文章。

信息单元是一组表达同一件事的连续字幕。保留所有有意义的信息：观点、理由、解释、例子、步骤、入口路径、按钮和选项、权限、输入目的、可调参数、等待时间、费用或额度、输出字段、验证方法、失败条件、命令、代码、URL、数字、限定条件、例外、比较、警告、精确人名/产品名/标题和结论。相邻重复表述合并成一个单元，但细节不得泛化。不要把包含多个不同事实、结果或例子的长段字幕偷懒合并成一个单元；连续内容超过约 30 秒或 5 条字幕时，应在语义目标、步骤、结果或例子变化处继续切分。

只有四类内容可丢弃：纯口癖 filler、没有完成含义的假开头 false_start、求赞关注等 promotional_request、已经被相邻单元完整覆盖的真正重复 true_repetition。带有动作、名称、数字、原因、条件或结论的字幕不得丢弃。不得添加字幕没有表达的背景、用途、价值判断或推论。

每个单元只返回 start_source_id，第一单元必须从输入第一条字幕开始，后续起点必须来自输入并按时间递增；程序会确定性分配起点之间的字幕。action 为 keep 或 drop；drop_reason 只能使用上述四值。keep 必须填写可独立理解的 text；details 列出不能在后续写作中丢失的具体细节。

严格返回 JSON：{"units":[{"start_source_id":"s000001","action":"keep","kind":"claim|explanation|example|step|list|warning|conclusion|other","topic":"简短主题","text":"忠实的信息单元","details":["具体细节"],"drop_reason":null}]}。"""

OUTLINE_SYSTEM_PROMPT = """你是长视频文字稿的结构编辑。输入是已经过取证审计的有意义信息单元。只规划文章结构，不写正文。

按视频真实话题、目标或操作阶段组织小节。标题必须具体，例如“在 Codex 中设置项目与检索文献”，不要用“第一部分”“更多内容”等空标题。标题使用视频中出现的完整产品名，不使用未解释的临时缩写。保持原始顺序；不能遗漏、复制或重排信息单元。相邻且服务于同一工作流目标的准备、操作、原因和验证应放在同一节；只有主目标或操作阶段真正改变时才新开一节。短视频不要机械拆太多节，不要为了让每节更短而把一个连续流程拆开。

每节只返回 start_unit_id，第一节必须从第一个信息单元开始，后续起点严格递增，程序会确定性分配范围。objective 说明本节要完整讲清什么；format_hint 只能是 prose、steps、list 或 mixed。

严格返回 JSON：{"sections":[{"start_unit_id":"u000001","title":"具体标题","objective":"本节覆盖目标","format_hint":"prose"}]}。"""

SECTION_SYSTEM_PROMPT = f"""你是视频详细文字稿编辑。根据本节的信息单元写成接近创作者原始脚本的中文笔记：有结构、有细节、自然可读，但不是摘要，也不是逐字字幕。

必须覆盖每个信息单元及其 details，保留精确名称、标题、命令、代码、URL、数字、条件、例外、比较、提醒和操作细节。允许合并相邻重复表达、删除口癖、压缩啰嗦句式、修正常见 ASR 同音错误。不得添加来源没有表达的背景、价值判断、受众建议或推论；不要写“这说明”“价值在于”“适合用来”“形成互补”“可以帮助”等二次总结，除非来源明确说过。

正文应像经过编辑的完整讲稿：围绕一个意思形成自然段；步骤或清单确实存在时才使用 Markdown 列表。把对观众的口语称呼改成中性笔记表达，不保留“今天手把手教同学们”“大家可以”“你只需要”“我们接下来”等主持式话语。观点或推荐需要说明归属时使用“作者”或“视频”；操作步骤直接陈述动作。宣传性、主观性或无法独立核验的评价必须保留其来源归属，不能直接写成无条件事实。例如把“今天手把手教大家怎么用 Codex”写成“视频演示如何使用 Codex”，把“大家可以到网站自取”写成“配套资料可在该网站获取”。不要保留“然后、接着、我们可以看到”等字幕式流水账，不要在已有小节标题下再以“总结：”开头。中英文和中文与数字之间使用自然、统一的排版。每段尽量不超过 500 个汉字。

每段只返回 start_unit_id，第一段必须从本节第一个单元开始，后续起点来自本节并严格递增；程序会确定性分配单元范围。不要返回 heading，标题由程序写入。

{EDITORIAL_BRIEF}

严格返回 JSON：{{"paragraphs":[{{"start_unit_id":"u000001","text":"详细、自然、忠实的正文"}}]}}。"""

AUDIT_SYSTEM_PROMPT = f"""你是视频文字稿终审员。逐项对照信息单元与成稿，并按照黄金编辑准则检查是否真的完成了编辑，而不只是事实没有出错。

检查：1）每个 keep 信息单元的观点和 details 是否完整出现；2）精确名称、标题、命令、URL、数字、条件、例外是否被泛化；3）同一个人、产品、网站或文件格式在不同小节中是否使用一致的完整名称，是否残留与完整名称冲突的 ASR 变体或未解释缩写；4）是否添加来源没有支持的事实或编辑者推论；5）是否仍像连续字幕堆砌、逐条复述信息单元，或保留主持式话语和大量“然后、接着、再来”；6）标题与段落是否按真实话题/操作阶段组织；7）段落是否把相关的动作、原因、结果、例子和限制组织在一起。内容完整但仍像清洗过的字幕，也必须判 repair。

来源信息单元中直接出现的产品名、界面标签或缩写本身就是可用证据；只要成稿忠实保留且文字通顺，不得仅因你不熟悉该名称、无法从外部确认其官方拼写或没有扩写缩写，就判为 exact_detail。终审只检查来源与成稿，不要求外部知识证明。

终审没有删除 keep 信息的权限。口语化的具体例子、时间对比、数量、场景和作者结论仍是有意义证据；若表达不够书面，只能要求中性改写并融入段落，不能要求删掉。repair instruction 必须明确“保留全部信息，仅重组或改写”，不得以改善风格为由牺牲细节。

{EDITORIAL_BRIEF}

必须逐节返回 section_reviews，不能只给整篇一个笼统结论。若某节 status 为 repair，issues 中必须至少有一项对应问题；若全部小节均 pass，整篇 verdict 才能为 pass。

严格返回 JSON：{{"verdict":"pass|repair","section_reviews":[{{"section_id":"sec001","status":"pass|repair","reason":"简短、具体的判断"}}],"issues":[{{"section_id":"sec001","unit_ids":["u000001"],"kind":"missing|unsupported|transcript_like|unit_stitching|presenter_voice|structure|exact_detail","instruction":"具体返工要求"}}]}}。"""

TERMINOLOGY_SYSTEM_PROMPT = """你是谨慎的视频 ASR 术语校订员。输入是完整视频的信息单元和视频标题。

先建立同一视频内部的实体词表，并逐项检查输入提供的 latin_entity_candidates：若某个人名、产品名、网站名或文件格式在一个单元中出现清晰完整形式，而其他单元出现发音相近、字母错位、被错误拆分或临时缩写的形式，可把完整形式作为高置信度证据统一修正。只修正根据全局上下文可以高置信度确认的同音字、产品名拆分、字母倒置或明显不成立的短语。若 ASR 把常见复合词错误拆成相邻列表项，且同一列表的语法类别足以唯一确认正确边界，也应做最小修正。original 必须是某一个 unit 中精确存在的最小连续子串，replacement 必须是最小替换。无法确认的数量不要猜数字，可改成不带新数量的保守表达。不得改写整句、润色风格、删除细节、添加背景或凭空发明名称、URL、命令和数字。普通口语问题留给写作阶段，不属于 corrections。

严格返回 JSON：{"corrections":[{"unit_id":"u000001","original":"错误短语","replacement":"保守修正","confidence":"high","reason":"为什么能从全局上下文确认"}]}。没有可靠修正时返回 {"corrections":[]}。"""

ENTITY_RECONCILIATION_SYSTEM_PROMPT = """你只复核同一视频中的拉丁字母实体一致性，不做普通润色。

输入包含完整信息单元和多词拉丁候选。寻找同一个人、产品、网站、文件格式在不同单元里的完整形式与明显 ASR 变体，例如字母错位、发音相近、错误拆分或临时缩写。只有当完整形式已在其他信息单元明确出现、上下文又指向同一实体时才修正；普通缩写（如 API、DOI）、不同产品和无法确认的候选保持不变。

original 必须是目标 unit 中唯一存在的最小连续子串，replacement 必须直接使用其他 unit 已出现的完整形式。不得添加新实体、改写整句或改变事实。

严格返回 JSON：{"corrections":[{"unit_id":"u000001","original":"错误变体","replacement":"同视频已出现的完整形式","confidence":"high","reason":"跨单元证据"}]}。没有可靠修正时返回 {"corrections":[]}。"""

FINAL_AUDIT_ADJUDICATION_SYSTEM_PROMPT = """你是视频文字稿的保真仲裁员，不负责重新写稿。输入包含终审在返工上限后仍提出的问题、相关 keep 信息单元和对应成稿。

逐项判断终审意见是否真的是发布阻塞：
- 缺失事实、添加来源外事实、精确名称/数字/命令错误、明显 ASR 乱码，或当前成稿仍可明确观察到逐信息单元拼句、冗余“总结：”、主持式称呼和流水连接词，valid=true。
- 终审仅仅不喜欢 UP 主自己的语气、观点、结论、口语化例子或具体场景，或者要求删除 keep 信息，valid=false。作者说“非常简单”“可以去食堂吃饭”、时间对比等属于来源内容；可被中性整合，但不能仅因非正式就成为永久阻塞。
- 不要相信 issue 自己声称“缺失”，必须逐句读取当前 draft。issue 要求的设置、动作、数字或例子已经以同义方式出现在 draft 时，valid=false。只有能明确列出仍未出现的事实时，missing 才能 valid=true。
- 信息单元中直接出现的产品名、界面标签或缩写本身就是来源证据。若 draft 忠实保留且语句通顺，不得仅因无法从外部确认官方名称、不了解该标签或没有扩写缩写就判 valid；仲裁不做外部事实核验。
- 前一轮已经通过、当前成稿也没有出现该问题时，valid=false。
- 不得因为想让任务完成而放过事实问题。

每个 issue_index 必须恰好返回一次。missing_details 仅列出对照 draft 后确实缺失的事实；draft_evidence 引用已经出现的相关原文。严格返回 JSON：{"decisions":[{"issue_index":0,"valid":true,"missing_details":["确实缺失的事实"],"draft_evidence":["当前成稿原文"],"reason":"基于信息单元与成稿的具体理由"}]}。"""

PUBLICATION_COPYEDIT_INSTRUCTION = """这是发布前逐句校对，不是重新摘要或重新规划结构。请对照 current_draft 和全部 information_units，保持现有小节目标与信息顺序，完成以下工作：
1. 修复仍然不通顺的 ASR 同音、错词、断词和列表词边界；若无法高置信度还原，改成不引入新事实的保守通顺表达，不得留下明显不成句的碎片。
2. 删除口癖、主持式称呼、冗余“总结：”和流水连接词，把同一意思组织成自然段，但不能删除任何事实、例子、步骤、数字、条件、界面名称和结论。
3. 将宣传性、主观性或无法独立核验的评价归属给“作者”或“视频”，不要把来源观点写成无条件事实。
4. 统一人名、产品名、文件格式及中英文和数字排版。
5. 不添加背景知识、用途、价值判断或来源没有表达的推论。
返回完整的本节 paragraphs；必须逐项覆盖 information_units。"""

ASR_RESIDUE_PROOFREAD_SYSTEM_PROMPT = """你是每轮终审前及发布前的独立 ASR 残留审校员，不负责改结构、润色风格或总结内容。逐节对照 information_units 与 current_draft，只检查最终文字中仍然不成句的同音错词、断词、错误列表边界、乱码短语和同一实体的不一致写法。

对每个问题返回最小 original 和 replacement。能从语法、同节信息或完整实体高置信确认时直接修正；若原始语音本身含混，replacement 应改成来源已支持的保守上位表达，不得猜测具体名称、参数或数字。不得删除一整条有意义事实，不得添加背景或编辑评价。不要修改本来通顺的作者观点、例子和口语场景。

必须逐句检查每个 section_id。典型错误及安全处理：
- “封面页、眉页、脚”是同类别复合词边界错误，应改为“封面、页眉、页脚”。
- “切换指文”无法解释，但上下文只要求列出若干参数，可改为“其他相关参数”。
- “复制串软络并粘贴”中的宾语无法可靠还原，但动作明确，可改为“复制相应内容并粘贴”。
- “请根据我的论文浏览器”中的对象明显不成句，而上下文只支持论文信息，可改为“根据论文信息生成参考文献列表”。
这些都是保留已知动作或类别、舍弃无法恢复的乱码表面形式，不是删除有意义事实。反之，`SR pick` 等来源中直接出现、放在句中也通顺的产品或界面标签不得因为你不了解其官方含义而擅自改名或要求外部证明。

若 original 在指定段落中出现多次：所有位置都是同一种乱码时设置 `replace_all:true`；只有某一次需要修正时设置 `occurrence:1`（从 1 开始计数）并提供足够具体的上下文。若仍有无法给出安全 replacement 的不通顺片段，写入 unresolved；不要假装 pass。严格返回 JSON：{"checked_section_ids":["sec001"],"corrections":[{"section_id":"sec001","paragraph_index":0,"original":"原文中的连续片段","replacement":"保守通顺替换","replace_all":false,"occurrence":1,"reason":"具体理由"}],"unresolved":[]}。"""


def _payload_segments(segments: Iterable[Segment]) -> str:
    return "\n".join(f"[{s.id} {s.start:.2f}-{s.end:.2f}] {s.text}" for s in segments)


def _chunks(segments: list[Segment], max_chars: int = 3400) -> list[list[Segment]]:
    result: list[list[Segment]] = []
    current: list[Segment] = []
    size = 0
    for segment in segments:
        cost = len(segment.text) + 32
        if current and size + cost > max_chars:
            result.append(current)
            current, size = [], 0
        current.append(segment)
        size += cost
    if current:
        result.append(current)
    return result


def _call_json(
    client: OpenAICompatibleClient,
    system: str,
    payload: dict[str, Any],
    *,
    max_tokens: int = 6000,
) -> dict[str, Any]:
    text = client.chat(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        temperature=0.1,
        max_tokens=max_tokens,
        json_mode=True,
    )
    return parse_json_object(text)


def _start_ranges(items: list[Any], starts: list[str], *, id_attr: str = "id") -> list[list[Any]]:
    ids = [str(getattr(item, id_attr)) for item in items]
    if not items or not starts or starts[0] != ids[0] or len(starts) != len(set(starts)):
        raise ValueError("range starts must begin at the first item and be unique")
    positions = []
    previous = -1
    for start in starts:
        if start not in ids:
            raise ValueError(f"unknown range start: {start}")
        position = ids.index(start)
        if position <= previous:
            raise ValueError("range starts must be chronological")
        positions.append(position)
        previous = position
    return [items[position : positions[index + 1] if index + 1 < len(positions) else len(items)] for index, position in enumerate(positions)]


def _anchors(text: str) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for match in HARD_ANCHOR_RE.finditer(text):
        value = match.group(0).rstrip(".,;:!?，。；：！？")
        key = value.casefold()
        if value and key not in seen:
            seen.add(key)
            result.append(value)
    return result


def _contains(text: str, anchor: str) -> bool:
    normalized = re.sub(r"\s+", "", text).casefold()
    target = re.sub(r"\s+", "", anchor).casefold()
    return target in normalized


def _chinese_integer(text: str) -> int | None:
    digits = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "幺": 1}
    units = {"十": 10, "百": 100, "千": 1000}
    if not text or any(char not in digits and char not in units for char in text):
        return None
    if not any(char in units for char in text):
        try:
            return int("".join(str(digits[char]) for char in text))
        except ValueError:
            return None
    total = 0
    current = 0
    for char in text:
        if char in digits:
            current = digits[char]
        else:
            total += (current or 1) * units[char]
            current = 0
    return total + current


def _anchor_supported(evidence: str, anchor: str) -> bool:
    if _contains(evidence, anchor):
        return True
    percent = re.fullmatch(r"(\d+)\s*%", anchor)
    if percent:
        expected = int(percent.group(1))
        for verbal in re.findall(r"百分之([零〇一二两三四五六七八九幺十百千]+)", evidence):
            if _chinese_integer(verbal) == expected:
                return True
    measured = re.fullmatch(
        r"(\d+)\s*(GB|MB|KB|TB|元|美元|分钟|小时|天|条|个|次|篇|页|字|token)s?",
        anchor,
        flags=re.I,
    )
    if measured:
        expected = int(measured.group(1))
        unit = measured.group(2)
        pattern = rf"([零〇一二两三四五六七八九幺十百千]+)\s*{re.escape(unit)}"
        for verbal in re.findall(pattern, evidence, flags=re.I):
            if _chinese_integer(verbal) == expected:
                return True
    return False


def _strip_transcript_opening(text: str) -> str:
    cleaned = TRANSCRIPT_OPENING_RE.sub("", text, count=1).lstrip("，、：:；;。 ")
    return cleaned if cleaned else text


def _neutralize_direct_address(text: str) -> str:
    result = re.sub(r"今天手把手教(?:同学们|大家)?(?:怎么|如何)?", "视频演示如何", text)
    result = result.replace("硕士生、博士生同学们", "硕士生和博士生")
    result = result.replace("同学们的", "不同用户的")
    result = result.replace("同学们", "用户")
    result = result.replace("大家可以", "可")
    result = result.replace("你只需要", "只需")
    result = result.replace("你可以", "可以")
    result = result.replace("咱们", "操作中")
    result = result.replace("我们需要", "需要")
    result = result.replace("我们接下来", "接下来")
    result = result.replace("我们下面", "下面")
    result = result.replace("我们开始", "开始")
    return result


def manuscript_quality_report(
    sections: list[OutlineSection],
    paragraphs_by_section: dict[str, list[Paragraph]],
) -> dict[str, Any]:
    """Return deterministic editorial diagnostics without pretending to judge meaning.

    Semantic completeness remains an evidence/model audit. These checks cover only
    observable failure modes that previously let cleaned subtitle dumps look complete.
    """
    paragraphs = [
        paragraph
        for section in sections
        for paragraph in paragraphs_by_section.get(section.id, [])
    ]
    full_text = "\n".join(paragraph.text for paragraph in paragraphs)
    paragraph_lengths = [len(paragraph.text) for paragraph in paragraphs]
    presenter_terms = sorted(set(DIRECT_ADDRESS_RE.findall(full_text)))
    transcript_connectives = TRANSCRIPT_CONNECTIVE_RE.findall(full_text)
    commentary = sorted(
        phrase for phrase in UNSUPPORTED_EDITOR_COMMENTARY if phrase in full_text
    )
    blockers: list[str] = []
    if not sections or not paragraphs:
        blockers.append("empty_manuscript")
    if any(length > 650 for length in paragraph_lengths):
        blockers.append("giant_paragraph")
    if presenter_terms:
        blockers.append("presenter_voice")
    return {
        "status": "pass" if not blockers else "fail",
        "blockers": blockers,
        "section_count": len(sections),
        "paragraph_count": len(paragraphs),
        "maximum_paragraph_chars": max(paragraph_lengths, default=0),
        "presenter_language": presenter_terms,
        "transcript_connective_count": len(transcript_connectives),
        "unsupported_commentary_candidates": commentary,
    }


def _extract_units(
    segments: list[Segment],
    client: OpenAICompatibleClient,
    context: str,
) -> list[InformationUnit]:
    units: list[InformationUnit] = []
    unit_number = 1
    all_chunks = _chunks(segments)
    for chunk_index, chunk in enumerate(all_chunks):
        positions = {segment.id: index for index, segment in enumerate(segments)}
        start = positions[chunk[0].id]
        stop = positions[chunk[-1].id] + 1
        neighbour = {
            "previous_context": _payload_segments(segments[max(0, start - 3) : start]),
            "subsequent_context": _payload_segments(segments[stop : stop + 3]),
        }
        error = ""
        for attempt in range(UNIT_ATTEMPTS):
            payload = {
                "video_context": context,
                "chunk_index": chunk_index + 1,
                "chunk_count": len(all_chunks),
                "segments": _payload_segments(chunk),
                **neighbour,
            }
            if error:
                payload["repair_required"] = error
            try:
                raw = _call_json(client, UNIT_SYSTEM_PROMPT, payload)
                rows = raw.get("units")
                if not isinstance(rows, list) or not rows:
                    raise ValueError("units must be a non-empty list")
                starts = [str(row.get("start_source_id") or "") for row in rows]
                ranges = _start_ranges(chunk, starts)
                chunk_units: list[InformationUnit] = []
                for row, assigned in zip(rows, ranges):
                    action = str(row.get("action") or "keep").strip().lower()
                    reason = str(row.get("drop_reason") or "").strip().lower() or None
                    text = " ".join(str(row.get("text") or "").split()).strip()
                    details = [" ".join(str(value).split()).strip() for value in (row.get("details") or []) if str(value).strip()]
                    if action not in {"keep", "drop"}:
                        raise ValueError("unit action must be keep or drop")
                    if action == "drop" and reason not in LEGAL_DROP_REASONS:
                        raise ValueError("drop unit has an illegal reason")
                    if action == "keep" and (not text or reason is not None):
                        raise ValueError("kept unit requires text and no drop_reason")
                    if (
                        action == "keep"
                        and len(assigned) > 5
                        and assigned[-1].end - assigned[0].start > 30
                    ):
                        raise ValueError(
                            "meaningful unit is too broad; split it at semantic changes"
                        )
                    source_text = " ".join(item.text for item in assigned)
                    anchors = _anchors(source_text)
                    if action == "drop" and anchors:
                        raise ValueError(f"drop unit contains exact details: {', '.join(anchors[:3])}")
                    introduced = [
                        anchor for anchor in _anchors(text + " " + " ".join(details))
                        if not _anchor_supported(source_text + " " + context, anchor)
                    ]
                    if introduced:
                        raise ValueError(f"unit invented exact details: {', '.join(introduced[:3])}")
                    chunk_units.append(
                        InformationUnit(
                            id=f"u{unit_number + len(chunk_units):06d}",
                            source_ids=[item.id for item in assigned],
                            start=assigned[0].start,
                            end=assigned[-1].end,
                            action=action,
                            kind=str(row.get("kind") or "other").strip() or "other",
                            topic=str(row.get("topic") or "").strip(),
                            text=text,
                            details=details,
                            exact_anchors=anchors,
                            drop_reason=reason,
                        )
                    )
                units.extend(chunk_units)
                unit_number += len(chunk_units)
                break
            except Exception as exc:
                error = f"上次输出未通过信息单元门禁：{exc}。请重新返回完整、合法的 JSON。"
        else:
            raise RuntimeError(f"第 {chunk_index + 1} 批信息单元提取失败：{error}")
    return units


def _plan_outline(
    units: list[InformationUnit],
    client: OpenAICompatibleClient,
    context: str,
) -> list[OutlineSection]:
    kept = [unit for unit in units if unit.action == "keep"]
    if not kept:
        raise RuntimeError("没有提取到可写入文稿的有效信息")
    rows_payload = [
        {"id": unit.id, "kind": unit.kind, "topic": unit.topic, "text": unit.text, "details": unit.details}
        for unit in kept
    ]
    if len(kept) == 1:
        preferred_sections = max_sections = 1
    else:
        preferred_sections = min(7, max(2, (len(kept) + 11) // 12))
        max_sections = min(len(kept), min(8, preferred_sections + 1))
    error = ""
    for _attempt in range(OUTLINE_ATTEMPTS):
        payload: dict[str, Any] = {
            "video_context": context,
            "information_units": rows_payload,
            "preferred_sections": preferred_sections,
            "maximum_sections": max_sections,
            "outline_instruction": (
                "优先接近 preferred_sections；只有确有独立主目标时才增加，且不得超过 maximum_sections。"
                "不要为一句过渡、一个小补充或同一流程的连续动作单独建节。"
            ),
        }
        if error:
            payload["repair_required"] = error
        try:
            raw = _call_json(client, OUTLINE_SYSTEM_PROMPT, payload, max_tokens=3000)
            rows = raw.get("sections")
            if not isinstance(rows, list) or not rows:
                raise ValueError("sections must be a non-empty list")
            if len(rows) > max_sections:
                raise ValueError(f"outline is over-fragmented: {len(rows)} > {max_sections}")
            starts = [str(row.get("start_unit_id") or "") for row in rows]
            ranges = _start_ranges(kept, starts)
            sections = []
            for index, (row, assigned) in enumerate(zip(rows, ranges), 1):
                title = " ".join(str(row.get("title") or "").split()).strip()
                if len(title) < 4 or re.fullmatch(r"第[一二三四五六七八九十\d]+部分", title):
                    raise ValueError("section title is empty or generic")
                hint = str(row.get("format_hint") or "prose").strip().lower()
                if hint not in {"prose", "steps", "list", "mixed"}:
                    hint = "prose"
                assigned_evidence = " ".join(
                    unit.text + " " + " ".join(unit.details) for unit in assigned
                )
                invented_title_anchors = [
                    anchor for anchor in _anchors(title) if not _anchor_supported(assigned_evidence, anchor)
                ]
                if invented_title_anchors:
                    raise ValueError(
                        f"section title invented exact details: {', '.join(invented_title_anchors[:3])}"
                    )
                sections.append(
                    OutlineSection(
                        id=f"sec{index:03d}",
                        title=title,
                        unit_ids=[unit.id for unit in assigned],
                        objective=str(row.get("objective") or "").strip(),
                        format_hint=hint,
                    )
                )
            return sections
        except Exception as exc:
            error = f"上次大纲未通过门禁：{exc}。请按原顺序重新规划。"
    raise RuntimeError(f"文章大纲生成失败：{error}")


def _apply_contextual_normalizations(
    units: list[InformationUnit], context: str
) -> list[dict[str, str]]:
    corrections: list[dict[str, str]] = []
    aliases: list[tuple[re.Pattern[str], str]] = []
    if "codex" in context.casefold():
        aliases.append((
            re.compile(r"(?<![A-Za-z0-9])(?:Goodex|Couldx|Couldex|Credex|Oodex)(?![A-Za-z0-9])", re.I),
            "Codex",
        ))

    spoken_domain = re.compile(
        r"([零〇一二两三四五六七八九幺十百千]{1,8})([A-Za-z][A-Za-z0-9-]*)点([A-Za-z]{2,12})",
        re.I,
    )
    malformed_quantity = re.compile(r"自带超([^，。；\s]{1,6})篇文献")
    malformed_overdue = re.compile(r"自带超期(?:一篇)?文献")
    malformed_outline = re.compile(r"和[三一]{1,3}大纲(?:政策)?(?:的)?论文初稿")

    def normalized(text: str, unit_id: str) -> str:
        def replace_domain(match: re.Match[str]) -> str:
            number = _chinese_integer(match.group(1))
            if number is None:
                return match.group(0)
            replacement = f"{number}{match.group(2)}.{match.group(3)}"
            corrections.append({
                "unit_id": unit_id, "original": match.group(0), "replacement": replacement,
                "reason": "口述域名确定性规范化",
            })
            return replacement

        result = spoken_domain.sub(replace_domain, text)
        for pattern, replacement in aliases:
            for match in list(pattern.finditer(result)):
                corrections.append({
                    "unit_id": unit_id, "original": match.group(0), "replacement": replacement,
                    "reason": "标题上下文支持的产品名别名",
                })
            result = pattern.sub(replacement, result)

        def replace_quantity(match: re.Match[str]) -> str:
            verbal = match.group(1)
            if _chinese_integer(verbal) is not None:
                return match.group(0)
            replacement = "自带大量文献"
            corrections.append({
                "unit_id": unit_id, "original": match.group(0), "replacement": replacement,
                "reason": "无法确认的 ASR 数量去量化",
            })
            return replacement

        result = malformed_quantity.sub(replace_quantity, result)
        for match in list(malformed_overdue.finditer(result)):
            corrections.append({
                "unit_id": unit_id, "original": match.group(0),
                "replacement": "自带大量文献", "reason": "明显不成立的 ASR 数量短语去量化",
            })
        result = malformed_overdue.sub("自带大量文献", result)
        result = re.sub(
            r"(?<![A-Za-z0-9])SY\s*paper(?![A-Za-z0-9])",
            "SY Paper",
            result,
            flags=re.I,
        )
        for match in list(malformed_outline.finditer(result)):
            corrections.append({
                "unit_id": unit_id, "original": match.group(0),
                "replacement": "和大纲生成的论文初稿", "reason": "不成句的大纲短语保守修正",
            })
        return malformed_outline.sub("和大纲生成的论文初稿", result)

    for unit in units:
        unit.text = normalized(unit.text, unit.id)
        unit.details = [normalized(detail, unit.id) for detail in unit.details]
    deduplicated: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for correction in corrections:
        key = (correction["unit_id"], correction["original"], correction["replacement"])
        if key not in seen:
            seen.add(key)
            deduplicated.append(correction)
    return deduplicated


def _normalize_units(
    units: list[InformationUnit],
    client: OpenAICompatibleClient,
    context: str,
) -> list[dict[str, str]]:
    kept = [unit for unit in units if unit.action == "keep"]
    deterministic = _apply_contextual_normalizations(kept, context)
    payload_units = [
        {"id": unit.id, "topic": unit.topic, "text": unit.text, "details": unit.details}
        for unit in kept
    ]
    entity_candidates = sorted(
        {
            match.group(0).strip()
            for unit in kept
            for match in LATIN_ENTITY_RE.finditer(
                unit.text + " " + " ".join(unit.details)
            )
            if len(match.group(0).strip()) >= 2
        },
        key=str.casefold,
    )
    error = ""
    for _attempt in range(TERMINOLOGY_ATTEMPTS):
        payload: dict[str, Any] = {
            "video_context": context,
            "information_units": payload_units,
            "latin_entity_candidates": entity_candidates,
        }
        if error:
            payload["repair_required"] = error
        try:
            raw = _call_json(client, TERMINOLOGY_SYSTEM_PROMPT, payload, max_tokens=3000)
            rows = raw.get("corrections")
            if not isinstance(rows, list):
                raise ValueError("corrections must be a list")
            by_id = _unit_map(units)
            working = {
                unit.id: {"text": unit.text, "details": list(unit.details)} for unit in kept
            }
            accepted: list[dict[str, str]] = list(deterministic)

            def accept_rows(correction_rows: list[Any]) -> int:
                accepted_count = 0
                for row in correction_rows:
                    if not isinstance(row, dict):
                        raise ValueError("correction row must be an object")
                    unit_id = str(row.get("unit_id") or "")
                    original = str(row.get("original") or "").strip()
                    replacement = str(row.get("replacement") or "").strip()
                    confidence = str(row.get("confidence") or "").strip().lower()
                    if unit_id not in by_id or by_id[unit_id].action != "keep":
                        raise ValueError(f"unknown correction unit: {unit_id}")
                    if confidence != "high" or not original or not replacement or original == replacement:
                        continue
                    fields = [working[unit_id]["text"], *working[unit_id]["details"]]
                    if sum(field.count(original) for field in fields) != 1:
                        raise ValueError(f"correction original is not unique in {unit_id}")
                    source_support = " ".join(fields) + " " + context
                    unsupported = [
                        anchor for anchor in _anchors(replacement)
                        if not _anchor_supported(source_support, anchor)
                    ]
                    if unsupported:
                        raise ValueError(
                            "correction introduced unsupported exact detail: "
                            + ", ".join(unsupported)
                        )
                    working[unit_id]["text"] = working[unit_id]["text"].replace(
                        original, replacement, 1
                    )
                    working[unit_id]["details"] = [
                        detail.replace(original, replacement, 1)
                        for detail in working[unit_id]["details"]
                    ]
                    accepted.append(
                        {
                            "unit_id": unit_id,
                            "original": original,
                            "replacement": replacement,
                            "reason": str(row.get("reason") or "").strip(),
                        }
                    )
                    accepted_count += 1
                return accepted_count

            model_correction_count = accept_rows(rows)
            multiword_candidates = [
                candidate for candidate in entity_candidates if " " in candidate
            ]
            if len(multiword_candidates) >= 2 and model_correction_count == 0:
                try:
                    entity_units = [
                        {
                            "id": unit.id,
                            "text": working[unit.id]["text"],
                            "details": working[unit.id]["details"],
                        }
                        for unit in kept
                    ]
                    entity_raw = _call_json(
                        client,
                        ENTITY_RECONCILIATION_SYSTEM_PROMPT,
                        {
                            "video_context": context,
                            "information_units": entity_units,
                            "multiword_latin_candidates": multiword_candidates,
                        },
                        max_tokens=2000,
                    )
                    entity_rows = entity_raw.get("corrections")
                    if isinstance(entity_rows, list):
                        accept_rows(entity_rows)
                except Exception:
                    # Entity reconciliation is an optional high-confidence pass;
                    # the ordinary evidence and final audit remain authoritative.
                    pass
            for unit in kept:
                unit.text = str(working[unit.id]["text"])
                unit.details = list(working[unit.id]["details"])
            return accepted
        except Exception as exc:
            error = f"上次术语校订输出无效：{exc}。只返回高置信度最小替换。"
    # Terminology correction is conservative and optional. Information-unit
    # text remains usable when the provider cannot return a valid correction set.
    return deterministic


def _unit_map(units: list[InformationUnit]) -> dict[str, InformationUnit]:
    return {unit.id: unit for unit in units}


def _compose_one_section(
    section: OutlineSection,
    units: list[InformationUnit],
    client: OpenAICompatibleClient,
    context: str,
    neighbouring_titles: list[str],
    repair_instruction: str = "",
    current_draft: list[Paragraph] | None = None,
) -> list[Paragraph]:
    by_id = _unit_map(units)
    selected = [by_id[unit_id] for unit_id in section.unit_ids]
    unit_payload = [
        {
            "id": unit.id,
            "kind": unit.kind,
            "topic": unit.topic,
            "text": unit.text,
            "details": unit.details,
            "exact_anchors": unit.exact_anchors,
        }
        for unit in selected
    ]
    error = repair_instruction
    for _attempt in range(SECTION_ATTEMPTS):
        payload: dict[str, Any] = {
            "video_context": context,
            "section": section.to_dict(),
            "neighbouring_section_titles": neighbouring_titles,
            "information_units": unit_payload,
        }
        if current_draft:
            payload["current_draft"] = [paragraph.to_dict() for paragraph in current_draft]
            payload["revision_instruction"] = (
                "对照 current_draft 定点修复 repair_required；不要忽略旧稿后从头生成近似版本。"
            )
        if error:
            payload["repair_required"] = error
        try:
            raw = _call_json(client, SECTION_SYSTEM_PROMPT, payload)
            rows = raw.get("paragraphs")
            if not isinstance(rows, list) or not rows:
                raise ValueError("paragraphs must be a non-empty list")
            starts = [str(row.get("start_unit_id") or "") for row in rows]
            ranges = _start_ranges(selected, starts)
            paragraphs: list[Paragraph] = []
            full_text = ""
            for index, (row, assigned) in enumerate(zip(rows, ranges)):
                text = _neutralize_direct_address(
                    _strip_transcript_opening(str(row.get("text") or "").strip())
                )
                if not text:
                    raise ValueError("paragraph text is empty")
                if len(text) > 650:
                    raise ValueError("paragraph is still transcript-sized")
                source_ids = [source_id for unit in assigned for source_id in unit.source_ids]
                paragraphs.append(
                    Paragraph(
                        source_ids=source_ids,
                        text=text,
                        start=assigned[0].start,
                        end=assigned[-1].end,
                        heading=section.title if index == 0 else None,
                    )
                )
                full_text += "\n" + text
            missing = sorted(
                {anchor for unit in selected for anchor in unit.exact_anchors if not _anchor_supported(full_text, anchor)},
                key=str.casefold,
            )
            if missing:
                raise ValueError(f"exact details missing: {', '.join(missing[:8])}")
            source_evidence = " ".join(
                unit.text + " " + " ".join(unit.details) for unit in selected
            )
            introduced = [
                anchor for anchor in _anchors(full_text) if not _anchor_supported(source_evidence, anchor)
            ]
            if introduced:
                raise ValueError(f"unsupported exact details introduced: {', '.join(introduced[:8])}")
            source_commentary = " ".join(unit.text for unit in selected)
            unsupported = [phrase for phrase in UNSUPPORTED_EDITOR_COMMENTARY if phrase in full_text and phrase not in source_commentary]
            if unsupported:
                raise ValueError(f"unsupported editor commentary: {', '.join(unsupported)}")
            direct_address = sorted(set(DIRECT_ADDRESS_RE.findall(full_text)))
            if direct_address:
                raise ValueError(
                    f"presenter-style audience language remains: {', '.join(direct_address[:6])}"
                )
            return paragraphs
        except Exception as exc:
            error = f"上次本节成稿未通过门禁：{exc}。逐项覆盖 information_units 后重写本节。"
    raise RuntimeError(f"第 {section.id} 节成稿失败：{error}")


def _audit(
    sections: list[OutlineSection],
    units: list[InformationUnit],
    paragraphs_by_section: dict[str, list[Paragraph]],
    client: OpenAICompatibleClient,
    context: str,
) -> dict[str, Any]:
    by_id = _unit_map(units)
    quality_report = manuscript_quality_report(sections, paragraphs_by_section)
    payload = {
        "video_context": context,
        "deterministic_editorial_diagnostics": quality_report,
        "sections": [
            {
                **section.to_dict(),
                "information_units": [by_id[unit_id].to_dict() for unit_id in section.unit_ids],
                "draft": [paragraph.to_dict() for paragraph in paragraphs_by_section[section.id]],
            }
            for section in sections
        ],
    }
    error = ""
    for _attempt in range(AUDIT_FORMAT_ATTEMPTS):
        request = dict(payload)
        if error:
            request["repair_required"] = error
        try:
            raw = _call_json(client, AUDIT_SYSTEM_PROMPT, request, max_tokens=4000)
            verdict = str(raw.get("verdict") or "repair").strip().lower()
            issues = raw.get("issues") if isinstance(raw.get("issues"), list) else []
            reviews = raw.get("section_reviews")
            if verdict not in {"pass", "repair"}:
                raise ValueError("audit verdict must be pass or repair")
            if not isinstance(reviews, list):
                raise ValueError("audit must review every section")
            review_ids = [str(review.get("section_id") or "") for review in reviews]
            expected_ids = [section.id for section in sections]
            if sorted(review_ids) != sorted(expected_ids) or len(review_ids) != len(set(review_ids)):
                raise ValueError("audit section_reviews must cover every section exactly once")
            review_status = {
                str(review.get("section_id") or ""): str(review.get("status") or "").lower()
                for review in reviews
            }
            if any(status not in {"pass", "repair"} for status in review_status.values()):
                raise ValueError("audit section status must be pass or repair")
            if verdict == "pass" and issues:
                raise ValueError("pass audit cannot contain issues")
            repaired_sections = {str(issue.get("section_id") or "") for issue in issues}
            expected_repairs = {
                section_id for section_id, status in review_status.items() if status == "repair"
            }
            if verdict == "pass" and expected_repairs:
                raise ValueError("pass audit cannot mark a section for repair")
            if verdict == "repair" and (not expected_repairs or not expected_repairs.issubset(repaired_sections)):
                raise ValueError("repair audit must provide an issue for every failed section")
            if quality_report["status"] != "pass" and verdict == "pass":
                raise ValueError(
                    "audit passed despite deterministic editorial blockers: "
                    + ", ".join(quality_report["blockers"])
                )
            return {
                "verdict": verdict,
                "section_reviews": reviews,
                "issues": issues,
                "deterministic_editorial_diagnostics": quality_report,
            }
        except Exception as exc:
            error = f"上次终审 JSON 无效：{exc}。重新输出简短、完整的 JSON。"
    raise RuntimeError(f"全稿终审响应无法解析：{error}")


def _adjudicate_final_audit(
    sections: list[OutlineSection],
    units: list[InformationUnit],
    paragraphs_by_section: dict[str, list[Paragraph]],
    issues: list[dict[str, Any]],
    client: OpenAICompatibleClient,
    context: str,
) -> dict[str, Any]:
    by_id = _unit_map(units)
    section_by_id = {section.id: section for section in sections}
    adjudication_items: list[dict[str, Any]] = []
    for index, issue in enumerate(issues):
        section_id = str(issue.get("section_id") or "")
        section = section_by_id.get(section_id)
        if section is None:
            raise ValueError(f"final audit issue references unknown section: {section_id}")
        adjudication_items.append(
            {
                "issue_index": index,
                "issue": issue,
                "information_units": [
                    by_id[unit_id].to_dict() for unit_id in section.unit_ids
                ],
                "draft": [
                    paragraph.to_dict()
                    for paragraph in paragraphs_by_section.get(section_id, [])
                ],
            }
        )
    raw = _call_json(
        client,
        FINAL_AUDIT_ADJUDICATION_SYSTEM_PROMPT,
        {"video_context": context, "items": adjudication_items},
        max_tokens=2500,
    )
    decisions = raw.get("decisions")
    if not isinstance(decisions, list):
        raise ValueError("final audit adjudication must return decisions")
    if not all(isinstance(decision, dict) for decision in decisions):
        raise ValueError("final audit adjudication decisions must be objects")
    indices = [decision.get("issue_index") for decision in decisions]
    if sorted(indices) != list(range(len(issues))) or len(indices) != len(set(indices)):
        raise ValueError("final audit adjudication must cover every issue exactly once")
    normalized = []
    for decision in sorted(decisions, key=lambda item: int(item["issue_index"])):
        if not isinstance(decision.get("valid"), bool):
            raise ValueError("final audit adjudication valid must be boolean")
        issue_index = int(decision["issue_index"])
        missing_details = decision.get("missing_details") or []
        draft_evidence = decision.get("draft_evidence") or []
        if not isinstance(missing_details, list) or not isinstance(draft_evidence, list):
            raise ValueError("final audit adjudication evidence fields must be lists")
        if (
            bool(decision["valid"])
            and str(issues[issue_index].get("kind") or "") == "missing"
            and not missing_details
        ):
            raise ValueError("valid missing issue must name truly missing details")
        normalized.append(
            {
                "issue_index": issue_index,
                "valid": bool(decision["valid"]),
                "missing_details": [str(value) for value in missing_details],
                "draft_evidence": [str(value) for value in draft_evidence],
                "reason": str(decision.get("reason") or "").strip(),
            }
        )
    return {
        "decisions": normalized,
        "valid_issue_indices": [
            decision["issue_index"] for decision in normalized if decision["valid"]
        ],
    }


def _proofread_asr_residue(
    sections: list[OutlineSection],
    units: list[InformationUnit],
    paragraphs_by_section: dict[str, list[Paragraph]],
    client: OpenAICompatibleClient,
    context: str,
) -> dict[str, Any]:
    by_id = _unit_map(units)
    payload = {
        "video_context": context,
        "sections": [
            {
                "section_id": section.id,
                "information_units": [by_id[unit_id].to_dict() for unit_id in section.unit_ids],
                "current_draft": [
                    {"paragraph_index": index, **paragraph.to_dict()}
                    for index, paragraph in enumerate(paragraphs_by_section[section.id])
                ],
            }
            for section in sections
        ],
    }
    expected_ids = [section.id for section in sections]
    error = ""
    for _attempt in range(2):
        request = dict(payload)
        if error:
            request["repair_required"] = error
        try:
            raw = _call_json(
                client,
                ASR_RESIDUE_PROOFREAD_SYSTEM_PROMPT,
                request,
                max_tokens=3000,
            )
            checked = raw.get("checked_section_ids")
            corrections = raw.get("corrections")
            unresolved = raw.get("unresolved")
            if not isinstance(checked, list) or sorted(map(str, checked)) != sorted(expected_ids):
                raise ValueError("ASR residue proofread must check every section")
            if not isinstance(corrections, list) or not isinstance(unresolved, list):
                raise ValueError("ASR residue proofread must return corrections and unresolved")
            if unresolved:
                raise ValueError("ASR residue proofread left unresolved fragments")
            normalized: list[dict[str, Any]] = []
            for correction in corrections:
                if not isinstance(correction, dict):
                    raise ValueError("ASR residue correction must be an object")
                section_id = str(correction.get("section_id") or "")
                if section_id not in expected_ids:
                    raise ValueError(f"unknown ASR residue section: {section_id}")
                paragraph_index = correction.get("paragraph_index")
                if not isinstance(paragraph_index, int):
                    raise ValueError("ASR residue paragraph_index must be an integer")
                paragraphs = paragraphs_by_section[section_id]
                if paragraph_index < 0 or paragraph_index >= len(paragraphs):
                    raise ValueError("ASR residue paragraph_index is out of range")
                original = str(correction.get("original") or "").strip()
                replacement = str(correction.get("replacement") or "").strip()
                paragraph = paragraphs[paragraph_index]
                occurrence_count = paragraph.text.count(original)
                if not original or not replacement or occurrence_count < 1:
                    raise ValueError("ASR residue correction must identify an existing fragment")
                section = next(item for item in sections if item.id == section_id)
                evidence = " ".join(
                    by_id[unit_id].text + " " + " ".join(by_id[unit_id].details)
                    for unit_id in section.unit_ids
                )
                introduced = [
                    anchor
                    for anchor in _anchors(replacement)
                    if not _anchor_supported(evidence + " " + context, anchor)
                ]
                if introduced:
                    raise ValueError(
                        f"ASR residue correction introduced exact details: {', '.join(introduced[:3])}"
                    )
                replace_all = correction.get("replace_all") is True
                occurrence = correction.get("occurrence")
                if occurrence_count > 1 and not replace_all:
                    if not isinstance(occurrence, int) or not 1 <= occurrence <= occurrence_count:
                        raise ValueError(
                            "repeated ASR residue requires replace_all or a valid occurrence"
                        )
                    pieces = paragraph.text.split(original)
                    paragraph.text = original.join(pieces[:occurrence]) + replacement + original.join(
                        pieces[occurrence:]
                    )
                    replaced_count = 1
                elif replace_all:
                    paragraph.text = paragraph.text.replace(original, replacement)
                    replaced_count = occurrence_count
                else:
                    paragraph.text = paragraph.text.replace(original, replacement, 1)
                    replaced_count = 1
                normalized.append(
                    {
                        "section_id": section_id,
                        "paragraph_index": paragraph_index,
                        "original": original,
                        "replacement": replacement,
                        "replaced_count": replaced_count,
                        "reason": str(correction.get("reason") or "").strip(),
                    }
                )
            return {
                "checked_section_ids": expected_ids,
                "corrections": normalized,
                "unresolved": [],
            }
        except Exception as exc:
            error = f"上次 ASR 残留审校无效：{exc}。重新检查每节并返回安全的最小修正。"
    raise RuntimeError(f"发布前 ASR 残留审校失败：{error}")


def _signature(segments: list[Segment], context: str) -> str:
    material = json.dumps(
        {
            "schema": "information-unit-outline-compose-v3",
            "pipeline_version": PIPELINE_VERSION,
            "context": context,
            "segments": [s.to_dict() for s in segments],
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def create_manuscript(
    segments: list[Segment],
    client: OpenAICompatibleClient | None,
    *,
    context: str = "",
    checkpoint_path: Path | None = None,
    publication_copyedit: bool = False,
) -> tuple[list[Paragraph], dict[str, Any]]:
    if not segments:
        raise RuntimeError("字幕为空，无法生成文字稿")
    if client is None:
        raise RuntimeError("缺少文本大模型配置，禁止发布未经编辑的原始 ASR 稿")

    signature = _signature(segments, context)
    checkpoint = load_json(checkpoint_path) if checkpoint_path else None
    if not isinstance(checkpoint, dict) or checkpoint.get("signature") != signature:
        checkpoint = {"signature": signature, "schema": "information-unit-outline-compose-v3"}

    units_payload = checkpoint.get("information_units")
    if isinstance(units_payload, list) and units_payload:
        units = [InformationUnit(**row) for row in units_payload]
    else:
        units = _extract_units(segments, client, context)
        checkpoint["information_units"] = [unit.to_dict() for unit in units]
        if checkpoint_path:
            atomic_json(checkpoint_path, checkpoint)

    corrections_payload = checkpoint.get("terminology_corrections")
    if isinstance(corrections_payload, list):
        corrections = corrections_payload
    else:
        corrections = _normalize_units(units, client, context)
        checkpoint["information_units"] = [unit.to_dict() for unit in units]
        checkpoint["terminology_corrections"] = corrections
        if checkpoint_path:
            atomic_json(checkpoint_path, checkpoint)

    outline_payload = checkpoint.get("outline")
    if isinstance(outline_payload, list) and outline_payload:
        sections = [OutlineSection(**row) for row in outline_payload]
    else:
        sections = _plan_outline(units, client, context)
        checkpoint["outline"] = [section.to_dict() for section in sections]
        if checkpoint_path:
            atomic_json(checkpoint_path, checkpoint)

    section_cache = checkpoint.get("sections") if isinstance(checkpoint.get("sections"), dict) else {}
    paragraphs_by_section: dict[str, list[Paragraph]] = {}
    titles = [section.title for section in sections]
    for index, section in enumerate(sections):
        cached = section_cache.get(section.id)
        if isinstance(cached, list) and cached:
            paragraphs = [Paragraph(**row) for row in cached]
        else:
            neighbours = titles[max(0, index - 1) : index] + titles[index + 1 : index + 2]
            paragraphs = _compose_one_section(section, units, client, context, neighbours)
            section_cache[section.id] = [paragraph.to_dict() for paragraph in paragraphs]
            checkpoint["sections"] = section_cache
            if checkpoint_path:
                atomic_json(checkpoint_path, checkpoint)
        paragraphs_by_section[section.id] = paragraphs

    if publication_copyedit:
        copyedit_cache = (
            checkpoint.get("publication_copyedit_sections")
            if isinstance(checkpoint.get("publication_copyedit_sections"), dict)
            else {}
        )
        for index, section in enumerate(sections):
            cached = copyedit_cache.get(section.id)
            if isinstance(cached, list) and cached:
                revised = [Paragraph(**row) for row in cached]
            else:
                neighbours = titles[max(0, index - 1) : index] + titles[index + 1 : index + 2]
                revised = _compose_one_section(
                    section,
                    units,
                    client,
                    context,
                    neighbours,
                    repair_instruction=PUBLICATION_COPYEDIT_INSTRUCTION,
                    current_draft=paragraphs_by_section[section.id],
                )
                copyedit_cache[section.id] = [paragraph.to_dict() for paragraph in revised]
                checkpoint["publication_copyedit_sections"] = copyedit_cache
                if checkpoint_path:
                    atomic_json(checkpoint_path, checkpoint)
            paragraphs_by_section[section.id] = revised
            section_cache[section.id] = [paragraph.to_dict() for paragraph in revised]
        checkpoint["sections"] = section_cache
        if checkpoint_path:
            atomic_json(checkpoint_path, checkpoint)

        # Give the fidelity/editorial audit a readable candidate.  Obvious ASR
        # residue is a sentence-level problem, not something the section
        # writer or final adjudicator should be asked to explain.  This pass is
        # repeated after every audit-driven rewrite and once more at release.
        pre_audit_residue = _proofread_asr_residue(
            sections,
            units,
            paragraphs_by_section,
            client,
            context,
        )
        checkpoint["pre_audit_asr_residue"] = pre_audit_residue
        section_cache = {
            section.id: [
                paragraph.to_dict()
                for paragraph in paragraphs_by_section[section.id]
            ]
            for section in sections
        }
        checkpoint["sections"] = section_cache
        if checkpoint_path:
            atomic_json(checkpoint_path, checkpoint)

    audit_history: list[dict[str, Any]] = []
    for repair_round in range(AUDIT_REPAIR_ATTEMPTS + 1):
        audit = _audit(sections, units, paragraphs_by_section, client, context)
        audit_history.append(audit)
        checkpoint["audit_history"] = audit_history
        if checkpoint_path:
            atomic_json(checkpoint_path, checkpoint)
        if audit["verdict"] == "pass" and not audit["issues"]:
            break
        if repair_round >= AUDIT_REPAIR_ATTEMPTS:
            try:
                adjudication = _adjudicate_final_audit(
                    sections,
                    units,
                    paragraphs_by_section,
                    audit["issues"],
                    client,
                    context,
                )
            except Exception as exc:
                audit["final_issue_adjudication"] = {
                    "status": "error",
                    "error": str(exc),
                }
                checkpoint["audit_history"] = audit_history
                if checkpoint_path:
                    atomic_json(checkpoint_path, checkpoint)
                raise RuntimeError("全稿忠实度终审裁决失败，已拒绝发布") from exc
            audit["final_issue_adjudication"] = adjudication
            if not adjudication["valid_issue_indices"]:
                audit["original_verdict"] = audit["verdict"]
                audit["original_issues"] = list(audit["issues"])
                audit["verdict"] = "pass"
                audit["issues"] = []
                audit["accepted_after_adjudication"] = True
                checkpoint["audit_history"] = audit_history
                if checkpoint_path:
                    atomic_json(checkpoint_path, checkpoint)
                break
            checkpoint["audit_history"] = audit_history
            if checkpoint_path:
                atomic_json(checkpoint_path, checkpoint)
            raise RuntimeError(
                "全稿忠实度终审仍有经独立裁决确认的问题，已拒绝发布"
            )
        issues_by_section: dict[str, list[dict[str, Any]]] = {}
        for issue in audit["issues"]:
            section_id = str(issue.get("section_id") or "")
            if section_id in {section.id for section in sections}:
                issues_by_section.setdefault(section_id, []).append(issue)
        if not issues_by_section:
            raise RuntimeError("全稿忠实度终审要求返工，但未返回可定位的问题")
        for index, section in enumerate(sections):
            if section.id not in issues_by_section:
                continue
            instruction = (
                "保留本节所有 keep information_units 的事实、例子、比较、数字和细节，仅重组或中性改写；"
                + "；".join(
                    str(issue.get("instruction") or issue.get("kind") or "修复忠实度")
                    for issue in issues_by_section[section.id]
                )
            )
            neighbours = titles[max(0, index - 1) : index] + titles[index + 1 : index + 2]
            paragraphs_by_section[section.id] = _compose_one_section(
                section,
                units,
                client,
                context,
                neighbours,
                repair_instruction=instruction,
                current_draft=paragraphs_by_section[section.id],
            )
            section_cache[section.id] = [
                paragraph.to_dict() for paragraph in paragraphs_by_section[section.id]
            ]
        checkpoint["sections"] = section_cache
        checkpoint["audit_history"] = audit_history
        if publication_copyedit:
            repair_residue = _proofread_asr_residue(
                sections,
                units,
                paragraphs_by_section,
                client,
                context,
            )
            checkpoint.setdefault("repair_asr_residue_history", []).append(
                repair_residue
            )
            section_cache = {
                section.id: [
                    paragraph.to_dict()
                    for paragraph in paragraphs_by_section[section.id]
                ]
                for section in sections
            }
            checkpoint["sections"] = section_cache
        if checkpoint_path:
            atomic_json(checkpoint_path, checkpoint)

    if publication_copyedit:
        residue_cache = checkpoint.get("asr_residue_sections")
        if isinstance(residue_cache, dict) and all(
            isinstance(residue_cache.get(section.id), list)
            and residue_cache.get(section.id)
            for section in sections
        ):
            for section in sections:
                paragraphs_by_section[section.id] = [
                    Paragraph(**row) for row in residue_cache[section.id]
                ]
        else:
            residue_audit = _proofread_asr_residue(
                sections,
                units,
                paragraphs_by_section,
                client,
                context,
            )
            residue_cache = {
                section.id: [
                    paragraph.to_dict()
                    for paragraph in paragraphs_by_section[section.id]
                ]
                for section in sections
            }
            checkpoint["asr_residue_audit"] = residue_audit
            checkpoint["asr_residue_sections"] = residue_cache
            checkpoint["sections"] = residue_cache
            if checkpoint_path:
                atomic_json(checkpoint_path, checkpoint)

    paragraphs = [paragraph for section in sections for paragraph in paragraphs_by_section[section.id]]
    kept_units = [unit for unit in units if unit.action == "keep"]
    represented_sources = {source_id for paragraph in paragraphs for source_id in paragraph.source_ids}
    expected_sources = {source_id for unit in kept_units for source_id in unit.source_ids}
    missing_sources = sorted(expected_sources - represented_sources)
    if missing_sources:
        raise RuntimeError(f"有意义信息单元覆盖失败：{', '.join(missing_sources[:8])}")
    editorial_diagnostics = manuscript_quality_report(sections, paragraphs_by_section)
    if editorial_diagnostics["status"] != "pass":
        raise RuntimeError(
            "成稿仍保留可确定识别的字幕式结构："
            + ", ".join(editorial_diagnostics["blockers"])
        )
    coverage = {
        "quality_status": "pass",
        "semantic_editing": True,
        "editing_architecture": "information_units_outline_sections_audit",
        "information_units": [unit.to_dict() for unit in units],
        "outline": [section.to_dict() for section in sections],
        "audit_history": audit_history,
        "pre_audit_asr_residue": checkpoint.get("pre_audit_asr_residue"),
        "repair_asr_residue_history": checkpoint.get(
            "repair_asr_residue_history", []
        ),
        "final_asr_residue_audit": checkpoint.get("asr_residue_audit"),
        "editorial_diagnostics": editorial_diagnostics,
        "terminology_corrections": corrections,
        "kept_unit_count": len(kept_units),
        "dropped_unit_count": len(units) - len(kept_units),
        "missing_ids": [],
        "warnings": [],
    }
    checkpoint["audit_history"] = audit_history
    checkpoint["completed"] = True
    if checkpoint_path:
        atomic_json(checkpoint_path, checkpoint)
    return paragraphs, coverage
