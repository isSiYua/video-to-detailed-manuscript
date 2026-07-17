from __future__ import annotations

import asyncio
import html
import os
import re
import urllib.parse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

try:
    from zhihu_cli import client as zhihu_client
    from zhihu_cli.auth import ZhihuCredential
    from zhihu_cli.exceptions import (
        AuthenticationError,
        NetworkError,
        NotFoundError,
        RateLimitError,
        ZhihuError,
    )
except ImportError:  # The adapter remains discoverable for diagnostics.
    zhihu_client = None
    ZhihuCredential = None

    class ZhihuError(Exception):
        """Fallback types keep fixture tests independent from optional dependencies."""

    class AuthenticationError(ZhihuError):
        pass

    class NetworkError(ZhihuError):
        pass

    class NotFoundError(ZhihuError):
        pass

    class RateLimitError(ZhihuError):
        pass

from .models import Segment
from .sources import SourceReference
from .web import GenericWebSourceAdapter, parse_html_document


@dataclass(frozen=True, slots=True)
class ZhihuInfo:
    url: str
    source_id: str
    title: str
    author: str
    site_name: str
    published_at: str
    extraction_engine: str
    segments: tuple[dict[str, Any], ...]
    images: tuple[dict[str, Any], ...]
    content_type: str
    content_id: str
    question_id: str
    access_mode: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_ANSWER_WITH_QUESTION_RE = re.compile(r"/question/(\d+)/answer/(\d+)(?:/|$)")
_ANSWER_RE = re.compile(r"/(?:en/)?answer/(\d+)(?:/|$)")
_ARTICLE_RE = re.compile(r"/(?:en/)?(?:p|article)/(\d+)(?:/|$)")


def _target(value: str) -> tuple[str, str, str] | None:
    text = str(value or "").strip()
    if "://" not in text:
        text = "https://" + text
    parsed = urllib.parse.urlparse(text)
    host = (parsed.hostname or "").lower()
    if host not in {"zhihu.com", "www.zhihu.com", "zhuanlan.zhihu.com"}:
        return None
    answer = _ANSWER_WITH_QUESTION_RE.search(parsed.path)
    if answer:
        return "answer", answer.group(2), answer.group(1)
    answer = _ANSWER_RE.search(parsed.path)
    if answer:
        return "answer", answer.group(1), ""
    article = _ARTICLE_RE.search(parsed.path)
    if article:
        return "article", article.group(1), ""
    return None


def _published_at(value: Any) -> str:
    try:
        seconds = int(value)
    except (TypeError, ValueError):
        return ""
    if seconds <= 0:
        return ""
    return datetime.fromtimestamp(seconds, tz=timezone.utc).isoformat()


class ZhihuSourceAdapter(GenericWebSourceAdapter):
    """Zhihu answer/article adapter backed by the Apache-2.0 zhihu-tui client."""

    platform = "zhihu"
    source_kind = "document"

    def can_handle(self, value: str) -> bool:
        return _target(value) is not None

    def normalize_input_url(self, value: str) -> str:
        target = _target(value)
        if target is None:
            raise ValueError("只支持知乎回答或专栏文章 URL")
        content_type, content_id, question_id = target
        if content_type == "article":
            return f"https://zhuanlan.zhihu.com/p/{content_id}"
        if question_id:
            return f"https://www.zhihu.com/question/{question_id}/answer/{content_id}"
        return f"https://www.zhihu.com/answer/{content_id}"

    def canonicalize_input(self, value: str) -> str:
        return self.normalize_input_url(value)

    def source_id_from_url(self, value: str) -> str:
        target = _target(value)
        if target is None:
            raise ValueError("只支持知乎回答或专栏文章 URL")
        content_type, content_id, _question_id = target
        return f"{content_type}-{content_id}"

    def selector_from_url(self, value: str, explicit: int | None = None) -> None:
        del value, explicit
        return None

    def inspect(self, value: str, selector: int | None = None) -> ZhihuInfo:
        del selector
        target = _target(value)
        if target is None:
            raise ValueError("只支持知乎回答或专栏文章 URL")
        if zhihu_client is None or ZhihuCredential is None:
            raise RuntimeError("缺少 zhihu-tui 依赖，请重新运行项目安装器")

        content_type, content_id, input_question_id = target
        z_c0 = os.getenv("ZHIHU_Z_C0", "").strip()
        credential = ZhihuCredential(z_c0=z_c0) if z_c0 else None
        try:
            if content_type == "answer":
                raw = asyncio.run(
                    zhihu_client.get_answer(int(content_id), credential=credential)
                )
            else:
                raw = asyncio.run(
                    zhihu_client.get_article(int(content_id), credential=credential)
                )
        except AuthenticationError as exc:
            if not z_c0:
                raise RuntimeError(
                    "知乎拒绝了无登录读取。请通过 SSH 运行 "
                    "scripts/vtm configure secret zhihu_z_c0，隐藏录入你自己的 z_c0 后重试"
                ) from exc
            raise RuntimeError(
                "知乎授权认证失败；请确认 z_c0 未过期且属于你自己的账号"
            ) from exc
        except RateLimitError as exc:
            raise RuntimeError("知乎触发了访问频率限制；请稍后重试，避免连续批量抓取") from exc
        except NotFoundError as exc:
            raise RuntimeError("知乎回答或文章不存在、已删除，或当前账号无权访问") from exc
        except NetworkError as exc:
            raise RuntimeError("连接知乎失败；请检查当前机器到知乎的网络后重试") from exc
        except ZhihuError as exc:
            raise RuntimeError(f"知乎上游读取失败：{type(exc).__name__}") from exc
        except Exception as exc:
            raise RuntimeError(f"知乎适配器意外失败：{type(exc).__name__}") from exc

        if not isinstance(raw, dict):
            raise RuntimeError("知乎上游没有返回结构化内容")
        author_data = raw.get("author") if isinstance(raw.get("author"), dict) else {}
        author = str(author_data.get("name") or "").strip()
        content = str(raw.get("content") or "").strip()
        if not content:
            raise RuntimeError("知乎内容为空，可能已删除、仅限登录、付费或受账号权限限制")

        if content_type == "answer":
            question = raw.get("question") if isinstance(raw.get("question"), dict) else {}
            question_id = str(question.get("id") or input_question_id or "")
            title = str(question.get("title") or raw.get("excerpt") or "知乎回答").strip()
            canonical = (
                f"https://www.zhihu.com/question/{question_id}/answer/{content_id}"
                if question_id
                else f"https://www.zhihu.com/answer/{content_id}"
            )
            published_at = _published_at(raw.get("created_time"))
        else:
            question_id = ""
            title = str(raw.get("title") or "知乎文章").strip()
            canonical = f"https://zhuanlan.zhihu.com/p/{content_id}"
            published_at = _published_at(raw.get("created"))

        escaped_title = html.escape(title, quote=True)
        escaped_author = html.escape(author, quote=True)
        escaped_published = html.escape(published_at, quote=True)
        source_html = (
            "<html><head>"
            f'<meta property="og:title" content="{escaped_title}">'
            f'<meta name="author" content="{escaped_author}">'
            f'<meta property="article:published_time" content="{escaped_published}">'
            '<meta property="og:site_name" content="知乎">'
            "</head><body>"
            f'<article class="zhihu-{content_type}-content">{content}</article>'
            "</body></html>"
        )
        metadata, segments, images = parse_html_document(
            source_html, canonical, minimum_text_chars=10
        )
        return ZhihuInfo(
            url=canonical,
            source_id=f"{content_type}-{content_id}",
            title=title or metadata["title"],
            author=author or metadata["author"],
            site_name="知乎",
            published_at=published_at or metadata["published_at"],
            extraction_engine=f"zhihu-tui+{metadata['extraction_engine']}",
            segments=tuple(segment.to_dict() for segment in segments),
            images=tuple(image.to_dict() for image in images),
            content_type=content_type,
            content_id=content_id,
            question_id=question_id,
            access_mode="authorized_session" if z_c0 else "public",
        )

    def reference(self, inspected: ZhihuInfo) -> SourceReference:
        return SourceReference(
            platform=self.platform,
            source_kind=self.source_kind,
            source_id=inspected.source_id,
            canonical_url=inspected.url,
            title=inspected.title,
            author=inspected.author,
        )

    def restore_info(self, metadata: dict[str, Any]) -> ZhihuInfo:
        fields = ZhihuInfo.__dataclass_fields__
        payload = {key: metadata[key] for key in fields}
        payload["segments"] = tuple(payload["segments"])
        payload["images"] = tuple(payload["images"])
        return ZhihuInfo(**payload)

    def content_segments(self, inspected: ZhihuInfo) -> tuple[list[Segment], dict[str, Any]]:
        segments = [Segment(**item) for item in inspected.segments]
        return segments, {
            "source": "zhihu_structured_document",
            "locator_kind": "document_order",
            "segment_count": len(segments),
            "image_count": len(inspected.images),
            "extraction_engine": inspected.extraction_engine,
            "content_type": inspected.content_type,
            "access_mode": inspected.access_mode,
        }

    def context(self, inspected: ZhihuInfo) -> str:
        kind = "回答" if inspected.content_type == "answer" else "文章"
        creator = f"；作者：{inspected.author}" if inspected.author else ""
        published = f"；发布时间：{inspected.published_at}" if inspected.published_at else ""
        return f"来源类型：知乎{kind}；标题：{inspected.title}{creator}{published}"

    def folder_marker(self, inspected: ZhihuInfo) -> str:
        prefix = "A" if inspected.content_type == "answer" else "P"
        return f"ZH-{prefix}-{inspected.content_id}"
