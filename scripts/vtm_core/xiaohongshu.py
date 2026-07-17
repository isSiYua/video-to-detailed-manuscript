"""Public Xiaohongshu / RedNote image-note adapter.

The ``window.__INITIAL_STATE__`` note mapping is adapted from
``JNHFlow21/social-post-extractor-mcp`` at commit
``72718caf3e7bb08a8bf4b990d074a53aec5bd5b9`` (Apache-2.0).  This version is
read-only, uses bounded standard-library networking, supports the current
RedNote domain, retains only ordered public text/images, and rejects video,
login-only, captcha, deleted, or risk-controlled notes.  See
``licenses/Apache-2.0.txt`` and ``references/third-party-notices.md``.
"""

from __future__ import annotations

import html
import json
import mimetypes
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import Frame, Segment
from .sources import SourceReference
from .web import (
    MAX_DOCUMENT_IMAGES,
    MAX_IMAGE_BYTES,
    DocumentImage,
    GenericWebSourceAdapter,
)


XHS_HOSTS = {
    "xiaohongshu.com",
    "www.xiaohongshu.com",
    "xhslink.com",
    "www.xhslink.com",
    "rednote.com",
    "www.rednote.com",
}
XHS_IMAGE_SUFFIXES = (
    ".xhscdn.com",
    ".rednotecdn.com",
    ".xhsimg.com",
    ".xiaohongshu.com",
    ".rednote.com",
)
XHS_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0.0.0 Safari/537.36"
)
MAX_NOTE_HTML_BYTES = 8 * 1024 * 1024
_INPUT_URL_RE = re.compile(
    r"(?:(?:https?://)?(?:www\.)?(?:xiaohongshu\.com|xhslink\.com|rednote\.com)/[^\s<>\"']+)",
    re.IGNORECASE,
)
_NOTE_ID_RE = re.compile(r"/(?:explore|search_result|discovery/item)/([0-9a-f]{24})(?:/|$)", re.I)
_INITIAL_STATE_RE = re.compile(r"window\.__INITIAL_STATE__=(.*?)</script>", re.DOTALL)


@dataclass(frozen=True, slots=True)
class XiaohongshuInfo:
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
    note_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _first_url(value: str) -> str:
    match = _INPUT_URL_RE.search(str(value or "").strip())
    if not match:
        raise ValueError("没有在小红书分享内容中找到支持的链接")
    result = match.group(0).rstrip(".,;!?)）】]，。；！")
    if "://" not in result:
        result = "https://" + result
    return result


def _platform_url(value: str) -> str:
    parsed = urllib.parse.urlparse(value)
    if parsed.scheme not in {"http", "https"} or (parsed.hostname or "").lower() not in XHS_HOSTS:
        raise ValueError("只支持公开 xiaohongshu.com、rednote.com 或 xhslink.com 链接")
    if parsed.username or parsed.password:
        raise ValueError("小红书链接不能包含用户名或密码")
    return value


def _image_url(value: Any) -> str:
    candidate = str(value or "").strip()
    if candidate.startswith("http://"):
        candidate = "https://" + candidate[7:]
    parsed = urllib.parse.urlparse(candidate)
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or not any(
        host == suffix[1:] or host.endswith(suffix) for suffix in XHS_IMAGE_SUFFIXES
    ):
        return ""
    return candidate


class _XhsRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> Any:
        _platform_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class _XhsImageRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> Any:
        if not _image_url(newurl):
            raise RuntimeError("小红书图片重定向到了不受信任的主机")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _opener() -> urllib.request.OpenerDirector:
    handlers: list[Any] = [_XhsRedirectHandler()]
    proxy = os.getenv("VTM_SOURCE_PROXY", "").strip()
    if proxy:
        handlers.append(urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
    return urllib.request.build_opener(*handlers)


def _image_opener() -> urllib.request.OpenerDirector:
    handlers: list[Any] = [_XhsImageRedirectHandler()]
    proxy = os.getenv("VTM_SOURCE_PROXY", "").strip()
    if proxy:
        handlers.append(urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
    return urllib.request.build_opener(*handlers)


def _canonical_direct(value: str) -> str:
    source = _platform_url(value)
    parsed = urllib.parse.urlparse(source)
    match = _NOTE_ID_RE.search(parsed.path)
    if not match:
        raise ValueError("小红书链接没有包含单篇笔记 ID")
    host = (parsed.hostname or "").lower()
    base = "https://www.rednote.com" if host.endswith("rednote.com") else "https://www.xiaohongshu.com"
    allowed_query = [
        (key, val)
        for key, val in urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
        if key in {"xsec_token", "xsec_source"}
    ]
    query = urllib.parse.urlencode(allowed_query)
    return urllib.parse.urlunparse(("https", urllib.parse.urlparse(base).netloc, f"/explore/{match.group(1)}", "", query, ""))


def _epoch_iso(value: Any) -> str:
    try:
        timestamp = int(value)
    except (TypeError, ValueError):
        return ""
    while timestamp > 9_999_999_999:
        timestamp //= 1000
    if timestamp <= 0:
        return ""
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def _html_title(raw_html: str) -> str:
    match = re.search(r"<title[^>]*>(.*?)</title>", raw_html, re.I | re.DOTALL)
    return html.unescape(re.sub(r"<[^>]+>", "", match.group(1))).strip() if match else ""


def parse_xhs_initial_state(raw_html: str, canonical_url: str) -> XiaohongshuInfo:
    """Convert one public initial-state note into ordered document evidence."""

    state_match = _INITIAL_STATE_RE.search(raw_html)
    if not state_match:
        raise RuntimeError("小红书公开页面中没有找到 __INITIAL_STATE__，可能需要登录或验证")
    state_blob = re.sub(r":undefined([,}])", r":null\1", state_match.group(1))
    try:
        state = json.loads(state_blob)
    except json.JSONDecodeError as exc:
        raise RuntimeError("小红书公开页面的 __INITIAL_STATE__ 无法解析") from exc

    note_map = ((state.get("note") or {}).get("noteDetailMap") or {}) if isinstance(state, dict) else {}
    requested_match = _NOTE_ID_RE.search(urllib.parse.urlparse(canonical_url).path)
    requested_id = requested_match.group(1) if requested_match else ""
    note: dict[str, Any] | None = None
    if isinstance(note_map, dict):
        preferred = note_map.get(requested_id)
        candidates = [preferred, *note_map.values()]
        for entry in candidates:
            candidate = entry.get("note") if isinstance(entry, dict) else None
            if isinstance(candidate, dict) and candidate.get("noteId"):
                note = candidate
                if str(candidate.get("noteId")) == requested_id:
                    break
    if note is None:
        raise RuntimeError("小红书没有返回公开笔记详情；内容可能已删除、需登录或受到风控")

    note_id = str(note.get("noteId") or requested_id)
    video = note.get("video") if isinstance(note.get("video"), dict) else {}
    if str(note.get("type") or "").lower() == "video" or video:
        raise RuntimeError("该链接是小红书视频笔记；当前阶段只处理公开图文笔记")

    title = str(note.get("title") or "").strip() or _html_title(raw_html) or f"小红书笔记 {note_id}"
    body = str(note.get("desc") or "").strip()
    text_blocks = [block.strip() for block in re.split(r"\n+", body) if block.strip()]
    tags = [
        str(item.get("name") or "").strip()
        for item in note.get("tagList") or []
        if isinstance(item, dict) and str(item.get("name") or "").strip()
    ]
    missing_tags = [tag for tag in tags if f"#{tag}" not in body and tag not in body]
    if missing_tags:
        text_blocks.append("话题：" + " ".join(f"#{tag}" for tag in missing_tags))
    if not text_blocks:
        text_blocks = [title]
    segments = [
        Segment(
            id=f"s{index:06d}",
            start=float(index - 1),
            end=float(index),
            text=block,
            locator_kind="document_order",
        )
        for index, block in enumerate(text_blocks, start=1)
    ]

    image_urls: list[str] = []
    seen: set[str] = set()
    for raw_image in note.get("imageList") or []:
        if not isinstance(raw_image, dict):
            continue
        candidate = _image_url(
            raw_image.get("urlDefault") or raw_image.get("urlPre") or raw_image.get("url")
        )
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        image_urls.append(candidate)
    images: list[DocumentImage] = []
    one_to_one = len(image_urls) == len(segments)
    for index, candidate in enumerate(image_urls):
        after_id = segments[index].id if one_to_one else segments[-1].id
        images.append(
            DocumentImage(
                url=candidate,
                after_segment_id=after_id,
                order=len(images) + 1,
            )
        )
    if not image_urls:
        raise RuntimeError("小红书图文笔记没有返回可用原图，可能需要登录或页面结构已变化")

    user = note.get("user") if isinstance(note.get("user"), dict) else {}
    author = str(user.get("nickname") or user.get("nickName") or user.get("name") or "").strip()
    parsed = urllib.parse.urlparse(canonical_url)
    base = "https://www.rednote.com" if (parsed.hostname or "").endswith("rednote.com") else "https://www.xiaohongshu.com"
    query = parsed.query
    final_url = f"{base}/explore/{note_id}" + (f"?{query}" if query else "")
    return XiaohongshuInfo(
        url=final_url,
        source_id=f"note-{note_id}",
        title=title,
        author=author,
        site_name="小红书 / RedNote",
        published_at=_epoch_iso(note.get("time")),
        extraction_engine="social-post-extractor-mcp-initial-state",
        segments=tuple(segment.to_dict() for segment in segments),
        images=tuple(image.to_dict() for image in images),
        content_type="image_note",
        note_id=note_id,
    )


class XiaohongshuSourceAdapter(GenericWebSourceAdapter):
    platform = "xiaohongshu"
    source_kind = "document"

    def can_handle(self, value: str) -> bool:
        try:
            _platform_url(_first_url(value))
            return True
        except ValueError:
            return False

    def _fetch_html(self, value: str) -> tuple[str, str]:
        request = urllib.request.Request(
            _platform_url(value),
            headers={"User-Agent": XHS_USER_AGENT, "Accept": "text/html,*/*"},
        )
        try:
            with _opener().open(request, timeout=30) as response:
                final_url = _platform_url(response.geturl())
                content_type = response.headers.get_content_type().lower()
                if content_type not in {"text/html", "application/xhtml+xml"}:
                    raise RuntimeError(f"小红书返回了不支持的内容类型：{content_type}")
                declared = int(response.headers.get("Content-Length") or 0)
                if declared > MAX_NOTE_HTML_BYTES:
                    raise RuntimeError("小红书页面超过安全大小限制")
                body = response.read(MAX_NOTE_HTML_BYTES + 1)
                if len(body) > MAX_NOTE_HTML_BYTES:
                    raise RuntimeError("小红书页面超过安全大小限制")
                charset = response.headers.get_content_charset() or "utf-8"
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError("连接小红书公开页面失败") from exc
        try:
            return body.decode(charset, errors="replace"), final_url
        except LookupError:
            return body.decode("utf-8", errors="replace"), final_url

    def normalize_input_url(self, value: str) -> str:
        source = _platform_url(_first_url(value))
        try:
            return _canonical_direct(source)
        except ValueError:
            _html, resolved = self._fetch_html(source)
            return _canonical_direct(resolved)

    def canonicalize_input(self, value: str) -> str:
        return self.normalize_input_url(value)

    def source_id_from_url(self, value: str) -> str:
        match = _NOTE_ID_RE.search(urllib.parse.urlparse(self.canonicalize_input(value)).path)
        if not match:
            raise ValueError("小红书链接没有包含单篇笔记 ID")
        return f"note-{match.group(1)}"

    def selector_from_url(self, value: str, explicit: int | None = None) -> None:
        del value, explicit
        return None

    def inspect(self, value: str, selector: int | None = None) -> XiaohongshuInfo:
        del selector
        canonical = self.canonicalize_input(value)
        raw_html, final_url = self._fetch_html(canonical)
        if "/404" in urllib.parse.urlparse(final_url).path:
            raise RuntimeError("小红书笔记当前无法公开浏览，可能已删除、需登录或受到风控")
        return parse_xhs_initial_state(raw_html, canonical)

    def reference(self, inspected: XiaohongshuInfo) -> SourceReference:
        return SourceReference(
            platform=self.platform,
            source_kind=self.source_kind,
            source_id=inspected.source_id,
            canonical_url=inspected.url,
            title=inspected.title,
            author=inspected.author,
        )

    def restore_info(self, metadata: dict[str, Any]) -> XiaohongshuInfo:
        field_names = XiaohongshuInfo.__dataclass_fields__
        payload = {key: metadata[key] for key in field_names}
        payload["segments"] = tuple(payload["segments"])
        payload["images"] = tuple(payload["images"])
        return XiaohongshuInfo(**payload)

    def content_segments(self, inspected: XiaohongshuInfo) -> tuple[list[Segment], dict[str, Any]]:
        segments = [Segment(**item) for item in inspected.segments]
        return segments, {
            "source": "xiaohongshu_initial_state",
            "locator_kind": "document_order",
            "segment_count": len(segments),
            "image_count": len(inspected.images),
            "extraction_engine": inspected.extraction_engine,
            "content_type": inspected.content_type,
        }

    def download_images(
        self, inspected: XiaohongshuInfo, assets_dir: Path, *, limit: int = 60
    ) -> list[Frame]:
        assets_dir.mkdir(parents=True, exist_ok=True)
        frames: list[Frame] = []
        segment_order = {str(item["id"]): float(item["end"]) for item in inspected.segments}
        retained_limit = min(MAX_DOCUMENT_IMAGES, max(0, limit))
        for image in inspected.images:
            if len(frames) >= retained_limit:
                break
            source_url = _image_url(image.get("url"))
            if not source_url:
                continue
            request = urllib.request.Request(
                source_url,
                headers={"User-Agent": XHS_USER_AGENT, "Referer": inspected.url},
            )
            try:
                with _image_opener().open(request, timeout=30) as response:
                    final_url = _image_url(response.geturl())
                    if not final_url:
                        raise RuntimeError("小红书图片返回了不受信任的主机")
                    content_type = response.headers.get_content_type().lower()
                    if not content_type.startswith("image/"):
                        raise RuntimeError("小红书原图返回了非图片内容")
                    declared = int(response.headers.get("Content-Length") or 0)
                    if declared > MAX_IMAGE_BYTES:
                        raise RuntimeError("小红书单张原图超过安全大小限制")
                    body = response.read(MAX_IMAGE_BYTES + 1)
                    if len(body) > MAX_IMAGE_BYTES:
                        raise RuntimeError("小红书单张原图超过安全大小限制")
            except Exception:
                continue
            suffix = mimetypes.guess_extension(content_type or "") or ".jpg"
            if suffix == ".jpe":
                suffix = ".jpg"
            if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".gif"}:
                continue
            path = assets_dir / f"source-{len(frames) + 1:03d}{suffix}"
            path.write_bytes(body)
            after_id = str(image.get("after_segment_id") or "")
            frames.append(
                Frame(
                    timestamp=segment_order.get(after_id, float(image.get("order") or len(frames) + 1)),
                    path=str(path),
                    source_ids=[after_id] if after_id else [],
                    content_kind="source_image",
                    keep_image=True,
                    media_kind="source_image",
                    locator_label=f"原文第 {int(image.get('order') or len(frames) + 1)} 张图片",
                    source_url=final_url,
                )
            )
        return frames

    def context(self, inspected: XiaohongshuInfo) -> str:
        creator = f"；作者：{inspected.author}" if inspected.author else ""
        published = f"；发布时间：{inspected.published_at}" if inspected.published_at else ""
        return f"来源类型：小红书图文笔记；标题：{inspected.title}{creator}{published}"

    def folder_marker(self, inspected: XiaohongshuInfo) -> str:
        return f"XHS-{inspected.note_id}"
