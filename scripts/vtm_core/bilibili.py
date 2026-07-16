from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import Segment

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
ALLOWED_HOSTS = {"bilibili.com", "www.bilibili.com", "m.bilibili.com", "b23.tv", "www.b23.tv"}
API_HOSTS = {"api.bilibili.com"}
SUBTITLE_HOST_SUFFIXES = (".bilibili.com", ".hdslb.com", ".bilivideo.com")
MEDIA_HOST_SUFFIXES = (".bilivideo.com", ".hdslb.com")
MIXIN_KEY_ENC_TAB = (
    46, 47, 18, 2, 53, 8, 23, 32, 15, 50, 10, 31, 58, 3, 45, 35,
    27, 43, 5, 49, 33, 9, 42, 19, 29, 28, 14, 39, 12, 38, 41, 13,
    37, 48, 7, 16, 24, 55, 40, 61, 26, 17, 0, 1, 60, 51, 30, 4,
    22, 25, 54, 21, 56, 59, 6, 63, 57, 62, 11, 36, 20, 34, 44, 52,
)


@dataclass(slots=True)
class VideoInfo:
    url: str
    bvid: str
    cid: int
    part: int
    title: str
    part_title: str
    duration: float
    owner: str
    cover: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "bvid": self.bvid,
            "cid": self.cid,
            "part": self.part,
            "title": self.title,
            "part_title": self.part_title,
            "duration": self.duration,
            "owner": self.owner,
            "cover": self.cover,
        }


class BilibiliClient:
    """Minimal subtitle-first Bilibili client adapted from BiliNote's MIT flow."""

    def __init__(self, cookie: str | None = None, timeout: float = 20.0):
        self.cookie = cookie if cookie is not None else os.getenv("BILIBILI_COOKIE", "")
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        headers = {"User-Agent": UA, "Referer": "https://www.bilibili.com/"}
        if self.cookie:
            headers["Cookie"] = self.cookie
        return headers

    @staticmethod
    def _validate_url(url: str) -> None:
        parsed = urllib.parse.urlparse(url)
        host = (parsed.hostname or "").lower()
        if parsed.scheme not in {"http", "https"} or host not in ALLOWED_HOSTS:
            raise ValueError("Only bilibili.com and b23.tv HTTP(S) links are accepted")

    @staticmethod
    def normalize_input_url(value: str) -> str:
        """Accept full links as well as bare BV/av identifiers."""
        text = str(value or "").strip()
        if re.fullmatch(r"BV[0-9A-Za-z]{10}", text, flags=re.I):
            return f"https://www.bilibili.com/video/{text}/"
        if re.fullmatch(r"av\d+", text, flags=re.I):
            return f"https://www.bilibili.com/video/{text}/"
        return text

    def _request(self, url: str, params: dict[str, Any] | None = None) -> Any:
        parsed_input = urllib.parse.urlparse(url)
        if parsed_input.scheme != "https" or (parsed_input.hostname or "").lower() not in API_HOSTS:
            raise ValueError("Rejected unexpected Bilibili API host")
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"
        request = urllib.request.Request(url, headers=self._headers())
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                final = response.geturl()
                parsed_final = urllib.parse.urlparse(final)
                if (parsed_final.hostname or "").lower() not in API_HOSTS:
                    raise RuntimeError("Bilibili API redirected to an unexpected host")
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Bilibili request failed: {type(exc).__name__}") from exc

    def resolve(self, url: str) -> str:
        url = self.normalize_input_url(url)
        self._validate_url(url)
        if "b23.tv" not in (urllib.parse.urlparse(url).hostname or ""):
            return url
        # Short-link resolution never needs the authenticated cookie. Omitting
        # it prevents credential forwarding across redirect hops.
        request = urllib.request.Request(
            url,
            headers={"User-Agent": UA, "Referer": "https://www.bilibili.com/"},
            method="HEAD",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                final = response.geturl()
        except urllib.error.HTTPError as exc:
            final = exc.geturl()
        self._validate_url(final)
        return final

    @staticmethod
    def extract_bvid(url: str) -> str:
        match = re.search(r"(?:video/)?(BV[0-9A-Za-z]{10})", url, flags=re.I)
        if not match:
            raise ValueError("Could not find a BV id in the Bilibili URL")
        return "BV" + match.group(1)[2:]

    @staticmethod
    def extract_part(url: str, explicit: int | None = None) -> int:
        if explicit is not None:
            return explicit
        query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        try:
            return max(1, int(query.get("p", [1])[0]))
        except (TypeError, ValueError):
            return 1

    def inspect(self, url: str, part: int | None = None) -> VideoInfo:
        resolved = self.resolve(url)
        try:
            requested_id: dict[str, Any] = {"bvid": self.extract_bvid(resolved)}
        except ValueError:
            av_match = re.search(r"(?:video/)?av(\d+)", resolved, flags=re.I)
            if not av_match:
                raise ValueError("Could not find a BV or av id in the Bilibili URL")
            requested_id = {"aid": int(av_match.group(1))}
        selected_part = self.extract_part(resolved, part)
        payload = self._request(
            "https://api.bilibili.com/x/web-interface/view", requested_id
        )
        if payload.get("code") != 0:
            raise RuntimeError(f"Bilibili metadata error: {payload.get('message', payload.get('code'))}")
        data = payload.get("data") or {}
        bvid = str(data.get("bvid") or requested_id.get("bvid") or "")
        pages = data.get("pages") or []
        if selected_part < 1 or selected_part > max(1, len(pages)):
            raise ValueError(f"Part {selected_part} is out of range (1..{max(1, len(pages))})")
        page = pages[selected_part - 1] if pages else data
        cid = int(page.get("cid") or data.get("cid") or 0)
        if cid <= 0:
            raise RuntimeError("Bilibili metadata did not contain a usable cid")
        return VideoInfo(
            url=resolved,
            bvid=bvid,
            cid=cid,
            part=selected_part,
            title=str(data.get("title") or bvid),
            part_title=str(page.get("part") or ""),
            duration=float(page.get("duration") or data.get("duration") or 0),
            owner=str((data.get("owner") or {}).get("name") or ""),
            cover=str(data.get("pic") or ""),
        )

    def subtitles(self, info: VideoInfo) -> tuple[list[Segment], dict[str, Any]]:
        payload = self._request(
            "https://api.bilibili.com/x/player/wbi/v2",
            {"bvid": info.bvid, "cid": info.cid},
        )
        if payload.get("code") != 0:
            return [], {"warning": payload.get("message", "player API error")}
        tracks = (((payload.get("data") or {}).get("subtitle") or {}).get("subtitles") or [])
        if not tracks:
            return [], {"warning": "No native subtitle tracks returned"}

        def score(track: dict[str, Any]) -> tuple[int, int]:
            lan = str(track.get("lan") or "").lower()
            chinese = lan.startswith("zh") or lan == "ai-zh"
            manual = not bool(track.get("ai_type"))
            return (2 if chinese else 0, 1 if manual else 0)

        track = sorted(tracks, key=score, reverse=True)[0]
        subtitle_url = str(track.get("subtitle_url") or "")
        if not subtitle_url:
            return [], {"warning": "Subtitle track has no URL; login cookie may be required"}
        if subtitle_url.startswith("//"):
            subtitle_url = "https:" + subtitle_url
        parsed = urllib.parse.urlparse(subtitle_url)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or not parsed.hostname.lower().endswith(SUBTITLE_HOST_SUFFIXES)
        ):
            raise RuntimeError("Rejected unexpected subtitle host")
        # The signed subtitle URL is sufficient; do not disclose the account
        # cookie to a CDN host.
        request = urllib.request.Request(
            subtitle_url,
            headers={"User-Agent": UA, "Referer": f"https://www.bilibili.com/video/{info.bvid}/"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            body = json.loads(response.read().decode("utf-8")).get("body") or []
        segments = [
            Segment(
                id=f"s{index:06d}",
                start=float(item.get("from") or 0),
                end=float(item.get("to") or 0),
                text=str(item.get("content") or "").strip(),
            )
            for index, item in enumerate(body, start=1)
            if str(item.get("content") or "").strip()
        ]
        return segments, {
            "source": "bilibili_subtitle",
            "language": track.get("lan"),
            "ai_type": track.get("ai_type"),
            "track_id": track.get("id"),
        }

    @staticmethod
    def _wbi_sign(params: dict[str, Any], img_key: str, sub_key: str) -> dict[str, Any]:
        raw = img_key + sub_key
        mixin_key = "".join(raw[index] for index in MIXIN_KEY_ENC_TAB if index < len(raw))[:32]
        signed = {**params, "wts": int(time.time())}
        signed = {
            key: re.sub(r"[!'()*]", "", str(value))
            for key, value in signed.items()
        }
        query = urllib.parse.urlencode(sorted(signed.items()))
        signed["w_rid"] = hashlib.md5((query + mixin_key).encode()).hexdigest()
        return signed

    def _wbi_keys(self) -> tuple[str, str]:
        payload = self._request("https://api.bilibili.com/x/web-interface/nav")
        data = payload.get("data") or {}
        wbi = data.get("wbi_img") or {}

        def filename(value: str) -> str:
            return Path(urllib.parse.urlparse(value).path).stem

        img_key = filename(str(wbi.get("img_url") or ""))
        sub_key = filename(str(wbi.get("sub_url") or ""))
        if not img_key or not sub_key:
            raise RuntimeError("Bilibili did not return WBI signing keys")
        return img_key, sub_key

    def ai_transcript(self, info: VideoInfo) -> tuple[list[Segment], dict[str, Any]]:
        """Use Bilibili's optional AI transcript when an authenticated account has access."""
        if not self.cookie:
            return [], {"warning": "Bilibili AI transcript requires a login cookie"}
        try:
            img_key, sub_key = self._wbi_keys()
            params = self._wbi_sign(
                {"bvid": info.bvid, "cid": info.cid}, img_key, sub_key
            )
            payload = self._request(
                "https://api.bilibili.com/x/web-interface/view/conclusion/get",
                params,
            )
            data = payload.get("data") or {}
            model_result = data.get("model_result") or {}
            groups = model_result.get("subtitle") or []
            body = [
                item
                for group in groups
                for item in (group.get("part_subtitle") or [])
            ]
            segments = [
                Segment(
                    id=f"s{index:06d}",
                    start=float(item.get("start_timestamp") or 0),
                    end=float(item.get("end_timestamp") or 0),
                    text=str(item.get("content") or "").strip(),
                )
                for index, item in enumerate(body, start=1)
                if str(item.get("content") or "").strip()
            ]
            if not segments:
                return [], {"warning": "No Bilibili AI transcript is available"}
            return segments, {
                "source": "bilibili_ai_conclusion",
                "stid": data.get("stid"),
                "result_type": model_result.get("result_type"),
            }
        except Exception as exc:
            return [], {"warning": f"Bilibili AI transcript unavailable ({type(exc).__name__})"}

    @staticmethod
    def _validate_media_url(url: str) -> str:
        parsed = urllib.parse.urlparse(url)
        host = (parsed.hostname or "").lower()
        if (
            parsed.scheme != "https"
            or not host
            or not host.endswith(MEDIA_HOST_SUFFIXES)
        ):
            raise RuntimeError("Rejected unexpected Bilibili media host")
        return url

    def media_stream(
        self,
        info: VideoInfo,
        *,
        audio_only: bool,
        max_height: int = 720,
    ) -> dict[str, Any]:
        """Select a public player stream without downloading the video webpage.

        This is a bounded fallback for cases where yt-dlp's webpage path is rate
        limited. It uses the same public player API family as Bilibili clients and
        still respects login/quality restrictions; it is not an anti-bot bypass.
        """
        # Bilibili's qn value controls which DASH representations are returned.
        # Ask for the target quality, then still enforce max_height below.
        if max_height >= 2160:
            requested_qn = 120
        elif max_height >= 1080:
            requested_qn = 80
        elif max_height >= 720:
            requested_qn = 64
        elif max_height >= 480:
            requested_qn = 32
        else:
            requested_qn = 16
        payload = self._request(
            "https://api.bilibili.com/x/player/playurl",
            {
                "bvid": info.bvid,
                "cid": info.cid,
                "qn": requested_qn,
                "fnver": 0,
                "fnval": 16,
                "fourk": 0,
            },
        )
        if payload.get("code") != 0:
            raise RuntimeError(
                f"Bilibili player API error: {payload.get('message', payload.get('code'))}"
            )
        data = payload.get("data") or {}
        dash = data.get("dash") or {}
        if audio_only and dash.get("audio"):
            tracks = sorted(
                dash["audio"], key=lambda item: int(item.get("bandwidth") or 0), reverse=True
            )
            track = tracks[0]
            url = track.get("baseUrl") or track.get("base_url")
            return {
                "url": self._validate_media_url(str(url or "")),
                "kind": "audio",
                "container": "m4s",
                "quality": track.get("id"),
            }
        if not audio_only and dash.get("video"):
            all_tracks = list(dash["video"])
            tracks = [
                item for item in dash["video"] if int(item.get("height") or 0) <= max_height
            ]
            if not tracks:
                minimum = min(all_tracks, key=lambda item: int(item.get("height") or 99999))
                raise RuntimeError(
                    "Bilibili returned no video stream within the requested height "
                    f"(minimum available: {int(minimum.get('height') or 0)}p)"
                )
            track = sorted(
                tracks,
                key=lambda item: (
                    int(item.get("height") or 0),
                    int(item.get("bandwidth") or 0),
                ),
                reverse=True,
            )[0]
            url = track.get("baseUrl") or track.get("base_url")
            return {
                "url": self._validate_media_url(str(url or "")),
                "kind": "video",
                "container": "m4s",
                "quality": track.get("id"),
                "height": track.get("height"),
            }

        # Older/public responses may expose a progressive stream instead of DASH.
        progressive = data.get("durl") or []
        if progressive:
            url = progressive[0].get("url")
            return {
                "url": self._validate_media_url(str(url or "")),
                "kind": "combined",
                "container": "flv",
                "quality": data.get("quality"),
            }
        raise RuntimeError("Bilibili player API returned no usable media stream")
