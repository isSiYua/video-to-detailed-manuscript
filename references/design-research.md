# Design research

This file records the upstream behaviors that shaped the acquisition order. It is background for maintainers, not a license to circumvent platform controls.

## BiliNote

BiliNote uses Bilibili subtitle-first handling: its browser extension can use the user's login state to fetch subtitles and skip audio transcription, while its backend subtitle fetcher uses the player API as a fallback. This Skill follows the same architectural decision but keeps an independent, smaller implementation.

Source: https://github.com/JefferyHcool/BiliNote

The current BiliNote implementation also contributes request-size-aware chunking, transient-provider retry with exponential backoff, timestamped screenshot markers, and evidence caching. This Skill retains an independent CLI and evidence audit instead of importing BiliNote's web application or deployment stack.

## VideoLingo

VideoLingo separates semantic preparation from expressive rewriting and supplies previous/next context plus shared terminology to bounded batches. Version 6 uses the same separation more explicitly: bounded batches produce chronological information units, the complete unit set produces an article outline, and only then does DeepSeek compose section prose. Neighbouring transcript chunks remain context-only and cannot become current-chunk evidence.

Source: https://github.com/Huanshere/VideoLingo

## summarize

summarize interleaves stable slide markers with timestamped transcript text, validates marker order, uses scene detection plus interval fallback, and rejects weak OCR before it becomes prose. This Skill keeps its own paragraph/time mapping but adopts the same deterministic evidence-placement and OCR-garbage filtering principles. Its AI article planner supplies transcript-grounded visual time windows; every locally distinct slide/scene inside those windows can be reviewed, so density follows the video's information changes rather than a fixed minutes-per-frame rule. A selected image remains an auditable frame even when OCR or visual classification fails.

Source: https://github.com/steipete/summarize

## yt-dlp

yt-dlp's Bilibili extractor retrieves formats and subtitles through Bilibili player APIs. It remains a useful compatibility fallback, but a webpage/risk-control error on that path does not prove the player media endpoint is unavailable.

Source: https://github.com/yt-dlp/yt-dlp/blob/master/yt_dlp/extractor/bilibili.py

## Bilibili AI conclusion

The community API collection documents an authenticated, WBI-signed AI-conclusion response that may include `part_subtitle` entries with start/end timestamps. The Skill tries this only when the user configured a login cookie, and treats failure as non-fatal.

Source: https://github.com/pskdje/bilibili-API-collect/blob/main/docs/video/summary.md

## FunASR

FunASR documents CPU-capable Chinese models and combines Paraformer with FSMN-VAD and CT-Punctuation. The Skill uses that three-model pipeline because usable sentence timing is required for coverage checks and image alignment; a single multi-minute SRT block is explicitly rejected. Models are prepared during deployment, never inside a Feishu video job.

Source: https://github.com/modelscope/FunASR

## Hugging Face fallback

Hugging Face documents `HF_HUB_DISABLE_XET` and offline/cache environment controls. They can help with an already chosen faster-whisper deployment, but they do not make Hugging Face the preferred model source for this mainland ECS workflow.

Source: https://github.com/huggingface/huggingface_hub/blob/main/docs/source/en/package_reference/environment_variables.md
