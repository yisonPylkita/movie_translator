"""Hardsub-OCR Polish-subtitle proof-of-concept.

Standalone PoC (NOT wired into the Rust pipeline). It resolves an
ogladajanime.pl episode's player embed URLs, downloads a low-res copy of a
PL-hardsubbed stream, and OCRs the baked-in Polish subtitles into an .srt.

Resolving the embed URLs cannot be done headlessly: the site is gated by
Cloudflare Turnstile and its anti-debug bounces any DevTools/CDP-driven
browser. So URL resolution happens in a real browser via either
`ogladajanime_resolver.user.js` (Tampermonkey) or a mitmproxy capture read
by `flow_extract.py`; both emit a players JSON that `__main__` consumes.

Design: docs/superpowers/specs/2026-06-03-hardsub-ocr-poc-design.md
"""
