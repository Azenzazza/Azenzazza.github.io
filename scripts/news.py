"""
毎日1回、技術ニュースと一般ニュースを生成する。
- news/tech.html   : Zenn / Qiita / Hacker News から
- news/normal.html : Google News（日本＋国際）から
アーカイブなし（毎日上書き）。

必要な環境変数:
  GROQ_API_KEY - Groq APIキー
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import feedparser
import requests
from openai import OpenAI

# ============================================================
# 設定
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_PATH = ROOT / "scripts" / "news_template.html"
NEWS_DIR = ROOT / "news"

GROQ_MODEL_PLAN = "openai/gpt-oss-20b"
GROQ_MODEL_WRITE = "openai/gpt-oss-120b"

# 技術ニュースのソース
ZENN_RSS = "https://zenn.dev/feed"
QIITA_RSS = "https://qiita.com/popular-items/feed"
HN_TOP = "https://hacker-news.firebaseio.com/v0/topstories.json"
HN_ITEM = "https://hacker-news.firebaseio.com/v0/item/{}.json"

# 一般ニュースのソース（Google News）
GNEWS_JP = "https://news.google.com/rss?hl=ja&gl=JP&ceid=JP:ja"
GNEWS_WORLD = "https://news.google.com/rss/headlines/section/topic/WORLD?hl=ja&gl=JP&ceid=JP:ja"

FETCH_LIMIT = 12
JST = timezone(timedelta(hours=9))

groq = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)


# ============================================================
# LLM呼び出し
# ============================================================

def chat(model: str, prompt: str, retries: int = 3, json_mode: bool = False) -> str:
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    for attempt in range(retries):
        try:
            res = groq.chat.completions.create(**kwargs)
            return res.choices[0].message.content or ""
        except Exception as e:
            wait = 2 ** attempt
            print(f"[warn] LLM呼び出し失敗 ({attempt + 1}/{retries}): {e}")
            if attempt == retries - 1:
                raise
            time.sleep(wait)
    return ""


def strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def parse_json(text: str) -> dict:
    cleaned = strip_code_fence(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise


def strip_html_tags(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    return text.strip()


# ============================================================
# データ取得（技術）
# ============================================================

def fetch_zenn(limit: int = FETCH_LIMIT) -> list[dict]:
    try:
        feed = feedparser.parse(ZENN_RSS)
        items = []
        for entry in feed.entries[:limit]:
            items.append({
                "source": "Zenn",
                "title": entry.get("title", ""),
                "url": entry.get("link", ""),
                "summary": strip_html_tags(entry.get("summary", ""))[:400],
            })
        print(f"[fetch] Zenn: {len(items)}件")
        return items
    except Exception as e:
        print(f"[warn] Zenn取得失敗: {e}")
        return []


def fetch_qiita(limit: int = FETCH_LIMIT) -> list[dict]:
    try:
        feed = feedparser.parse(QIITA_RSS)
        items = []
        for entry in feed.entries[:limit]:
            items.append({
                "source": "Qiita",
                "title": entry.get("title", ""),
                "url": entry.get("link", ""),
                "summary": strip_html_tags(entry.get("summary", ""))[:400],
            })
        print(f"[fetch] Qiita: {len(items)}件")
        return items
    except Exception as e:
        print(f"[warn] Qiita取得失敗: {e}")
        return []


def fetch_hn(limit: int = FETCH_LIMIT) -> list[dict]:
    try:
        res = requests.get(HN_TOP, timeout=10)
        ids = res.json()[:limit]
        items = []
        for story_id in ids:
            item = requests.get(HN_ITEM.format(story_id), timeout=10).json()
            if not item or item.get("type") != "story":
                continue
            items.append({
                "source": "Hacker News",
                "title": item.get("title", ""),
                "url": item.get("url") or f"https://news.ycombinator.com/item?id={story_id}",
                "summary": f"Score: {item.get('score', 0)} / Comments: {item.get('descendants', 0)}",
            })
        print(f"[fetch] Hacker News: {len(items)}件")
        return items
    except Exception as e:
        print(f"[warn] Hacker News取得失敗: {e}")
        return []


# ============================================================
# データ取得（一般）
# ============================================================

def fetch_gnews(url: str, source_name: str, limit: int = FETCH_LIMIT) -> list[dict]:
    try:
        feed = feedparser.parse(url)
        items = []
        for entry in feed.entries[:limit]:
            title = entry.get("title", "")
            # Google Newsは「見出し - 媒体名」の形式が多い
            items.append({
                "source": source_name,
                "title": title,
                "url": entry.get("link", ""),
                "summary": strip_html_tags(entry.get("summary", ""))[:400],
            })
        print(f"[fetch] {source_name}: {len(items)}件")
        return items
    except Exception as e:
        print(f"[warn] {source_name}取得失敗: {e}")
        return []


# ============================================================
# テーマ選定
# ============================================================

def pick_theme(sources: list[dict], genre: str) -> dict:
    """AIが今日のテーマを1つ選ぶ"""
    numbered = []
    for i, s in enumerate(sources):
        numbered.append(f"[{i}] ({s['source']}) {s['title']}\n    概要: {s['summary'][:200]}")

    if genre == "tech":
        role = "技術ニュースの編集長"
        criteria = (
            "- 複数の記事に共通するトレンド、または特に注目度の高いトピック\n"
            "- 日本語で読者に伝える価値があるもの\n"
            "- 技術者にとって興味深いもの"
        )
    else:
        role = "ニュースの編集長"
        criteria = (
            "- 日本国内の主要ニュース、または日本に影響のある海外ニュース\n"
            "- 複数の記事に共通するテーマ、または特に注目度の高いトピック\n"
            "- 一般読者にとって関心が高いもの"
        )

    prompt = f"""あなたは{role}です。
以下の記事リストから、今日取り上げるべき「テーマ」を1つ選んでください。

【記事リスト】
{chr(10).join(numbered)}

【選定基準】
{criteria}

【出力形式】JSONのみ。
{{
  "theme": "テーマ（20字以内の日本語）",
  "title": "記事タイトル（30字以内の日本語）",
  "reason": "なぜこのテーマを選んだか（100字程度）",
  "source_indices": [関連する記事のインデックス番号を3〜5個]
}}
"""
    raw = chat(GROQ_MODEL_PLAN, prompt, json_mode=True)
    theme = parse_json(raw)

    for key in ("theme", "title", "reason", "source_indices"):
        if key not in theme:
            raise ValueError(f"テーマ選定の出力に {key} がありません: {theme}")

    return theme


# ============================================================
# 記事執筆
# ============================================================

def write_article(theme: dict, selected: list[dict], genre: str) -> str:
    materials = []
    for s in selected:
        materials.append(
            f"【{s['source']}】{s['title']}\nURL: {s['url']}\n概要: {s['summary']}"
        )

    if genre == "tech":
        role = "技術ニュースの記者"
        tone = "技術者向けに、専門用語は補足しながら分かりやすく"
    else:
        role = "ニュース記者"
        tone = "一般読者向けに、背景から丁寧に分かりやすく"

    prompt = f"""あなたは{role}です。
以下のテーマと素材をもとに、日本語で長文のニュース記事を書いてください。

【テーマ】{theme['theme']}
【記事タイトル】{theme['title']}
【選定理由】{theme['reason']}

【素材】
{chr(10).join(materials)}

【記事の構成】
- リード文（150〜200字）：今日の話題を端的に
- 背景（400〜500字）：なぜこの話題が注目されているか
- 詳細（800〜1000字）：素材の情報を統合して掘り下げる
- 影響・展望（400〜500字）：社会・業界への影響、今後の見通し
- まとめ（150〜200字）：要点の再確認

合計 2000〜2500字程度。

【トーン】
{tone}

【使用可能なHTMLタグ】
<h2>, <h3>, <p>, <ul>, <li>, <strong>, <a>

【厳守事項】
- 素材に書かれていないことは書かないこと
- 推測や憶測を書く場合は「〜とみられる」「〜の可能性がある」と明示すること
- 具体的な数値や固有名詞は素材に書かれているもののみ使うこと
- 出典URLは本文中に埋め込まず、最後にまとめる（Python側で付与）
- HTMLの断片のみを出力すること。<html>や<body>は書かない
- コードブロック記号（```）は一切付けない
"""
    body = chat(GROQ_MODEL_WRITE, prompt)
    return strip_code_fence(body)


# ============================================================
# HTML生成
# ============================================================

def build_sources_html(sources: list[dict]) -> str:
    items = []
    for s in sources:
        items.append(
            f'<li><a href="{s["url"]}" target="_blank" rel="noopener">'
            f'{s["title"]}</a>'
            f'<span class="src-name">({s["source"]})</span></li>'
        )
    return "<ul>" + "".join(items) + "</ul>"


def render_html(theme: dict, body: str, sources: list[dict], genre: str) -> str:
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    today = datetime.now(JST).strftime("%Y年%m月%d日")
    sources_html = build_sources_html(sources)

    tech_active = ' class="active"' if genre == "tech" else ""
    normal_active = ' class="active"' if genre == "normal" else ""

    html = template.replace("{{TITLE}}", theme["title"])
    html = html.replace("{{DESCRIPTION}}", theme["reason"])
    html = html.replace("{{DATE}}", today)
    html = html.replace("{{BODY}}", body)
    html = html.replace("{{SOURCES}}", sources_html)
    html = html.replace("{{TECH_ACTIVE}}", tech_active)
    html = html.replace("{{NORMAL_ACTIVE}}", normal_active)
    return html


# ============================================================
# ニュース生成（共通）
# ============================================================

def generate_news(genre: str, sources: list[dict]) -> bool:
    """1つのニュースを生成する"""
    label = "技術ニュース" if genre == "tech" else "一般ニュース"
    print(f"\n===== {label} =====")

    if len(sources) < 3:
        print(f"[abort] {label}: 素材が不足しています（{len(sources)}件）")
        return False

    time.sleep(2)

    theme = pick_theme(sources, genre)
    print(f"[theme] {theme['theme']} - {theme['title']}")
    time.sleep(2)

    indices = theme.get("source_indices", [])
    selected = [sources[i] for i in indices if isinstance(i, int) and 0 <= i < len(sources)]
    if len(selected) < 2:
        selected = sources[:5]
    print(f"[select] {len(selected)}件の素材を選択")

    body = write_article(theme, selected, genre)
    print(f"[write] {len(body)}文字の記事を生成")
    time.sleep(2)

    html = render_html(theme, body, selected, genre)
    output_path = NEWS_DIR / f"{genre}.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"[save] {output_path}")
    return True


# ============================================================
# メイン
# ============================================================

def main() -> int:
    print("[start] ニュースを生成します")

    # 技術ニュース
    tech_sources = fetch_zenn() + fetch_qiita() + fetch_hn()
    tech_ok = generate_news("tech", tech_sources)

    # 一般ニュース
    normal_sources = (
        fetch_gnews(GNEWS_JP, "Google News 日本")
        + fetch_gnews(GNEWS_WORLD, "Google News 国際")
    )
    normal_ok = generate_news("normal", normal_sources)

    if not tech_ok and not normal_ok:
        print("[abort] 両方のニュース生成に失敗しました")
        return 1

    print("\n[done]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
