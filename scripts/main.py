"""
毎日1つ、AIがWebツールを生成してサイトに追加する。
テンプレートをベースに生成するので、全ツールのデザインが統一される。
Blueskyの環境変数が設定されていれば自動投稿、未設定ならスキップ。

必要な環境変数:
  GROQ_API_KEY       - Groq APIキー（必須）
  BSKY_HANDLE        - Blueskyハンドル（任意）
  BSKY_APP_PASSWORD  - Blueskyアプリパスワード（任意）
  SITE_URL           - 例: https://azenzazza.github.io
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import date
from pathlib import Path

from openai import OpenAI

# ============================================================
# 設定
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
TOOLS_DIR = ROOT / "tools"
DATA_DIR = ROOT / "data"
INDEX_JSON = DATA_DIR / "tools.json"
TEMPLATE_PATH = ROOT / "scripts" / "template.html"

SITE_URL = os.environ.get("SITE_URL", "https://azenzazza.github.io")

# モデル（Groqに統一）
GROQ_MODEL_PLAN = "openai/gpt-oss-20b"
GROQ_MODEL_REVIEW = "openai/gpt-oss-20b"
GROQ_MODEL_POST = "openai/gpt-oss-20b"
GROQ_MODEL_CODE = "openai/gpt-oss-120b"

MAX_REVIEW_RETRY = 3
SLEEP_BETWEEN_CALLS = 2  # 秒

# Blueskyが有効かどうか
BSKY_ENABLED = bool(
    os.environ.get("BSKY_HANDLE") and os.environ.get("BSKY_APP_PASSWORD")
)

# ============================================================
# LLMクライアント
# ============================================================

groq = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)


def chat(client: OpenAI, model: str, prompt: str, retries: int = 3) -> str:
    """LLM呼び出し（429時は指数バックオフでリトライ）"""
    for attempt in range(retries):
        try:
            res = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            return res.choices[0].message.content or ""
        except Exception as e:
            wait = 2 ** attempt
            print(f"[warn] LLM呼び出し失敗 ({attempt + 1}/{retries}): {e}")
            if attempt == retries - 1:
                raise
            time.sleep(wait)
    return ""


def strip_code_fence(text: str) -> str:
    """AIが ``` で囲んできた場合に除去"""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def parse_json(text: str) -> dict:
    """AI出力からJSONを抽出"""
    cleaned = strip_code_fence(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise


# ============================================================
# tools.json の読み書き
# ============================================================

def load_tools() -> list[dict]:
    if not INDEX_JSON.exists():
        return []
    return json.loads(INDEX_JSON.read_text(encoding="utf-8"))


def save_tools(tools: list[dict]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_JSON.write_text(
        json.dumps(tools, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def next_id(tools: list[dict]) -> str:
    return f"{len(tools) + 1:03d}"


# ============================================================
# エージェント
# ============================================================

def plan_tool(existing: list[dict]) -> dict:
    """企画AI：重複しないツール案を1つ提案"""
    existing_brief = [
        {"title": t["title"], "category": t.get("category", "")}
        for t in existing
    ]
    prompt = f"""あなたは「毎日1つ便利なWebツールを作る」プロジェクトの企画担当です。
以下の既存ツールと重複しない、新しいWebツールを1つ提案してください。

【制約】
- HTML/CSS/JSのみで完結（外部API・外部ライブラリ禁止）
- ログイン不要、1ファイルで動作
- 実装が30分以内で終わる小規模なもの
- 実用的で、誰かが「使いたい」と思うもの
- 既存ツールと機能が被らないこと

【既存ツール（{len(existing_brief)}件）】
{json.dumps(existing_brief, ensure_ascii=False, indent=2)}

【出力形式】JSONのみ。コードブロックで囲まないこと。
{{
  "slug": "英数字とハイフンのみ（例: char-count）",
  "title": "日本語タイトル（20文字以内）",
  "description": "30文字以内の説明",
  "category": "text|image|calc|convert|generate|other のいずれか",
  "spec": "実装仕様を箇条書きで3〜6行"
}}
"""
    raw = chat(groq, GROQ_MODEL_PLAN, prompt)
    plan = parse_json(raw)

    for key in ("slug", "title", "description", "category", "spec"):
        if key not in plan:
            raise ValueError(f"企画AIの出力に {key} がありません: {plan}")

    plan["slug"] = re.sub(r"[^a-z0-9-]", "", plan["slug"].lower()) or "tool"
    return plan


def implement_tool(plan: dict) -> str:
    """実装AI：テンプレートをベースにHTMLを生成"""
    template = TEMPLATE_PATH.read_text(encoding="utf-8")

    prompt = f"""あなたはWebツールの実装担当です。
以下の「テンプレート」をベースに、指定されたツールを実装してください。

【ツール名】{plan['title']}
【説明】{plan['description']}
【仕様】
{plan['spec']}

【実装ルール】
- テンプレートの構造（header, main, footer, style, script）は絶対に変更しない
- プレースホルダを以下のように置き換える：
  - {{TITLE}} → {plan['title']}
  - {{DESCRIPTION}} → {plan['description']}
  - {{CONTENT}} → ツールの入力フォームやボタンなど（HTML）
  - {{SCRIPT}} → ツールの動作を実装するJavaScript
- テンプレートの <style> は変更しない（既存のクラス・ボタン・テーブルスタイルをそのまま使う）
- 結果は id="result" の要素に textContent で出力する
- innerHTML / eval / document.write は使わない
- 外部ライブラリ・外部API禁止
- 出力はHTMLのみ。説明文・コードブロック記号は一切付けない

【テンプレート】
{template}

【出力】
上記テンプレートのプレースホルダを埋めた、完全なHTMLファイルを出力してください。
"""
    html = chat(groq, GROQ_MODEL_CODE, prompt)
    return strip_code_fence(html)


def review_tool(html: str) -> dict:
    """検証AI：セキュリティ・バグをチェック"""
    prompt = f"""以下のHTMLコードをレビューしてください。

【チェック項目】
- innerHTML / eval / document.write の使用
- 無限ループ、メモリリーク
- ユーザー入力のエスケープ漏れ
- 明らかなバグ、動作しない箇所
- 外部リソース（CDN・API）への依存
- プレースホルダ（{{{{TITLE}}}} など）が残っていないか

【出力形式】JSONのみ。コードブロックで囲まないこと。
{{
  "ok": true または false,
  "issues": ["問題点を簡潔に"],
  "fix_hint": "修正の方向性を1〜2行で"
}}

【コード】
{html}
"""
    raw = chat(groq, GROQ_MODEL_REVIEW, prompt)
    try:
        return parse_json(raw)
    except Exception:
        return {"ok": False, "issues": ["レビュー結果のパース失敗"], "fix_hint": ""}


def fix_tool(html: str, issues: list[str], hint: str) -> str:
    """修正AI：指摘を反映"""
    prompt = f"""以下のHTMLコードに問題があります。修正してください。

【問題点】
{json.dumps(issues, ensure_ascii=False, indent=2)}

【修正の方向性】
{hint}

【厳守事項】
- HTML 1ファイルで完結
- 外部ライブラリ・外部API禁止
- innerHTML / eval / document.write 禁止
- 既存の <style> は変更しない
- 出力はHTMLのみ。説明文・コードブロック記号は一切付けない。

【修正前のコード】
{html}
"""
    fixed = chat(groq, GROQ_MODEL_CODE, prompt)
    return strip_code_fence(fixed)


# ============================================================
# ファイル保存
# ============================================================

def write_tool(tool_id: str, plan: dict, html: str) -> str:
    dir_name = f"{tool_id}-{plan['slug']}"
    tool_dir = TOOLS_DIR / dir_name
    tool_dir.mkdir(parents=True, exist_ok=True)
    (tool_dir / "index.html").write_text(html, encoding="utf-8")
    return f"{SITE_URL}/tools/{dir_name}/"


def append_to_index(tool_id: str, plan: dict) -> None:
    tools = load_tools()
    tools.append({
        "id": tool_id,
        "slug": plan["slug"],
        "title": plan["title"],
        "description": plan["description"],
        "category": plan["category"],
        "created": date.today().isoformat(),
        "path": f"tools/{tool_id}-{plan['slug']}/",
    })
    save_tools(tools)


# ============================================================
# Bluesky投稿（環境変数があれば実行）
# ============================================================

def generate_post_text(plan: dict, url: str) -> str:
    """告知AI：Bluesky投稿文を生成"""
    prompt = f"""以下をBlueskyに投稿する文面にしてください。

【ツール名】{plan['title']}
【説明】{plan['description']}
【URL】{url}

【制約】
- 全体で280文字以内（URL含む）
- 絵文字を1〜2個
- 宣伝臭を抑え、便利さを端的に
- 末尾にハッシュタグ #AIが作ったツール #毎日ツール
- 出力は投稿文のみ。説明・引用符・コードブロックは付けない。
"""
    return chat(groq, GROQ_MODEL_POST, prompt).strip()


def post_to_bluesky(text: str) -> None:
    """Blueskyに投稿（環境変数未設定ならスキップ）"""
    if not BSKY_ENABLED:
        print("[skip] Bluesky未設定のため投稿をスキップ")
        return
    try:
        from atproto import Client
        client = Client()
        client.login(
            os.environ["BSKY_HANDLE"],
            os.environ["BSKY_APP_PASSWORD"],
        )
        client.send_post(text=text)
        print("[ok] Blueskyに投稿しました")
    except Exception as e:
        print(f"[warn] Bluesky投稿に失敗: {e}")


# ============================================================
# メイン
# ============================================================

def main() -> int:
    print("[start] 今日のツールを生成します")
    print(f"[info] Bluesky: {'有効' if BSKY_ENABLED else '無効（未設定）'}")

    existing = load_tools()
    tool_id = next_id(existing)

    # 1. 企画
    plan = plan_tool(existing)
    print(f"[plan] {plan['title']} ({plan['slug']})")
    time.sleep(SLEEP_BETWEEN_CALLS)

    # 2. 実装
    html = implement_tool(plan)
    print(f"[code] {len(html)} 文字のHTMLを生成")
    time.sleep(SLEEP_BETWEEN_CALLS)

    # 3. 検証 → 必要なら修正
    for attempt in range(MAX_REVIEW_RETRY):
        review = review_tool(html)
        if review.get("ok"):
            print("[review] OK")
            break
        print(f"[review] NG ({attempt + 1}/{MAX_REVIEW_RETRY}): {review.get('issues')}")
        html = fix_tool(html, review.get("issues", []), review.get("fix_hint", ""))
        time.sleep(SLEEP_BETWEEN_CALLS)
    else:
        print("[abort] 検証を通過できませんでした。今日は公開しません。")
        return 1

    # 4. 保存
    url = write_tool(tool_id, plan, html)
    append_to_index(tool_id, plan)
    print(f"[save] {url}")

    # 5. Bluesky告知（有効なら）
    if BSKY_ENABLED:
        post_text = generate_post_text(plan, url)
        print(f"[post] {post_text}")
        post_to_bluesky(post_text)

    print("[done]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
