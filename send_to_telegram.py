import os
import json
import requests

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
GITHUB_REPOSITORY = os.environ.get("GITHUB_REPOSITORY")  # GitHub Actionsが自動設定する "owner/repo"
NOTIFY_PAYLOAD_FILE = "notify_payloads.json"


def public_raw_url(relative_path: str) -> str:
    branch = os.environ.get("GITHUB_REF_NAME", "main")
    return f"https://raw.githubusercontent.com/{GITHUB_REPOSITORY}/{branch}/{relative_path}"


def send_photo(image_url: str, caption: str = "") -> None:
    resp = requests.post(
        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto",
        data={"chat_id": TELEGRAM_CHAT_ID, "photo": image_url, "caption": caption[:1024]},
        timeout=20,
    )
    if resp.status_code == 200:
        print("✅ Telegramへ画像を送信しました")
    else:
        print(f"⚠️ Telegram画像送信失敗: HTTP {resp.status_code} {resp.text}")


def send_text(text: str) -> None:
    # Telegramの1通あたり上限は4096文字。安全のため4000で区切って分割送信する。
    for i in range(0, len(text), 4000):
        chunk = text[i:i + 4000]
        resp = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            data={"chat_id": TELEGRAM_CHAT_ID, "text": chunk},
            timeout=15,
        )
        if resp.status_code == 200:
            print("✅ Telegramへテキストを送信しました")
        else:
            print(f"⚠️ Telegramテキスト送信失敗: HTTP {resp.status_code} {resp.text}")


def main():
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID が未設定のため送信をスキップします")
        return

    if not os.path.exists(NOTIFY_PAYLOAD_FILE):
        print(f"⚠️ {NOTIFY_PAYLOAD_FILE} が見つかりません。送信するものがありません。")
        return

    with open(NOTIFY_PAYLOAD_FILE, "r", encoding="utf-8") as f:
        payloads = json.load(f)

    if not payloads:
        print("送信対象がありません(全銘柄で分析に失敗した可能性があります)。")
        return

    # 画像には直後のテキストの先頭部分をキャプションとして添えず、
    # 画像→本文の順で見やすく分けて送る
    for item in payloads:
        if item["type"] == "image":
            send_photo(public_raw_url(item["path"]))
        elif item["type"] == "text":
            send_text(item["text"])


if __name__ == "__main__":
    main()
