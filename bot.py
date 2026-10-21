#!/usr/bin/env python3
"""telegram-agent - talk to the local engine from Telegram.

The bot talks to the bmoe-server that runs on this phone
(http://127.0.0.1:8765/v1), so nothing leaves the device: the phone reads the
text, the engine answers, the phone sends the answer back.

Configuration (~/ai/etc/ai.env, or the environment):
    AI_TELEGRAM_TOKEN     bot token from @BotFather
    AI_TELEGRAM_USER_ID  your numeric telegram id - only this user is answered
    AI_BASE_URL          default http://127.0.0.1:8765/v1
    AI_MODEL_NAME        model name the server exposes (default: local)
"""

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

AI_ROOT = Path(os.environ.get("AI_ROOT", Path.home() / "ai"))
AI_ENV = AI_ROOT / "etc" / "ai.env"


def load_env_file(path: Path) -> None:
    """Read KEY=VALUE pairs from a shell-style env file into os.environ.

    Values already in the environment win: the file is a default, not an
    override.
    """
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


load_env_file(AI_ENV)

TOKEN = os.environ.get("AI_TELEGRAM_TOKEN", "").strip()
ALLOWED_USER_ID = os.environ.get("AI_TELEGRAM_USER_ID", "").strip()
BASE_URL = os.environ.get("AI_BASE_URL", "http://127.0.0.1:8765/v1").rstrip("/")
MODEL = os.environ.get("AI_MODEL_NAME", "local")
WHISPER_BIN = os.environ.get("AI_WHISPER_BIN", "").strip()
WHISPER_MODEL = os.environ.get("AI_WHISPER_MODEL", "").strip()

if not TOKEN:
    sys.exit("AI_TELEGRAM_TOKEN is not set - put it in %s" % AI_ENV)

try:
    from telegram import Update
    from telegram.ext import (
        Application,
        CommandHandler,
        MessageHandler,
        ContextTypes,
        filters,
    )
    from openai import OpenAI
except ImportError as exc:  # pragma: no cover - the phone installs these
    sys.exit("missing python package (%s): pip install python-telegram-bot openai" % exc)

client = OpenAI(base_url=BASE_URL, api_key=os.environ.get("AI_API_KEY", "local"), timeout=600)


def allowed(update: Update) -> bool:
    """Only the owner of the stack gets answers."""
    if not ALLOWED_USER_ID:
        return True
    user = update.effective_user
    return user is not None and str(user.id) == ALLOWED_USER_ID


async def say(update: Update, text: str) -> None:
    if update.effective_message is None:
        return
    await update.effective_message.reply_text(text[:4000])


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not allowed(update):
        return
    await say(
        update,
        "Local engine online.\n"
        "Send a message and I will answer from the model running on this phone.\n"
        "/stop - stop the engine\n/status - what is running\n/help - this list",
    )


async def cmd_stop(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not allowed(update):
        return
    subprocess.run(["ai", "stop"], check=False)
    await say(update, "Engine stopped.")


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not allowed(update):
        return
    out = subprocess.run(["ai", "status"], capture_output=True, text=True, check=False)
    await say(update, out.stdout or out.stderr or "no output")


async def transcribe(ogg_path: str) -> str:
    """Turn a voice note into text with a local whisper.cpp, when there is one."""
    if not WHISPER_BIN or not WHISPER_MODEL or not os.path.exists(WHISPER_BIN):
        return ""
    wav_path = ogg_path.rsplit(".", 1)[0] + ".wav"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", ogg_path, "-ar", "16000", "-ac", "1", wav_path],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            [WHISPER_BIN, "-m", WHISPER_MODEL, "-f", wav_path, "-otxt", "-of", wav_path[:-4]],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        text_file = Path(wav_path[:-4] + ".txt")
        return text_file.read_text(encoding="utf-8", errors="replace").strip() if text_file.exists() else ""
    except Exception:
        return ""
    finally:
        for path in (ogg_path, wav_path, wav_path[:-4] + ".txt"):
            try:
                os.unlink(path)
            except OSError:
                pass


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not allowed(update):
        return
    voice = update.effective_message.voice if update.effective_message else None
    if voice is None:
        return
    status = await update.effective_message.reply_text("transcribing...")
    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
        ogg_path = tmp.name
    try:
        await voice.get_file().download_to_drive(ogg_path)
        text = await transcribe(ogg_path)
    finally:
        if os.path.exists(ogg_path):
            os.unlink(ogg_path)
    if not text:
        await status.edit_text("I could not transcribe that voice note on this phone.")
        return
    await status.edit_text(f"you said: {text}")
    await answer(update, text)


async def answer(update: Update, prompt: str) -> None:
    """Ask the local engine and send the answer back, in chunks."""
    status = await update.effective_message.reply_text("thinking...")
    try:
        stream = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
    except Exception as exc:
        await status.edit_text(f"The engine did not answer: {exc}")
        return

    parts = []
    started = time.time()
    try:
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            piece = getattr(delta, "content", None)
            if piece:
                parts.append(piece)
    except Exception as exc:
        await status.edit_text(f"The engine stopped mid-answer: {exc}")
        return

    text = "".join(parts).strip()
    elapsed = time.time() - started
    if not text:
        await status.edit_text("The engine answered with nothing.")
        return
    header = f"({elapsed:.1f}s)\n\n"
    await status.edit_text((header + text)[:4000])
    if len(header + text) > 4000:
        for i in range(4000, len(header + text), 4000):
            await update.effective_message.reply_text((header + text)[i : i + 4000])


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not allowed(update):
        return
    await answer(update, update.effective_message.text)


async def handle_other(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not allowed(update):
        return
    await say(update, "I understand text and voice notes.")


def main() -> None:
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_start))
    app.add_handler(CommandHandler("stop", cmd_stop))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.ALL & ~filters.COMMAND & ~filters.VOICE, handle_other))
    print("telegram-agent: polling", flush=True)
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
