# telegram-agent

Talk to the local engine from Telegram, with nothing leaving the phone.

The bot is a thin client of the OpenAI-compatible server that `ai go` starts
on `127.0.0.1:8765`.

## Install

```bash
pip install python-telegram-bot openai
```

Then put the bot token in `~/ai/etc/ai.env`:

```
AI_TELEGRAM_TOKEN=123456:ABC-DEF...
AI_TELEGRAM_USER_ID=123456789
```

`ai go` starts the bridge only when the token is set.

## Commands

| command | what it does |
|---|---|
| `/start` | what this bot is |
| `/status` | the output of `ai status` |
| `/stop` | stops the whole stack |

Plain text is answered by the model on the phone. Voice notes are
transcribed locally when a whisper.cpp binary and model are configured
(`AI_WHISPER_BIN`, `AI_WHISPER_MODEL`).

## Notes

- `AI_TELEGRAM_USER_ID` is an allowlist: without it the bot answers anybody
  who finds it, which is a model on your phone answering strangers.
- long polling is used, so no port has to be open in the router.
