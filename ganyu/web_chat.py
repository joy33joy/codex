"""
Chat with Ganyu in your web browser.

Starts a small web server on your computer that talks to the fine-tuned
model on Tinker (no system prompt, same as chat.py). Your Tinker key stays
on the server; the browser never sees it.

Run:  python ganyu/web_chat.py
Then open http://localhost:8000 in your browser.

Options:
  --port 8080          use a different port
  --host 0.0.0.0       let other devices on your network open it (your
                       Tinker credits pay for everyone who chats)
  --model-path tinker://...   chat with a different checkpoint
"""

import argparse
from pathlib import Path

import tinker
from aiohttp import web

from common import CHECKPOINT_FILE, STUDENT_MODEL, STUDENT_RENDERER, chat_once, get_renderer, reply_as_ganyu

PAGE = Path(__file__).parent / "web" / "index.html"
MAX_TURNS = 40  # keep only the last 40 messages so requests stay small
MAX_CHARS = 4000  # longest message accepted from the browser


def build_app(model_path: str) -> web.Application:
    app = web.Application(client_max_size=256 * 1024)
    renderer = get_renderer(STUDENT_MODEL, STUDENT_RENDERER)

    async def on_startup(app: web.Application) -> None:
        app["ganyu"] = await tinker.ServiceClient().create_sampling_client(model_path=model_path)
        print(f"Ganyu is ready ({model_path})")

    async def index(request: web.Request) -> web.Response:
        return web.FileResponse(PAGE)

    async def chat(request: web.Request) -> web.Response:
        try:
            body = await request.json()
            messages = [
                {"role": m["role"], "content": str(m["content"])[:MAX_CHARS]}
                for m in body["messages"]
                if m.get("role") in ("user", "assistant") and str(m.get("content", "")).strip()
            ][-MAX_TURNS:]
        except Exception:
            return web.json_response({"error": "Bad request."}, status=400)
        if not messages or messages[-1]["role"] != "user":
            return web.json_response({"error": "The last message must be yours."}, status=400)
        try:
            reply = await reply_as_ganyu(app["ganyu"], renderer, messages)
        except Exception as e:
            print(f"Tinker error: {type(e).__name__}: {e}")
            return web.json_response({"error": "Ganyu couldn't answer just now. Please try again."},
                                     status=502)
        return web.json_response({"reply": reply})

    app.on_startup.append(on_startup)
    app.router.add_get("/", index)
    app.router.add_post("/api/chat", chat)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with Ganyu in your browser.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-path", default=None, help="default: the address in checkpoint.txt")
    args = parser.parse_args()
    model_path = args.model_path or CHECKPOINT_FILE.read_text().strip()
    print(f"Open http://localhost:{args.port} in your browser (Ctrl+C to stop).")
    web.run_app(build_app(model_path), host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
