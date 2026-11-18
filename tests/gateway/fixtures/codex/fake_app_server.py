"""A stand-in for ``codex app-server``: answers the JSON-RPC the backend speaks and replays recorded
turns. ``FAKE_CODEX_SCENARIO`` names a JSON scenario; everything received goes, one JSON line each, to
``FAKE_CODEX_LOG``, after a first line with the environment it runs in.

The scenario is read as each call needs it, so a test may set it after the process started. Its
``turns`` are taken one per ``turn/start``, each a list of frames (or the name of a
recording beside this file): ``{"notify": method, "params"}``, ``{"request": method, "params"}`` (waits
for the answer), ``{"await": method}`` (waits until the client sends it), ``{"exit": code}``, and
``{"end": status}``: the turn ends unannounced, as when it ended just as a steer was on its way; the
next ``turn/steer`` hears of it first (``turn/completed``, then the steer's error). In a
frame, ``$THREAD``, ``$TURN`` and ``$CWD`` stand for the turn's thread, its id and the thread's folder.
"""

import json
import os
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))


def scenario():
    with open(os.environ["FAKE_CODEX_SCENARIO"]) as file:
        return json.load(file)


log = open(os.environ["FAKE_CODEX_LOG"], "a", buffering=1)
log.write(json.dumps({"env": {key: os.environ.get(key) for key in ("HOME", "CODEX_HOME")}}) + "\n")

out = threading.Lock()
seen = threading.Condition()
received: list[dict] = []  # every message from the client, for the frames that wait on one
answers: dict[int, dict] = {}
threads: dict[str, str] = {}  # id -> cwd
taken = iter(range(1_000_000))  # the turns this process started
active: dict[str, str] = {}  # thread -> its running turn
ended: dict[str, tuple] = {}  # thread -> the turn that ended unannounced, and how
counter = iter(range(1, 1_000_000))


def write(message):
    with out:
        sys.stdout.write(json.dumps({"jsonrpc": "2.0", **message}) + "\n")
        sys.stdout.flush()


def fill(value, thread, turn):
    text = json.dumps(value).replace("$THREAD", thread).replace("$TURN", turn).replace("$CWD", threads[thread])
    return json.loads(text)


def replay(frames, thread, turn):
    mark = len(received)
    for frame in frames:
        if "exit" in frame:
            os._exit(frame["exit"])
        if "end" in frame:
            active.pop(thread, None)
            ended[thread] = (turn, frame["end"])
            continue
        if "await" in frame:
            with seen:
                seen.wait_for(lambda: any(m.get("method") == frame["await"] for m in received[mark:]))
                mark = next(i for i in range(mark, len(received)) if received[i].get("method") == frame["await"]) + 1
            continue
        params = fill(frame["params"], thread, turn)
        if "request" in frame:
            request_id = next(counter)
            write({"id": request_id, "method": frame["request"], "params": params})
            with seen:
                seen.wait_for(lambda: request_id in answers)
            continue
        if frame["notify"] == "turn/completed":
            active.pop(thread, None)
        write({"method": frame["notify"], "params": params})


def turn_frames(thread, turn):
    turns, index = scenario().get("turns", []), next(taken)
    if index >= len(turns):
        return [{"notify": "turn/started", "params": {"threadId": "$THREAD", "turn": {"id": "$TURN"}}},
                {"notify": "turn/completed", "params": {"threadId": "$THREAD", "turn": {"id": "$TURN", "status": "completed", "error": None}}}]
    frames = turns[index]
    return json.load(open(os.path.join(HERE, frames))) if isinstance(frames, str) else frames


def thread_result(thread, params):
    return {"thread": {"id": thread, "parentThreadId": None, "cwd": threads[thread]},
            "model": params.get("model") or "gpt-5.6-sol", "reasoningEffort": "low"}


def handle(message):
    method, params, request_id = message.get("method"), message.get("params") or {}, message.get("id")
    if method is None:
        with seen:
            answers[request_id] = message
            seen.notify_all()
        return
    with seen:
        received.append(message)
        seen.notify_all()
    if request_id is None:
        return
    result = None
    if method == "initialize":
        result = {"userAgent": "fake", "codexHome": os.environ.get("CODEX_HOME"), "platformFamily": "unix", "platformOs": "macos"}
    elif method == "account/read":
        result = {"account": scenario().get("account", {"type": "chatgpt", "email": None, "planType": "pro"}),
                  "requiresOpenaiAuth": True}
    elif method == "model/list":
        models = scenario().get("models") or json.load(open(os.path.join(HERE, "models.json")))
        result = {"data": models, "nextCursor": None}
    elif method == "config/read":
        result = {"config": scenario().get("config", {})}
    elif method == "configRequirements/read":
        result = {"requirements": scenario().get("requirements")}
    elif method == "thread/start":
        thread = f"thread-{next(counter)}"
        threads[thread] = params.get("cwd") or ""
        result = thread_result(thread, params)
    elif method == "thread/resume":
        if params["threadId"] in scenario().get("lost", []):
            write({"id": request_id, "error": {"code": -32600, "message": f"no rollout found for thread id {params['threadId']}"}})
            return
        threads[params["threadId"]] = params.get("cwd") or ""
        result = thread_result(params["threadId"], params)
    elif method == "thread/read":
        result = {"thread": {"id": params["threadId"], "parentThreadId": scenario().get("parents", {}).get(params["threadId"])}}
    elif method == "thread/unsubscribe":
        result = {"status": "unsubscribed"}
    elif method == "turn/start":
        thread, turn = params["threadId"], f"turn-{next(counter)}"
        active[thread] = turn
        write({"id": request_id, "result": {"turn": {"id": turn, "items": [], "status": "inProgress", "error": None}}})
        threading.Thread(target=replay, args=(turn_frames(thread, turn), thread, turn), daemon=True).start()
        return
    elif method == "turn/steer":
        if params["threadId"] in ended:
            turn, status = ended.pop(params["threadId"])
            write({"method": "turn/completed", "params": {"threadId": params["threadId"], "turn": {
                "id": turn, "status": status, "error": None}}})
        if active.get(params["threadId"]) != params["expectedTurnId"]:
            write({"id": request_id, "error": {"code": -32600, "message": "no active turn to steer"}})
            return
        result = {"turnId": params["expectedTurnId"]}
    elif method == "turn/interrupt":
        result = {}
    if result is None:
        write({"id": request_id, "error": {"code": -32601, "message": f"{method} is not faked"}})
        return
    write({"id": request_id, "result": result})


for line in sys.stdin:
    message = json.loads(line)
    log.write(json.dumps(message) + "\n")
    handle(message)
