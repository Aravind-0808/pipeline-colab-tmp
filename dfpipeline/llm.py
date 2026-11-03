"""Single gateway to the LLM.

Everything AI-facing goes through here: one OpenAI-compatible endpoint, so the
model, key and budget live in one place and can be swapped without touching the
callers. No weights are ever loaded locally — this is an HTTP client.

Every call goes through this client, which is what makes the concurrency gate
and the circuit breaker below mean anything — a caller that reached the gateway
another way would be bound by neither.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass

from dfpipeline import limits
from dfpipeline.config import settings
from dfpipeline.redis import complain, get_redis

log = logging.getLogger("llm")

_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S)
_THINK = re.compile(r"<think>.*?</think>", re.S)


class LlmUnavailable(RuntimeError):
    """Raised when the gateway is unreachable or misconfigured.

    Callers must treat this as 'undecided', never as a rejection — we do not
    penalise an operator because our own infrastructure failed.
    """


# A model that is down does not fail fast. It accepts the connection and never
# answers, so every call costs the full timeout and both retries — about three
# minutes of a worker slot to learn what the previous clip already proved. With
# four slots, a queue of fifty clips spends hours discovering the same thing
# fifty times.
#
# So the first few failures are paid for, and then the calls stop for a while.
# The clips still come back undecided, which is the same answer they would have
# got, and the queue drains in seconds instead of hours. One call is allowed
# through when the pause ends, so a model that comes back is noticed.
_BREAKER_TRIPS_AFTER = 3
_BREAKER_PAUSE_SEC = 120.0


# "Did not answer" rather than "answered with a refusal". The difference
# decides both whether another route is worth trying and whether re-asking
# without extra_body could possibly help.
_TRANSPORT_WORDS = (
    "connection", "connect", "timed out", "timeout", "network", "unreachable", "refused",
)


def _looks_like_transport_error(msg: str) -> bool:
    low = msg.lower()
    return any(w in low for w in _TRANSPORT_WORDS)


_BREAKER_KEY = "llm:breaker"
_BREAKER_KEEP_SEC = 24 * 3600
_GATE_KEY = "llm:gate"


class _Breaker:
    """Consecutive failures, and when to stop asking.

    Shared through Redis when it is configured: the calls are made by worker
    processes on more than one machine, and a breaker each of them keeps alone
    lets every process pay for its own three failures — and shows the monitor,
    which runs in the API process, a breaker that never trips at all. Without
    Redis it is this process's own, as it always was.
    """

    def __init__(self) -> None:
        self.failures = 0
        self.paused_until = 0.0  # time.monotonic(), local state only

    def _shared(self) -> tuple[int, float] | None:
        r = get_redis()
        if r is None:
            return None
        try:
            state = r.hgetall(_BREAKER_KEY)
            return int(state.get("failures") or 0), float(state.get("paused_until") or 0)
        except Exception as exc:
            complain("the model breaker", exc)
            return None

    def state(self) -> tuple[int, float]:
        """(consecutive failures, seconds until calls resume)."""
        shared = self._shared()
        if shared is not None:
            failures, until = shared
            return failures, max(0.0, until - time.time())
        return self.failures, max(0.0, self.paused_until - time.monotonic())

    def open(self) -> bool:
        return self.state()[1] > 0

    def wait_sec(self) -> int:
        return round(self.state()[1])

    def record_failure(self, error: object = None) -> None:
        r = get_redis()
        if r is not None:
            try:
                failures = int(r.hincrby(_BREAKER_KEY, "failures", 1))
                # The outcome, for the Monitor: it shows the last real call
                # rather than asking the model on a timer of its own.
                r.hset(_BREAKER_KEY, mapping={"failed_at": time.time(),
                                              "error": str(error or "no answer").splitlines()[0][:300]})
                if failures >= _BREAKER_TRIPS_AFTER:
                    r.hset(_BREAKER_KEY, "paused_until", time.time() + _BREAKER_PAUSE_SEC)
                    self._tripped(failures)
                r.expire(_BREAKER_KEY, _BREAKER_KEEP_SEC)
                return
            except Exception as exc:
                complain("the model breaker", exc)
        self.failures += 1
        if self.failures >= _BREAKER_TRIPS_AFTER:
            self.paused_until = time.monotonic() + _BREAKER_PAUSE_SEC
            self._tripped(self.failures)

    @staticmethod
    def _tripped(failures: int) -> None:
        log.warning(
            "the model has failed %d times in a row - not calling it again for %.0fs",
            failures,
            _BREAKER_PAUSE_SEC,
        )

    def record_success(self) -> None:
        if self.state()[0]:
            log.info("the model is answering again")
        self.failures = 0
        self.paused_until = 0.0
        r = get_redis()
        if r is not None:
            try:
                r.delete(_BREAKER_KEY)
                r.hset(_BREAKER_KEY, "ok_at", time.time())
                r.expire(_BREAKER_KEY, _BREAKER_KEEP_SEC)
            except Exception as exc:
                complain("the model breaker", exc)


_breaker = _Breaker()


# The gateway fronts one GPU box. Nothing here was stopping us from sending it
# as many calls at once as there are worker slots — and a second worker process
# doubled that again. A bulk re-check of fifty clips would have arrived as a
# burst, which is how you take a machine down rather than keep it busy.
#
# One slot per call, held for the whole exchange with the model, so one clip
# occupies one slot from first byte to last. The slots are counted in Redis
# when it is configured, so the limit holds across every worker process on
# every machine; without it, per process as before. A slot is leased for three
# timeouts, so a worker killed mid-call gives its slot back by itself.
def _gate_lease() -> float:
    return settings.llm_timeout_sec * 3


class _Slot:
    """One permit to talk to the model, waited for and given back."""

    def __enter__(self):
        # Bounded so a saturated gateway cannot pile up workers for ever; the
        # caller treats the timeout as "undecided", never a rejection.
        self._held = limits.slot(
            _GATE_KEY,
            max(1, settings.llm_max_concurrency),
            _gate_lease(),
            wait_sec=settings.llm_timeout_sec * 2,
        )
        try:
            self._held.__enter__()
        except limits.SlotTimeout:
            raise LlmUnavailable(
                f"waited {settings.llm_timeout_sec * 2:.0f}s for a free slot on the model "
                "and the queue is still busy"
            ) from None
        return self

    def __exit__(self, *exc):
        self._held.__exit__(*exc)
        return False


def gate_state() -> dict:
    """How much of the model we are using, for the monitor to show."""
    return {
        "calls_in_flight": limits.held(_GATE_KEY, _gate_lease()),
        "max_concurrent_calls": settings.llm_max_concurrency,
    }


def breaker_state() -> dict:
    """What the breaker is doing, for the monitor to show."""
    failures, wait = _breaker.state()
    return {
        "consecutive_failures": failures,
        "paused": wait > 0,
        "resumes_in_sec": round(wait),
    }


def last_call() -> dict | None:
    """The outcome of the last real call to the model, as the breaker kept it:
    {ok, at, error}, or None when nothing is known."""
    r = get_redis()
    if r is None:
        return None
    state = r.hgetall(_BREAKER_KEY)
    failed, ok = float(state.get("failed_at") or 0), float(state.get("ok_at") or 0)
    if not failed and not ok:
        return None
    if failed > ok:
        return {"ok": False, "at": failed, "error": state.get("error")}
    return {"ok": True, "at": ok, "error": None}


def strip_to_json(text: str) -> str:
    text = _THINK.sub("", text or "").strip()
    if m := _FENCE.search(text):
        text = m.group(1).strip()
    if not text.startswith("{") and (start := text.find("{")) != -1:
        depth = 0
        for i, ch in enumerate(text[start:], start):
            depth += ch == "{"
            depth -= ch == "}"
            if depth == 0:
                return text[start : i + 1]
    return text


@dataclass
class LlmClient:
    api_base: str | None = None
    api_key: str = "EMPTY"
    model: str | None = None
    timeout: float = 90.0

    @classmethod
    def from_settings(cls) -> "LlmClient":
        return cls(
            api_base=settings.llm_api_base,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            timeout=settings.llm_timeout_sec,
        )

    @property
    def configured(self) -> bool:
        return bool(self.api_base and self.model)

    def _candidate_urls(self) -> list[str]:
        if not self.api_base:
            return []
        candidates = [self.api_base]
        from urllib.parse import urlparse, urlunparse
        try:
            parsed = urlparse(self.api_base)
            host = (parsed.hostname or "").lower()
            if host and host not in ("litellm", "host.docker.internal", "127.0.0.1", "localhost"):
                port_suffix = f":{parsed.port}" if parsed.port else ""
                candidates.append(urlunparse(parsed._replace(netloc=f"host.docker.internal{port_suffix}")))
                candidates.append(urlunparse(parsed._replace(netloc=f"172.17.0.1{port_suffix}")))
        except Exception:
            pass
        return candidates

    def _client(self, base_url: str | None = None):
        from openai import OpenAI

        # No retries of the SDK's own (it makes two by default): each one is
        # another request at a gateway that just failed, unseen by the breaker,
        # so one check against a dead model showed in LiteLLM as three.
        return OpenAI(
            base_url=base_url or self.api_base, api_key=self.api_key, timeout=self.timeout, max_retries=0
        )

    def chat_json(
        self,
        messages: list[dict],
        max_tokens: int = 400,
        temperature: float = 0.1,
    ) -> dict:
        """Ask for strict JSON, with one self-correction retry.

        Mirrors what lerobot's own VLM client does: models occasionally wrap the
        answer in prose or fences, and a single re-ask is far cheaper than
        discarding the work.

        What each call cost is the gateway's own account to keep — LiteLLM
        already records every request against a key — so nothing here writes a
        second copy of it.
        """
        if not self.configured:
            raise LlmUnavailable("LLM_API_BASE / LLM_MODEL are not configured")
        if _breaker.open():
            raise LlmUnavailable(
                f"the model has not answered its last {_breaker.state()[0]} calls; "
                f"not asking again for {_breaker.wait_sec()}s"
            )

        # Thinking modes burn the token budget meant for the answer.
        extra = {"chat_template_kwargs": {"enable_thinking": False}}

        with _Slot():
            last_exc = None
            for base_url in self._candidate_urls():
                try:
                    client = self._client(base_url)
                    return self._exchange(client, messages, max_tokens, temperature, extra)
                except LlmUnavailable as exc:
                    last_exc = exc
                    if _looks_like_transport_error(str(exc)):
                        continue
                    _breaker.record_failure(exc)
                    raise
            if last_exc:
                # One failure for the request, however many routes it tried.
                _breaker.record_failure(last_exc)
                raise last_exc
            raise LlmUnavailable("No candidate endpoints available")

    def _exchange(self, client, messages, max_tokens, temperature, extra) -> dict:
        """One conversation with the model, holding one slot throughout."""
        try:
            reply = client.chat.completions.create(
                model=self.model, messages=messages, max_tokens=max_tokens,
                temperature=temperature, extra_body=extra,
            )
        except Exception as exc:  # transport, auth, rate limit, or unsupported extra_body
            # Only worth re-asking when the gateway answered and refused the
            # parameter. A call that timed out would simply time out again, at
            # full cost, for a reason that has nothing to do with extra_body.
            if extra and not _looks_like_transport_error(str(exc)):
                try:
                    reply = client.chat.completions.create(
                        model=self.model, messages=messages, max_tokens=max_tokens,
                        temperature=temperature,
                    )
                except Exception as inner_exc:
                    raise LlmUnavailable(str(inner_exc)) from inner_exc
            else:
                raise LlmUnavailable(str(exc)) from exc
        _breaker.record_success()

        msg = reply.choices[0].message
        text = msg.content or getattr(msg, "reasoning_content", None) or ""
        try:
            parsed = json.loads(strip_to_json(text))
        except (ValueError, TypeError):
            parsed = None

        if parsed is not None:
            return parsed

        try:
            retry = client.chat.completions.create(
                model=self.model,
                messages=[
                    *messages,
                    {"role": "assistant", "content": text},
                    {"role": "user", "content": "Your previous reply was not valid JSON. "
                                                "Reply with strictly valid JSON, no prose, no fences."},
                ],
                max_tokens=max_tokens, temperature=0.0, extra_body=extra,
            )
            retry_msg = retry.choices[0].message
            retry_text = retry_msg.content or getattr(retry_msg, "reasoning_content", None) or ""
            parsed = json.loads(strip_to_json(retry_text))
        except Exception as exc:
            log.warning("llm returned unusable output: %s", text[:200])
            raise LlmUnavailable(f"model did not return JSON: {exc}") from exc

        return parsed


def _litellm_model_health(base: str, client: LlmClient) -> tuple[bool, str | None] | None:
    """(healthy, error) from LiteLLM's `GET /health?model=`, or None when the
    gateway has no such endpoint (not LiteLLM, or the key may not use it)."""
    import urllib.error
    import urllib.parse
    import urllib.request

    root = base.rstrip("/").removesuffix("/v1")
    req = urllib.request.Request(
        f"{root}/health?{urllib.parse.urlencode({'model': client.model})}",
        headers={"Authorization": f"Bearer {client.api_key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15.0) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        if e.code in (401, 403, 404, 405):
            return None
        return False, f"LiteLLM health check answered {e.code}"
    except Exception as e:  # noqa: BLE001 - the gateway answered /models a moment ago
        return False, str(e)
    if not isinstance(data, dict) or "healthy_endpoints" not in data:
        return None
    if data.get("healthy_endpoints"):
        return True, None
    bad = data.get("unhealthy_endpoints") or []
    error = next((str(b.get("error")) for b in bad if isinstance(b, dict) and b.get("error")), None)
    return False, (error or "LiteLLM reports no healthy endpoint for the model").splitlines()[0][:300]


def probe_health() -> dict:
    """Diagnostic probe to verify LLM gateway reachability and model responsiveness."""
    client = LlmClient.from_settings()
    if not client.configured:
        return {
            "ok": False,
            "status": "not_configured",
            "configured": False,
            "model": None,
            "endpoint": None,
            "gateway_latency_ms": None,
            "chat_latency_ms": None,
            "models_available": [],
            "error": "LLM_API_BASE or LLM_MODEL is not configured in .env",
            "message": "No model configured",
            "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    # 1. Gateway probe
    gateway_ok = False
    gateway_ms = None
    models_available = []
    gateway_err = None
    active_base = client.api_base

    import urllib.request

    for base in client._candidate_urls():
        t0 = time.perf_counter()
        try:
            url = base.rstrip("/")
            models_url = f"{url}/models" if url.endswith("/v1") else f"{url}/v1/models"
            req = urllib.request.Request(
                models_url,
                headers={"Authorization": f"Bearer {client.api_key}"},
            )
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                data = json.loads(resp.read().decode())
                gateway_ms = round((time.perf_counter() - t0) * 1000, 1)
                models_available = [
                    m.get("id") for m in data.get("data", []) if isinstance(m, dict) and "id" in m
                ]
                gateway_ok = True
                active_base = base
                break
        except Exception as e:
            gateway_err = str(e)

    if not gateway_ok:
        return {
            "ok": False,
            "status": "unreachable",
            "configured": True,
            "model": client.model,
            "endpoint": client.api_base,
            "gateway_latency_ms": None,
            "chat_latency_ms": None,
            "models_available": [],
            "error": f"Gateway unreachable: {gateway_err}",
            "message": "Could not connect to the model gateway",
            "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    # 2. The model itself: LiteLLM's own health check of it, else a tiny chat
    chat_ms = None
    chat_ok = False
    chat_err = None
    t1 = time.perf_counter()
    litellm = _litellm_model_health(active_base, client)
    if litellm is not None:
        chat_ok, chat_err = litellm
    else:
        try:
            ai = client._client(active_base)
            ai.chat.completions.create(
                model=client.model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=2,
                timeout=10.0,
            )
            chat_ok = True
        except Exception as e:
            chat_err = str(e)
    if chat_ok:
        chat_ms = round((time.perf_counter() - t1) * 1000, 1)
        _breaker.record_success()

    if chat_ok:
        return {
            "ok": True,
            "status": "healthy",
            "configured": True,
            "model": client.model,
            "endpoint": client.api_base,
            "gateway_latency_ms": gateway_ms,
            "chat_latency_ms": chat_ms,
            "models_available": models_available,
            "error": None,
            "message": f"Gateway reachable ({gateway_ms}ms) and model responding ({chat_ms}ms)",
            "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    else:
        return {
            "ok": False,
            "status": "degraded",
            "configured": True,
            "model": client.model,
            "endpoint": client.api_base,
            "gateway_latency_ms": gateway_ms,
            "chat_latency_ms": None,
            "models_available": models_available,
            "error": f"Model failed to answer test prompt: {chat_err}",
            "message": "Gateway answered, but the model failed to generate a reply",
            "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

