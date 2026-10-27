"""Thin adapters; the existing agent remains responsible for execution."""
from __future__ import annotations

import asyncio
import time

from dome_agent.agent import DomeAgent
from dome_agent.config import redact_mapping
from dome_agent.logging import AgentLog

from .service import WebError, iso


class BrowserLog(AgentLog):
    def __init__(self, *args, callback, **kwargs):
        self.callback = callback
        super().__init__(*args, **kwargs)

    def write(self, kind, *, ref=None, dome_time=None, **fields):
        seq = super().write(kind, ref=ref, dome_time=dome_time, **fields)
        record = {"ts": iso(), "run_id": self.run_id, "seq": seq, "kind": kind, **fields}
        if ref is not None:
            record["ref"] = ref
        self.callback(redact_mapping(record, extra_secrets=self._secrets))
        return seq


class BrowserAgent(DomeAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.web_replies = asyncio.Queue()

    async def web_reply(self, reply, signature):
        future = asyncio.get_running_loop().create_future()
        await self.web_replies.put((reply, tuple(signature), future))
        return await asyncio.wait_for(asyncio.shield(future), timeout=20)

    async def _handle_pending(self):
        while not self.web_replies.empty():
            reply, signature, future = self.web_replies.get_nowait()
            try:
                pending = self._pending
                if pending is None or pending.signature != signature or pending.expired(time.time()):
                    raise WebError("Предложение изменилось или устарело", 409)
                if self._role != "team":
                    raise WebError("Нет прав активной команды", 403)
                if reply == "cancel":
                    self._pending = None
                    self.log.write("decision", situation="Предложение отменено из панели", cause="Запрос пользователя", confidence=1, evidence=[], plan=[], source="rules")
                else:
                    await self._execute_pending(force=reply == "force")
                if not future.done():
                    future.set_result({"ok": True})
            except Exception as exc:
                if not future.done():
                    future.set_exception(exc)
            finally:
                self.web_replies.task_done()
        await super()._handle_pending()
