"""The deterministic claim workflow (LangGraph). Code picks every next step; there is no LLM
supervisor.

guardrails -> intake -> [coverage || fraud] -> adjudicator -> critic -> judge -> router
                             ^--- retry (<= max_retries) with issues ---'    |-> auto_decide
                                                                             '-> human_review
human_review -> memory. Invalid input: guardrails -> reject_input.
"""

import time
from collections.abc import Callable
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Checkpointer, interrupt
from pydantic import BaseModel, Field

from autoclaim.core.audit import AuditEvent, AuditSink
from autoclaim.core.budget import BudgetConfig, BudgetTrippedError, check_budget
from autoclaim.core.decision import HumanDecision
from autoclaim.core.finalize import FinalizationLedger
from autoclaim.core.lob import LineOfBusiness, MemoryStore, NodeResult, NoMemory
from autoclaim.core.router import RouterConfig, route_claim
from autoclaim.core.state import ClaimState


class HarnessConfig(BaseModel):
    max_retries: int = Field(ge=0)
    router: RouterConfig
    budget: BudgetConfig


NodeFn = Callable[[ClaimState], NodeResult]


class Harness:
    def __init__(
        self,
        lob: LineOfBusiness,
        cfg: HarnessConfig,
        audit: AuditSink,
        ledger: FinalizationLedger,
        memory: MemoryStore | None = None,
    ) -> None:
        self.lob = lob
        self.cfg = cfg
        self.audit = audit
        self.ledger = ledger
        self.memory = memory if memory is not None else NoMemory()  # an empty memory is falsy

    # ------------------------------------------------------------ node plumbing

    def _event(self, state: ClaimState, node: str, **kw: Any) -> dict[str, Any]:
        event = AuditEvent(claim_id=state["claim_id"], node=node, **kw)
        self.audit.write(event)
        return event.model_dump()

    def wrap(self, name: str, fn: NodeFn) -> Callable[[ClaimState], dict[str, Any]]:
        """Skip after a failsafe; turn any failure into a failsafe; audit + budget every run."""

        def node(state: ClaimState) -> dict[str, Any]:
            if state.get("failsafe"):
                return {"audit": [self._event(state, name, status="skipped")]}
            start = time.perf_counter()
            try:
                res = fn(state)
            except Exception as exc:  # fail safe: never let a broken node produce a decision
                reason = f"{name}: {type(exc).__name__}: {str(exc)[:300]}"
                event = self._event(state, name, status="failsafe", outputs={"error": reason})
                return {"failsafe": reason, "audit": [event]}
            calls = len(res.calls)
            tokens = sum(c.prompt_tokens + c.completion_tokens for c in res.calls if not c.cached)
            update: dict[str, Any] = dict(res.update)
            status = "ok"
            try:
                check_budget(
                    state.get("llm_calls", 0) + calls,
                    state.get("tokens", 0) + tokens,
                    self.cfg.budget,
                )
            except BudgetTrippedError as exc:
                update["failsafe"] = f"budget: {exc}"
                status = "failsafe"
            errors = [e for c in res.calls for e in c.errors]  # fallbacks and why
            outputs = res.outputs | ({"llm_errors": [e[:200] for e in errors]} if errors else {})
            event = self._event(
                state,
                name,
                status=status,
                outputs=outputs,
                model=",".join(sorted({c.model for c in res.calls if c.model})) or None,
                prompt_tokens=sum(c.prompt_tokens for c in res.calls),
                completion_tokens=sum(c.completion_tokens for c in res.calls),
                latency_s=round(time.perf_counter() - start, 4),
                cached=bool(res.calls) and all(c.cached for c in res.calls),
            )
            return {**update, "llm_calls": calls, "tokens": tokens, "audit": [event]}

        return node

    # ------------------------------------------------------------ core nodes

    def _retry(self, state: ClaimState) -> dict[str, Any]:
        issues = [
            *state.get("critic", {}).get("issues", []),
            *state.get("judge", {}).get("issues", []),
        ]
        retries = state.get("retries", 0) + 1
        event = self._event(state, "retry", outputs={"retry": retries, "issues": len(issues)})
        return {"retries": retries, "issues": issues, "judge": {}, "audit": [event]}

    def _checks_passed(self, state: ClaimState) -> bool:
        return bool(state.get("critic", {}).get("passed")) and bool(
            state.get("judge", {}).get("passed")
        )

    def _router(self, state: ClaimState) -> dict[str, Any]:
        decision = state.get("decision") or None
        rd = route_claim(
            decision,
            [] if state.get("failsafe") else self.lob.hard_escalations(state),
            self.lob.fraud_score(state),
            self._checks_passed(state),
            state.get("failsafe"),
            self.cfg.router,
        )
        event = self._event(state, "router", outputs=rd.model_dump())
        return {"route": rd.model_dump(), "audit": [event]}

    def _auto_decide(self, state: ClaimState) -> dict[str, Any]:
        payload = {"decided_by": "auto", **state["decision"]}
        stored, created = self.ledger.finalize(state["claim_id"], payload)
        final = {**stored, "duplicate": not created}
        event = self._event(state, "auto_decide", outputs={"outcome": stored["outcome"],
                                                           "duplicate": not created})  # fmt: skip
        return {"final": final, "audit": [event]}

    def _human_review(self, state: ClaimState) -> dict[str, Any]:
        already = self.ledger.get(state["claim_id"])
        if already is not None:  # resumed or re-run after finalization: never decide twice
            return {"final": {**already, "duplicate": True}}
        request = {
            "claim_id": state["claim_id"],
            "route_reasons": state.get("route", {}).get("reasons", []),
            "proposed_decision": state.get("decision") or None,
            "case_summary": self.lob.case_summary(state),
        }
        human = HumanDecision.model_validate(interrupt(request))  # pauses here until resumed
        payload = {"decided_by": "human", **human.model_dump()}
        stored, created = self.ledger.finalize(state["claim_id"], payload)
        outputs = {"outcome": human.outcome, "adjuster": human.adjuster}
        event = self._event(state, "human_review", outputs=outputs)
        final = {**stored, "duplicate": not created}
        return {"human": human.model_dump(), "final": final, "audit": [event]}

    def _memory(self, state: ClaimState) -> dict[str, Any]:
        self.memory.remember(state)
        return {"audit": [self._event(state, "memory")]}

    def _reject(self, state: ClaimState) -> dict[str, Any]:
        final = {"decided_by": "guardrails", "outcome": "rejected_input",
                 "reasons": state["guardrails"].get("errors", [])}  # fmt: skip
        return {"final": final, "audit": [self._event(state, "reject_input", outputs=final)]}

    # ------------------------------------------------------------ edges

    def _after_guardrails(self, state: ClaimState) -> str:
        if state.get("failsafe"):
            return "router"
        return "reject_input" if state.get("guardrails", {}).get("rejected") else "intake"

    def _after_check(self, key: str, passed_next: str) -> Callable[[ClaimState], str]:
        def edge(state: ClaimState) -> str:
            if state.get("failsafe"):
                return "router"
            check: dict[str, Any] = state.get(key) or {}  # type: ignore[assignment]
            if check.get("passed"):
                return passed_next
            return "retry" if state.get("retries", 0) < self.cfg.max_retries else "router"

        return edge

    def build(self, checkpointer: Checkpointer = None) -> Any:
        g: Any = StateGraph(ClaimState)  # untyped: LangGraph's generics infer Never here
        lob = self.lob
        g.add_node("guardrails", self.wrap("guardrails", lob.guardrails))
        g.add_node("reject_input", self._reject)
        g.add_node("intake", self.wrap("intake", lob.intake))
        g.add_node("coverage", self.wrap("coverage", lob.coverage))
        g.add_node("fraud", self.wrap("fraud", lob.fraud))
        g.add_node("adjudicator", self.wrap("adjudicator", lob.adjudicate))
        g.add_node("critic", self.wrap("critic", lob.critic))
        g.add_node("judge", self.wrap("judge", lob.judge))
        g.add_node("retry", self._retry)
        g.add_node("router", self._router)
        g.add_node("auto_decide", self._auto_decide)
        g.add_node("human_review", self._human_review)
        g.add_node("memory", self._memory)

        g.add_edge(START, "guardrails")
        g.add_conditional_edges("guardrails", self._after_guardrails,
                                ["reject_input", "intake", "router"])  # fmt: skip
        g.add_edge("reject_input", END)
        g.add_edge("intake", "coverage")
        g.add_edge("intake", "fraud")
        g.add_edge(["coverage", "fraud"], "adjudicator")  # waits for both
        g.add_edge("adjudicator", "critic")
        g.add_conditional_edges("critic", self._after_check("critic", "judge"),
                                ["judge", "retry", "router"])  # fmt: skip
        g.add_conditional_edges("judge", self._after_check("judge", "router"),
                                ["router", "retry"])  # fmt: skip
        g.add_edge("retry", "adjudicator")
        g.add_conditional_edges("router", lambda s: s["route"]["route"],
                                ["auto_decide", "human_review"])  # fmt: skip
        g.add_edge("auto_decide", END)
        g.add_edge("human_review", "memory")
        g.add_edge("memory", END)
        return g.compile(checkpointer=checkpointer)
