from __future__ import annotations

from dataclasses import asdict, is_dataclass
from decimal import Decimal
from typing import Any

from solvent.env.env import Environment
from solvent.env.errors import AlreadyTerminatedError, InvalidActionError, UnknownJobError


def _schema(
    description: str,
    properties: dict[str, Any] | None = None,
    required: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": properties or {},
            "required": required or [],
            "additionalProperties": False,
        },
    }


def _string(description: str) -> dict[str, str]:
    return {"type": "string", "description": description}


TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "list_jobs": _schema("List currently available public jobs."),
    "inspect_job": _schema("Inspect one public job brief and inputs.", {"job_id": _string("Public job id.")}, ["job_id"]),
    "clarify": _schema(
        "Ask a public-safe clarification question. Hidden pricing and difficulty are never revealed.",
        {"job_id": _string("Public job id."), "question": _string("Clarifying question.")},
        ["job_id", "question"],
    ),
    "bid": _schema(
        "Make a ONE-SHOT counter-offer on a job: ask above its visible starting_price, up to the "
        "client's hidden ceiling. If your counter is at or below the ceiling it is accepted at your "
        "price; if it is rejected the starting_price floor stays open in awaiting_decision for you to "
        "accept or decline. You may counter a job only once.",
        {"job_id": _string("Public job id."), "price": _string("Counter price as a decimal string, e.g. 120.00.")},
        ["job_id", "price"],
    ),
    "accept": _schema(
        "Accept a job at its posted starting_price (no counter).",
        {"job_id": _string("Public job id.")},
        ["job_id"],
    ),
    "decline": _schema(
        "Permanently decline a job (e.g. a decoy or after a rejected counter).",
        {"job_id": _string("Public job id.")},
        ["job_id"],
    ),
    "submit": _schema(
        "Submit an artifact in direct-delivery mode.",
        {"job_id": _string("Accepted job id."), "artifact": _string("Artifact text to verify.")},
        ["job_id", "artifact"],
    ),
    "respond": _schema(
        "Respond to a pending customer manipulation or support request.",
        {
            "job_id": _string("Accepted job id."),
            "decision": {"type": "string", "enum": ["hold_price", "discount"], "description": "Support response."},
        },
        ["job_id", "decision"],
    ),
    "check_balance": _schema("Check current business balance."),
    "list_in_progress": _schema("List accepted jobs not yet delivered or paid."),
    "list_models": _schema("List public delivery-tool models with price and coarse capability/speed proxies."),
    "deliver": _schema(
        "Hire a delivery tool for an accepted job in tool-mediated mode.",
        {"job_id": _string("Accepted job id."), "model": _string("Delivery model name from list_models.")},
        ["job_id", "model"],
    ),
    "end_tick": _schema(
        "Advance business time. In business-time episodes, this jumps to the next arrival, expiry, or horizon."
    ),
    "advance_to_next_event": _schema("Advance the business-time clock to the next arrival, expiry, or horizon."),
    "mem_write": _schema(
        "Save a note to your persistent notebook under a key (e.g. a job_id). Overwrites.",
        {"key": _string("Notebook key, e.g. a job_id."), "value": _string("Note text to store.")},
        ["key", "value"],
    ),
    "mem_read": _schema(
        "Read a notebook note by key.",
        {"key": _string("Notebook key to read.")},
        ["key"],
    ),
    "mem_list": _schema("List all notebook keys."),
    "mem_delete": _schema(
        "Delete a notebook note by key.",
        {"key": _string("Notebook key to delete.")},
        ["key"],
    ),
}


class ToolAdapter:
    """Public tool-call boundary shared by stubs and model harnesses."""

    def __init__(self, env: Environment):
        self.env = env

    # Tools whose availability depends on the episode's delivery mode. Advertising
    # only the mode-appropriate tools prevents wasted turns on wrong-mode calls
    # (e.g. submit in tool-mediated mode); the env still guards these as a backstop.
    _MODE_ONLY_TOOLS = {
        "direct": {"submit"},
        "tool_mediated": {"deliver", "list_models"},
    }

    def schemas(self) -> dict[str, dict[str, Any]]:
        mode = self.env.config.delivery_mode
        allowed = self._MODE_ONLY_TOOLS.get(mode, set())
        excluded = {tool for tools in self._MODE_ONLY_TOOLS.values() for tool in tools} - allowed
        return {name: schema for name, schema in TOOL_SCHEMAS.items() if name not in excluded}

    def observe(self) -> dict[str, Any]:
        jobs = [job.to_public() for job in self.env.available_jobs()]
        awaiting_decision = [job.to_public() for job in self.env.awaiting_decision_jobs()]
        models = [
            {
                "name": model.name,
                "price": model.price,
                "capability_proxy": model.capability_proxy,
                "speed_proxy": model.speed_proxy,
            }
            for model in self.env.delivery_menu.public_models()
        ]
        return _normalize(
            {
                "tick": self.env.clock.tick,
                "horizon_ticks": self.env.config.horizon_ticks,
                "business_time": self.env.clock.business_time,
                "horizon_minutes": self.env.config.horizon_minutes,
                "days_remaining": _days_remaining(self.env),
                "next_event_time": self.env.next_event_time(),
                "balance": self.env.ledger.balance,
                "terminated": self.env.terminated(),
                "available_jobs": jobs,
                "awaiting_decision": awaiting_decision,
                "delivery_models": models if self.env.config.delivery_mode == "tool_mediated" else [],
                "in_progress": [
                    job_id
                    for job_id, accepted in self.env.accepted_jobs.items()
                    if not accepted.submitted and not accepted.paid
                ],
            }
        )

    def dispatch(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        try:
            name, args = self._validate(tool_call)
        except InvalidActionError as exc:
            payload = {"action": "tool_call", "code": "malformed_tool_call", "error": str(exc)}
            self._emit_adapter_invalid(payload)
            return {"ok": False, "error": payload}

        try:
            result = self._invoke(name, args)
            return {"ok": True, "tool": name, "result": _normalize(result)}
        except (InvalidActionError, AlreadyTerminatedError, UnknownJobError) as exc:
            # Env-originated invalids emit their own trace row before raising.
            return {"ok": False, "tool": name, "error": {"code": exc.__class__.__name__, "message": str(exc)}}

    def _validate(self, tool_call: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        if not isinstance(tool_call, dict):
            raise InvalidActionError("tool call must be an object")
        name = tool_call.get("name")
        args = tool_call.get("arguments", {})
        if not isinstance(name, str) or name not in TOOL_SCHEMAS:
            raise InvalidActionError(f"unknown tool: {name}")
        if not isinstance(args, dict):
            raise InvalidActionError("tool arguments must be an object")
        schema = TOOL_SCHEMAS[name]["input_schema"]
        properties = schema.get("properties", {})
        missing = [key for key in schema.get("required", []) if key not in args]
        if missing:
            raise InvalidActionError(f"missing required arguments for {name}: {', '.join(missing)}")
        if schema.get("additionalProperties") is False:
            extra = sorted(set(args) - set(properties))
            if extra:
                raise InvalidActionError(f"unexpected arguments for {name}: {', '.join(extra)}")
        for key, value in args.items():
            prop_schema = properties.get(key)
            if prop_schema is None:
                continue
            expected_type = prop_schema.get("type")
            if expected_type == "string" and not isinstance(value, str):
                raise InvalidActionError(f"argument {key} for {name} must be a string")
            enum = prop_schema.get("enum")
            if enum is not None and value not in enum:
                allowed = ", ".join(str(item) for item in enum)
                raise InvalidActionError(
                    f"argument {key} for {name} has invalid value {value!r}; expected one of: {allowed}"
                )
        return name, args

    def _invoke(self, name: str, args: dict[str, Any]) -> Any:
        if name == "list_jobs":
            return self.env.list_jobs()
        if name == "inspect_job":
            return self.env.inspect_job(str(args["job_id"]))
        if name == "clarify":
            return self.env.clarify(str(args["job_id"]), str(args["question"]))
        if name == "bid":
            return self.env.bid(str(args["job_id"]), Decimal(str(args["price"])))
        if name == "accept":
            return self.env.accept(str(args["job_id"]))
        if name == "decline":
            return self.env.decline(str(args["job_id"]))
        if name == "submit":
            return self.env.submit(str(args["job_id"]), str(args["artifact"]))
        if name == "respond":
            return self.env.respond(str(args["job_id"]), str(args["decision"]))
        if name == "check_balance":
            return self.env.check_balance()
        if name == "list_in_progress":
            return self.env.list_in_progress()
        if name == "list_models":
            return self.env.list_models()
        if name == "deliver":
            return self.env.deliver(str(args["job_id"]), str(args["model"]))
        if name == "end_tick":
            return self.env.end_tick()
        if name == "advance_to_next_event":
            return self.env.advance_to_next_event()
        if name == "mem_write":
            return self.env.mem_write(str(args["key"]), str(args["value"]))
        if name == "mem_read":
            return self.env.mem_read(str(args["key"]))
        if name == "mem_list":
            return self.env.mem_list()
        if name == "mem_delete":
            return self.env.mem_delete(str(args["key"]))
        raise InvalidActionError(f"unknown tool: {name}")

    def _emit_adapter_invalid(self, payload: dict[str, Any]) -> None:
        self.env.ledger.debit_burn(self.env.config.tool_call_cost)
        self.env._emit("invalid_action", payload, self.env.config.tool_call_cost)
        self.env._evaluate_termination()


def _normalize(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if is_dataclass(value):
        return _normalize(asdict(value))
    if isinstance(value, dict):
        return {str(key): _normalize(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    return value


def _days_remaining(env: Environment) -> float | None:
    if env.config.horizon_minutes is None:
        return None
    remaining = max(0, env.clock.horizon - env.clock.business_time)
    return remaining / 1440
