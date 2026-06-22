from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from solvent.delivery.menu import DeliveryMenu
from solvent.env.pricing import PRICING_TABLE_VERSION
from solvent.experiment.config import ExperimentConfig
from solvent.experiment.estimate import ExperimentEstimate
from solvent.experiment.matrix import ExperimentCell

STATE_SCHEMA_VERSION = "solvent_experiment_state_v0_5"

PENDING = "pending"
RUNNING = "running"
COMPLETED = "completed"
FAILED = "failed"
SKIPPED_BUDGET = "skipped_budget"
FAILED_BUDGET = "failed_budget"
TERMINAL_STATUSES = {COMPLETED, FAILED, SKIPPED_BUDGET, FAILED_BUDGET}


@dataclass
class CellRecord:
    cell: ExperimentCell
    status: str = PENDING
    estimated_cost: Decimal = Decimal("0")
    provenance: dict[str, Any] = field(default_factory=dict)
    reserved_cost: Decimal = Decimal("0")
    actual_cost: Decimal = Decimal("0")
    trace_path: str = ""
    scorecard_path: str = ""
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CellRecord":
        return cls(
            cell=ExperimentCell(**data["cell"]),
            status=PENDING if data.get("status") == RUNNING else str(data.get("status", PENDING)),
            estimated_cost=Decimal(str(data.get("estimated_cost", "0"))),
            provenance=dict(data.get("provenance", {})),
            reserved_cost=Decimal("0") if data.get("status") == RUNNING else Decimal(str(data.get("reserved_cost", "0"))),
            actual_cost=Decimal(str(data.get("actual_cost", "0"))),
            trace_path=str(data.get("trace_path", "")),
            scorecard_path=str(data.get("scorecard_path", "")),
            started_at=data.get("started_at"),
            finished_at=data.get("finished_at"),
            error=data.get("error"),
        )


@dataclass
class ExperimentState:
    run_dir: Path
    manifest_path: Path
    ledger_path: Path
    records: dict[str, CellRecord] = field(default_factory=dict)

    @classmethod
    def load_or_create(
        cls,
        run_dir: Path,
        config: ExperimentConfig,
        estimate: ExperimentEstimate,
        cells: list[ExperimentCell],
    ) -> "ExperimentState":
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = run_dir / "manifest.json"
        ledger_path = run_dir / "ledger.json"
        if ledger_path.exists():
            payload = json.loads(ledger_path.read_text(encoding="utf-8"))
            records = {
                item["cell"]["cell_id"]: CellRecord.from_dict(item)
                for item in payload.get("cells", [])
            }
            state = cls(run_dir=run_dir, manifest_path=manifest_path, ledger_path=ledger_path, records=records)
            state._merge_new_cells(cells, _estimated_costs_by_model(estimate), _provenance_by_cell(config, cells))
            state.save(config, estimate)
            return state

        provenance = _provenance_by_cell(config, cells)
        records = {
            cell.cell_id: CellRecord(
                cell=cell,
                estimated_cost=_estimated_costs_by_model(estimate).get(cell.model, Decimal("0")),
                provenance=provenance[cell.cell_id],
            )
            for cell in cells
        }
        state = cls(run_dir=run_dir, manifest_path=manifest_path, ledger_path=ledger_path, records=records)
        state.save(config, estimate)
        return state

    @property
    def actual_spend(self) -> Decimal:
        return sum((record.actual_cost for record in self.records.values()), Decimal("0"))

    @property
    def reserved_spend(self) -> Decimal:
        return sum((record.reserved_cost for record in self.records.values()), Decimal("0"))

    def terminal_count(self) -> int:
        return sum(1 for record in self.records.values() if record.status in TERMINAL_STATUSES)

    def start(self, cell_id: str) -> None:
        record = self.records[cell_id]
        record.status = RUNNING
        record.reserved_cost = record.estimated_cost
        record.started_at = _now()
        record.finished_at = None
        record.error = None

    def complete(self, cell_id: str, actual_cost: Decimal, trace_path: Path, scorecard_path: Path) -> None:
        record = self.records[cell_id]
        record.status = COMPLETED
        record.reserved_cost = Decimal("0")
        record.actual_cost = actual_cost
        record.trace_path = str(trace_path)
        record.scorecard_path = str(scorecard_path)
        record.finished_at = _now()
        record.error = None

    def fail(self, cell_id: str, error: str, trace_path: Path | None = None) -> None:
        record = self.records[cell_id]
        record.status = FAILED
        record.reserved_cost = Decimal("0")
        if trace_path is not None:
            record.trace_path = str(trace_path)
        record.finished_at = _now()
        record.error = error

    def fail_budget(self, cell_id: str, actual_cost: Decimal, trace_path: Path, scorecard_path: Path, error: str) -> None:
        record = self.records[cell_id]
        record.status = FAILED_BUDGET
        record.reserved_cost = Decimal("0")
        record.actual_cost = actual_cost
        record.trace_path = str(trace_path)
        record.scorecard_path = str(scorecard_path)
        record.finished_at = _now()
        record.error = error

    def skip_budget(self, cell_id: str) -> None:
        record = self.records[cell_id]
        record.status = SKIPPED_BUDGET
        record.reserved_cost = Decimal("0")
        record.finished_at = _now()
        record.error = "estimated cell cost exceeds remaining budget"

    def save(self, config: ExperimentConfig, estimate: ExperimentEstimate) -> None:
        manifest = {
            "schema_version": STATE_SCHEMA_VERSION,
            "name": config.name,
            "run_dir": str(self.run_dir),
            "estimate": estimate.to_dict(),
            "created_or_updated_at": _now(),
        }
        ledger = {
            "schema_version": STATE_SCHEMA_VERSION,
            "name": config.name,
            "actual_spend": str(self.actual_spend),
            "reserved_spend": str(self.reserved_spend),
            "cells": [_normalize(record) for record in self.records.values()],
        }
        self.manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        self.ledger_path.write_text(json.dumps(ledger, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    def _merge_new_cells(
        self,
        cells: list[ExperimentCell],
        estimates: dict[str, Decimal],
        provenance: dict[str, dict[str, Any]],
    ) -> None:
        for cell in cells:
            if cell.cell_id not in self.records:
                self.records[cell.cell_id] = CellRecord(
                    cell=cell,
                    estimated_cost=estimates.get(cell.model, Decimal("0")),
                    provenance=provenance[cell.cell_id],
                )
            elif not self.records[cell.cell_id].provenance:
                self.records[cell.cell_id].provenance = provenance[cell.cell_id]
            else:
                for key in ("menu_version", "menu_checksum", "menu_schema_version"):
                    self.records[cell.cell_id].provenance.setdefault(key, provenance[cell.cell_id][key])


def _estimated_costs_by_model(estimate: ExperimentEstimate) -> dict[str, Decimal]:
    return {model.model: model.cost_per_cell for model in estimate.models}


def _provenance_by_cell(config: ExperimentConfig, cells: list[ExperimentCell]) -> dict[str, dict[str, Any]]:
    menu = DeliveryMenu.load_default()
    return {cell.cell_id: _cell_provenance(config, cell, menu) for cell in cells}


def _cell_provenance(config: ExperimentConfig, cell: ExperimentCell, menu: DeliveryMenu) -> dict[str, Any]:
    return {
        "model": cell.model_family,
        "model_config": cell.model,
        "config_id": cell.config_id,
        "ablation": cell.ablation,
        "condition": cell.condition,
        "redteam_enabled": cell.redteam_enabled,
        "seed": cell.seed,
        "sample_index": cell.sample_index,
        "pricing_table_version": PRICING_TABLE_VERSION,
        "menu_version": menu.version,
        "menu_checksum": menu.checksum,
        "menu_schema_version": menu.schema_version,
        "context_policy": config.context_policy,
        "ctx_window_tokens": config.ctx_window_tokens,
        "caching": config.caching,
        "horizon_minutes": config.horizon_minutes,
        "market": {
            "task_mix": config.market.task_mix,
            "arrival_rate_per_day": config.market.arrival_rate_per_day,
            "decoy_rate": config.market.decoy_rate,
            "manipulation_rate": config.market.manipulation_rate,
        },
        "temperature": config.temperature,
    }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _normalize(asdict(value))
    if isinstance(value, dict):
        return {str(key): _normalize(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    return value
