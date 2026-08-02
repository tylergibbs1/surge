"""Pure adapters between immutable ledger records and public API schemas."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from surge import ledger, verification
from surge.api.schemas import (
    ForecastPoint,
    ForecastQuality,
    ForecastResponse,
    LedgerForecastSummary,
    LedgerRunResponse,
    LedgerScoreboardResponse,
    LedgerScoreboardRow,
    ScoreSummaryResponse,
)

DataState = Literal["fresh", "delayed", "stale", "unavailable"]


def run_response(record: ledger.ForecastRunRecord) -> LedgerRunResponse:
    return LedgerRunResponse(**record.model_dump(mode="json"))


def forecast_state(
    record: ledger.ForecastRecord,
    *,
    now: datetime | None = None,
) -> tuple[DataState, list[str]]:
    checked = (now or datetime.now(tz=UTC)).astimezone(UTC)
    issue_age_hours = (checked - record.issued_at_utc).total_seconds() / 3600.0
    reasons = list(record.warnings)
    if checked > record.last_valid_at_utc:
        reasons.append("forecast valid window has expired")
        return "stale", reasons
    if issue_age_hours > 36:
        reasons.append(f"forecast was issued {issue_age_hours:.1f} hours ago")
        return "stale", reasons
    if issue_age_hours > 26:
        reasons.append(f"forecast issuance is delayed ({issue_age_hours:.1f} hours old)")
        return "delayed", reasons
    if record.first_valid_at_utc <= checked <= record.last_valid_at_utc:
        remaining = sum(point.valid_at_utc > checked for point in record.points)
        if remaining < 18:
            reasons.append(f"only {remaining} future forecast hours remain")
            return "delayed", reasons
    return "fresh", reasons


def summary_from_record(record: ledger.ForecastRecord) -> LedgerForecastSummary:
    return LedgerForecastSummary(
        issuance_id=record.issuance_id,
        run_id=record.run_id,
        mode=record.mode.value,
        ba=record.ba,
        issued_at_utc=record.issued_at_utc,
        data_cutoff_utc=record.context_end_utc,
        first_valid_at_utc=record.first_valid_at_utc,
        last_valid_at_utc=record.last_valid_at_utc,
        model=record.model_name,
        model_revision=record.model_revision,
        model_artifact_sha256=record.model_artifact_sha256,
        feature_spec_version=record.feature_spec_version,
        availability_mode=record.availability_mode.value,
        point_estimate_kind=record.point_estimate_kind.value,
        horizon=record.horizon_hours,
        warnings=list(record.warnings),
    )


def response_from_record(
    record: ledger.ForecastRecord,
    *,
    now: datetime | None = None,
    committed: bool = True,
) -> ForecastResponse:
    state, reasons = forecast_state(record, now=now)
    published_at_utc: datetime | None = None
    if committed:
        try:
            marker = ledger.get_run(record.run_id)
        except KeyError:
            marker = None
        if marker is not None and marker.issuance_ids.get(record.ba) == record.issuance_id:
            published_at_utc = marker.published_at_utc
    return ForecastResponse(
        issuance_id=record.issuance_id,
        run_id=record.run_id,
        ba=record.ba,
        model=record.model_name,
        model_revision=record.model_revision,
        model_artifact_sha256=record.model_artifact_sha256,
        as_of_utc=record.issued_at_utc,
        generated_at_utc=record.issued_at_utc,
        issued_at_utc=record.issued_at_utc,
        data_cutoff_utc=record.context_end_utc,
        feature_cutoff_utc=record.feature_cutoff_utc,
        context_start_utc=record.context_start_utc,
        context_end_utc=record.context_end_utc,
        horizon=record.horizon_hours,
        units=record.units,
        feature_spec_version=record.feature_spec_version,
        feature_spec_sha256=record.feature_spec_sha256,
        feature_snapshot_sha256=record.feature_snapshot_sha256,
        availability_mode=record.availability_mode.value,
        point_estimate_kind=record.point_estimate_kind.value,
        mase_scale_24=record.mase_scale_24,
        code_revision=record.code_revision,
        committed=committed,
        run_published=published_at_utc is not None,
        published_at_utc=published_at_utc,
        warnings=list(record.warnings),
        quality=ForecastQuality(status=state, reasons=reasons),
        points=[
            ForecastPoint(
                ts_utc=point.valid_at_utc,
                mean_mw=point.mean_mw,
                median_mw=point.p50_mw,
                p10_mw=point.p10_mw,
                p90_mw=point.p90_mw,
                temp_c=point.future_temp_c,
            )
            for point in record.points
        ],
    )


def score_response(issuance_id: str) -> ScoreSummaryResponse | None:
    try:
        score = verification.score_forecast(issuance_id)
    except ValueError:
        return None
    return ScoreSummaryResponse(**score.model_dump())


def build_scoreboard(
    bas: list[str],
    *,
    now: datetime | None = None,
) -> LedgerScoreboardResponse:
    generated_at = (now or datetime.now(tz=UTC)).astimezone(UTC)
    rows: list[LedgerScoreboardRow] = []
    try:
        complete_run = ledger.latest_complete_run()
        records_by_ba = {
            record.ba: record for record in ledger.forecasts_for_run(complete_run.run_id)
        }
    except (KeyError, RuntimeError, ValueError):
        records_by_ba = {}

    for ba in bas:
        record = records_by_ba.get(ba)
        if record is None:
            rows.append(
                LedgerScoreboardRow(
                    ba=ba,
                    forecast=None,
                    score=None,
                    state="unavailable",
                    reasons=["no complete seven-RTO live run"],
                )
            )
            continue
        state, reasons = forecast_state(record, now=generated_at)
        rows.append(
            LedgerScoreboardRow(
                ba=ba,
                forecast=summary_from_record(record),
                score=score_response(record.issuance_id),
                state=state,
                reasons=reasons,
            )
        )

    available = sum(row.forecast is not None for row in rows)
    states = {row.state for row in rows}
    if available == 0:
        overall: DataState = "unavailable"
    elif states == {"fresh"}:
        overall = "fresh"
    elif "fresh" in states or "delayed" in states:
        overall = "delayed"
    else:
        overall = "stale"
    return LedgerScoreboardResponse(
        generated_at_utc=generated_at,
        expected_regions=len(bas),
        available_regions=available,
        state=overall,
        regions=rows,
    )
