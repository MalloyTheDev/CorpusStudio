"""Read-only Hugging Face Hub dataset import (dependency-light).

Fetches rows from the public Hugging Face **datasets-server** JSON API with the
standard library only (the same urllib pattern the model backends use), so no
``datasets`` / ``huggingface_hub`` / pyarrow dependency is pulled in. Import is
read-only and public-dataset-only: no auth, no upload, no publishing — the
engine never touches the cloud on its own, and it never writes ``examples.jsonl``
(the desktop remains the single writer; imported rows land in a *staging* file
that flows through the existing import-preview / quarantine path).

Imported data is NOT assumed to be licensed for training — the dataset's license
is surfaced with a caveat so the user reviews it before training use.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen

from pydantic import BaseModel, Field

from corpus_studio.model_backends.retry import RetryPolicy, call_with_retry
from corpus_studio.schemas.base import DatasetSchema

UrlOpen = Callable[..., Any]

DATASETS_SERVER = "https://datasets-server.huggingface.co"
HUB_API = "https://huggingface.co/api"

# The datasets-server caps a single /rows page at 100 rows.
_MAX_PAGE = 100

LICENSE_CAVEAT = (
    "Imported rows are NOT assumed to be licensed for training. Review the "
    "dataset's license and terms before using this data to train a model."
)


class HfConfigSplit(BaseModel):
    config: str
    split: str


class HfDatasetInspection(BaseModel):
    dataset_id: str
    viewable: bool
    gated: bool = False
    license: str | None = None
    license_note: str
    configs_splits: list[HfConfigSplit] = Field(default_factory=list)
    # Columns of the first split, so the caller can map them to a schema.
    sample_columns: list[str] = Field(default_factory=list)


class HfRowsPage(BaseModel):
    dataset_id: str
    config: str
    split: str
    num_rows_total: int = 0
    columns: list[str] = Field(default_factory=list)
    rows: list[dict[str, Any]] = Field(default_factory=list)


class HfImportResult(BaseModel):
    dataset_id: str
    config: str
    split: str
    schema_id: str
    fetched_rows: int
    mapping: dict[str, str] = Field(default_factory=dict)
    unmapped_schema_fields: list[str] = Field(default_factory=list)
    unused_columns: list[str] = Field(default_factory=list)
    license: str | None = None
    license_note: str
    out_path: str | None = None


def _get_json(
    url: str,
    opener: UrlOpen | None = None,
    retry_policy: RetryPolicy | None = None,
) -> Any:
    """GET a URL and parse JSON, retrying transient failures (429/5xx/network).

    The datasets-server can return a transient 5xx while it materializes a
    dataset, so the shared backoff policy applies here too. ``opener`` is
    resolved at call time (default ``urlopen``) so tests can inject a fake.
    """
    resolved = opener or urlopen

    def _do() -> Any:
        request = Request(url, headers={"User-Agent": "corpus-studio"})
        with resolved(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))

    return call_with_retry(_do, retry_policy or RetryPolicy())


def _license_note(license_id: str | None, gated: bool, lookup_ok: bool = True) -> str:
    if license_id:
        prefix = f"License: {license_id}."
    elif lookup_ok:
        prefix = "License: not declared by the dataset."
    else:
        # Distinct from "not declared": we could not reach the metadata, so a
        # missing id here is our failure, not evidence the dataset is unlicensed.
        prefix = "License: could not be verified (metadata lookup failed) — check the dataset page."
    if gated:
        prefix = f"{prefix} This dataset is GATED (requires access approval)."
    return f"{prefix} {LICENSE_CAVEAT}"


def inspect_dataset(dataset_id: str, opener: UrlOpen | None = None) -> HfDatasetInspection:
    """List a dataset's configs/splits + license so columns can be mapped.

    Does not download rows beyond a single sample page used to read the column
    names of the first split.
    """
    ds = quote(dataset_id, safe="/")

    valid = _get_json(f"{DATASETS_SERVER}/is-valid?dataset={ds}", opener)
    viewable = bool(valid.get("viewer") or valid.get("preview"))

    # License + gated flag come from the hub metadata API, not datasets-server.
    # It's a single cheap call but important (the license drives the training
    # caveat), so it retries a little harder; if it still fails we say so
    # honestly rather than implying the dataset is unlicensed.
    license_id: str | None = None
    gated = False
    license_lookup_ok = True
    try:
        meta = _get_json(f"{HUB_API}/datasets/{ds}", opener, RetryPolicy(max_attempts=4))
        gated = bool(meta.get("gated"))
        card = meta.get("cardData") or {}
        raw_license = card.get("license")
        if isinstance(raw_license, list):
            license_id = ", ".join(str(item) for item in raw_license) or None
        elif raw_license:
            license_id = str(raw_license)
    except (OSError, ValueError):
        license_lookup_ok = False

    configs_splits: list[HfConfigSplit] = []
    if viewable:
        splits = _get_json(f"{DATASETS_SERVER}/splits?dataset={ds}", opener)
        for entry in splits.get("splits", []):
            if isinstance(entry, dict) and entry.get("config") and entry.get("split"):
                configs_splits.append(
                    HfConfigSplit(config=str(entry["config"]), split=str(entry["split"]))
                )

    sample_columns: list[str] = []
    if configs_splits:
        first = configs_splits[0]
        page = fetch_rows_page(dataset_id, first.config, first.split, 0, 1, opener)
        sample_columns = page.columns

    return HfDatasetInspection(
        dataset_id=dataset_id,
        viewable=viewable,
        gated=gated,
        license=license_id,
        license_note=_license_note(license_id, gated, license_lookup_ok),
        configs_splits=configs_splits,
        sample_columns=sample_columns,
    )


def fetch_rows_page(
    dataset_id: str,
    config: str,
    split: str,
    offset: int,
    length: int,
    opener: UrlOpen | None = None,
) -> HfRowsPage:
    """Fetch a single page of rows (length clamped to the datasets-server cap)."""
    ds = quote(dataset_id, safe="/")
    cfg = quote(config, safe="")
    spl = quote(split, safe="")
    length = max(0, min(length, _MAX_PAGE))
    url = (
        f"{DATASETS_SERVER}/rows?dataset={ds}&config={cfg}&split={spl}"
        f"&offset={offset}&length={length}"
    )
    payload = _get_json(url, opener)
    columns = [
        str(feature["name"])
        for feature in payload.get("features", [])
        if isinstance(feature, dict) and feature.get("name")
    ]
    rows = [
        entry["row"]
        for entry in payload.get("rows", [])
        if isinstance(entry, dict) and isinstance(entry.get("row"), dict)
    ]
    return HfRowsPage(
        dataset_id=dataset_id,
        config=config,
        split=split,
        num_rows_total=int(payload.get("num_rows_total") or 0),
        columns=columns,
        rows=rows,
    )


def fetch_rows(
    dataset_id: str,
    config: str,
    split: str,
    limit: int,
    opener: UrlOpen | None = None,
) -> HfRowsPage:
    """Fetch up to ``limit`` rows, paginating across the datasets-server cap."""
    collected: list[dict[str, Any]] = []
    columns: list[str] = []
    total = 0
    offset = 0
    while len(collected) < limit:
        page = fetch_rows_page(
            dataset_id, config, split, offset, min(_MAX_PAGE, limit - len(collected)), opener
        )
        total = page.num_rows_total
        if not columns:
            columns = page.columns
        if not page.rows:
            break  # reached the end of the split
        collected.extend(page.rows)
        offset += len(page.rows)
        if offset >= total:
            break
    return HfRowsPage(
        dataset_id=dataset_id,
        config=config,
        split=split,
        num_rows_total=total,
        columns=columns,
        rows=collected[:limit],
    )


def suggest_mapping(columns: list[str], schema: DatasetSchema) -> dict[str, str]:
    """Auto-map schema fields to HF columns by exact (case-insensitive) name."""
    by_lower = {column.lower(): column for column in columns}
    mapping: dict[str, str] = {}
    for field in schema.fields:
        match = by_lower.get(field.name.lower())
        if match is not None:
            mapping[field.name] = match
    return mapping


def map_rows(rows: list[dict[str, Any]], mapping: dict[str, str]) -> list[dict[str, Any]]:
    """Project HF rows onto schema-field-keyed rows using ``mapping``.

    Only mapped fields are emitted; a column missing from a given row yields no
    key for that field (validation downstream flags a required field that ends
    up absent).
    """
    mapped: list[dict[str, Any]] = []
    for row in rows:
        projected: dict[str, Any] = {}
        for field_name, column in mapping.items():
            if column in row:
                projected[field_name] = row[column]
        mapped.append(projected)
    return mapped
