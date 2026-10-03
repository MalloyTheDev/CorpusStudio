from typing import Any, Literal

from pydantic import BaseModel, Field


FieldType = Literal[
    "string",
    "text",
    "markdown",
    "integer",
    "float",
    "boolean",
    "list",
    "object",
    "messages",
    "file_path",
    "image_path",
    "code",
]


class SchemaField(BaseModel):
    name: str
    type: FieldType
    required: bool = False
    description: str | None = None
    # Element type for ``list`` fields (e.g. a list of strings).
    item_type: FieldType | None = None
    # Allowed values for a scalar field (e.g. a fixed classification label set).
    enum: list[Any] | None = None
    # Inclusive numeric bounds for ``integer`` / ``float`` fields.
    minimum: float | None = None
    maximum: float | None = None
    # Nested field shapes for an ``object`` field, validated recursively.
    fields: list["SchemaField"] | None = None
    # Per-element object shape for a ``list`` of ``object`` elements (``item_type == "object"``),
    # validated recursively against each element. None = element type only, no per-element shape check.
    item_fields: list["SchemaField"] | None = None
    # OPT-IN existence check for a ``file_path`` / ``image_path`` field: when the schema declares
    # it AND the caller supplies an explicit root, a row naming a file that is not under that root
    # fails validation. Both halves are required on purpose. The schema opts in because only the
    # schema's author knows whether the paths are expected to resolve yet, and the root is passed
    # in rather than guessed from the project directory because a staging lane legitimately
    # references files that have not been fetched. Default False: a path stays a string-only check.
    path_must_exist: bool = False


class DatasetSchema(BaseModel):
    id: str
    name: str
    version: str
    description: str | None = None
    fields: list[SchemaField] = Field(default_factory=list)
    example: dict[str, Any] | None = None


class DatasetExample(BaseModel):
    schema_id: str
    fields: dict[str, Any]
    metadata: dict[str, Any] = Field(default_factory=dict)


# Resolve the self-referencing ``SchemaField.fields`` forward reference.
SchemaField.model_rebuild()
