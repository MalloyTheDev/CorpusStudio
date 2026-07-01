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
