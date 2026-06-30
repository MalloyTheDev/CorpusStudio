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


class DatasetSchema(BaseModel):
    id: str
    name: str
    version: str
    fields: list[SchemaField] = Field(default_factory=list)


class DatasetExample(BaseModel):
    schema_id: str
    fields: dict[str, Any]
    metadata: dict[str, Any] = Field(default_factory=dict)
