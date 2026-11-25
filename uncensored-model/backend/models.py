"""Validated, restart-to-reload local model registry. No database migrations."""
import json
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field, model_validator

MAX_PROMPT_BYTES = 128 * 1024


class GenerationOptions(BaseModel):
    model_config = ConfigDict(extra='forbid')
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, gt=0, le=1)
    num_ctx: int = Field(default=4096, ge=4096, le=65536)
    num_predict: int | None = Field(default=None, ge=1)


class TextModel(BaseModel):
    model_config = ConfigDict(extra='forbid')
    id: str = Field(pattern=r'^[a-z0-9-]+$')
    displayName: str = Field(min_length=1)
    ollamaModel: str = Field(min_length=1)
    think: bool | None = None
    options: GenerationOptions = Field(default_factory=GenerationOptions)

    @model_validator(mode='after')
    def local_tag(self):
        if '://' in self.ollamaModel or 'cloud' in self.ollamaModel.lower() or any(c.isspace() for c in self.ollamaModel):
            raise ValueError('Use a local Ollama model tag, never a URL or cloud model')
        return self


class Registry(BaseModel):
    default_model: str
    system_instruction: str | None = Field(default=None, min_length=1)
    models: list[TextModel] = Field(min_length=1)

    @model_validator(mode='after')
    def valid_ids(self):
        ids = [model.id for model in self.models]
        if len(ids) != len(set(ids)) or self.default_model not in ids:
            raise ValueError('Model IDs must be unique and include default_model')
        return self

    def get(self, model_id=None):
        requested = model_id or self.default_model
        for model in self.models:
            if model.id == requested:
                return model
        raise ValueError('Unknown model ID')


REGISTRY = Registry.model_validate(json.loads(
    (Path(__file__).resolve().parent.parent / 'models.json').read_text(encoding='utf-8')))
