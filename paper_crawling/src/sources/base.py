from __future__ import annotations

from .catalog import SourceDefinition


class ConfiguredSource:
    def __init__(self, definition: SourceDefinition) -> None:
        self.definition = definition
        self.key = definition.key
        self.display_name = definition.display_name
