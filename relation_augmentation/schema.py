from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _as_type_set(value: Any) -> set[str]:
    if value == "*":
        return {"*"}
    if isinstance(value, str):
        return {value}
    if isinstance(value, (list, tuple, set)):
        return {str(item) for item in value}
    return set()


def _clean(value: Any) -> str:
    return " ".join(str(value or "").split())


@dataclass(frozen=True)
class RelationClaim:
    subject: str
    subject_type: str
    relation: str
    object: str
    object_type: str
    evidence_quote: str = ""
    confidence: float = 0.0
    qualifiers: dict[str, Any] = field(default_factory=dict)
    source: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: dict[str, Any], source: dict[str, Any]) -> "RelationClaim":
        return cls(
            subject=_clean(data.get("subject")),
            subject_type=_clean(data.get("subject_type")),
            relation=_clean(data.get("relation")).upper(),
            object=_clean(data.get("object")),
            object_type=_clean(data.get("object_type")),
            evidence_quote=_clean(data.get("evidence_quote")),
            confidence=_as_float(data.get("confidence")),
            qualifiers=data.get("qualifiers") if isinstance(data.get("qualifiers"), dict) else {},
            source=source,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "subject_type": self.subject_type,
            "relation": self.relation,
            "object": self.object,
            "object_type": self.object_type,
            "evidence_quote": self.evidence_quote,
            "confidence": self.confidence,
            "qualifiers": self.qualifiers,
            "source": self.source,
        }

    def dedupe_key(self) -> tuple[str, str, str, str, str]:
        return (
            self.subject.lower(),
            self.subject_type,
            self.relation,
            self.object.lower(),
            self.object_type,
        )


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


class RelationSchema:
    def __init__(self, entity_types: dict[str, str], relation_types: dict[str, dict[str, Any]]) -> None:
        self.entity_types = dict(entity_types)
        self.relation_types = {name.upper(): dict(spec) for name, spec in relation_types.items()}

    def relation_names(self) -> list[str]:
        return sorted(self.relation_types)

    def validate(self, claim: RelationClaim) -> list[str]:
        errors = []
        if not claim.subject:
            errors.append("missing subject")
        if not claim.object:
            errors.append("missing object")
        if claim.subject_type not in self.entity_types:
            errors.append(f"unknown subject_type: {claim.subject_type}")
        if claim.object_type not in self.entity_types:
            errors.append(f"unknown object_type: {claim.object_type}")
        relation_spec = self.relation_types.get(claim.relation)
        if relation_spec is None:
            errors.append(f"unknown relation: {claim.relation}")
            return errors
        if not self._type_allowed(claim.subject_type, relation_spec.get("domain")):
            errors.append(f"domain mismatch: {claim.subject_type} cannot use {claim.relation}")
        if not self._type_allowed(claim.object_type, relation_spec.get("range")):
            errors.append(f"range mismatch: {claim.relation} cannot target {claim.object_type}")
        return errors

    def prompt_text(self) -> str:
        lines = ["Entity types:"]
        for name, description in self.entity_types.items():
            lines.append(f"- {name}: {description}")
        lines.append("")
        lines.append("Relation types:")
        for name, spec in self.relation_types.items():
            domain = spec.get("domain")
            range_ = spec.get("range")
            description = spec.get("description", "")
            parent = spec.get("sub_property_of")
            parent_text = f"; subPropertyOf={parent}" if parent else ""
            lines.append(f"- {name}: domain={domain}; range={range_}; {description}{parent_text}")
        return "\n".join(lines)

    @staticmethod
    def _type_allowed(actual: str, allowed: Any) -> bool:
        allowed_types = _as_type_set(allowed)
        return "*" in allowed_types or actual in allowed_types
