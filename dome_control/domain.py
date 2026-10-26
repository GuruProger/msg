"""Pure catalog, validation and telemetry helpers shared by CLI and agent API."""

from __future__ import annotations

from typing import Any


class DomeValidationError(ValueError):
    """Raised when a requested node/action/query is not allowed."""


def filter_catalog(
    catalog: list[dict[str, Any]],
    *,
    system: str | None = None,
    search: str | None = None,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    system_value = (system or "").casefold().strip()
    search_value = (search or "").casefold().strip()
    for node in catalog:
        if not isinstance(node, dict):
            continue
        if system_value:
            fields = {
                str(node.get("system", "")).casefold(),
                str(node.get("category", "")).casefold(),
            }
            if system_value not in fields:
                continue
        if search_value:
            haystack = " ".join(
                str(node.get(key, ""))
                for key in ("node_id", "code_name", "title", "system", "category")
            ).casefold()
            if search_value not in haystack:
                continue
        result.append(node)
    return result


def filter_nodes(
    nodes: dict[str, Any],
    catalog: list[dict[str, Any]],
    *,
    system: str | None = None,
    search: str | None = None,
) -> dict[str, Any]:
    if not system and not search:
        return nodes
    allowed = {
        str(node.get("node_id"))
        for node in filter_catalog(catalog, system=system, search=search)
        if node.get("node_id")
    }
    return {node_id: node for node_id, node in nodes.items() if node_id in allowed}


def find_node(catalog: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    target = name.strip().casefold()
    for node in catalog:
        if not isinstance(node, dict):
            continue
        if str(node.get("node_id", "")).casefold() == target:
            return node
        code_name = str(node.get("code_name", "")).casefold()
        if code_name and code_name == target:
            return node
    return None


def telemetry_node(data: dict[str, Any], name: str) -> tuple[str, dict[str, Any]] | None:
    nodes = data.get("nodes") or {}
    if name in nodes:
        return name, nodes[name]
    target = name.casefold()
    for node_id, node in nodes.items():
        if str(node_id).casefold() == target and isinstance(node, dict):
            return str(node_id), node
    return None


def allowed_control_names(node: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    controls = [str(value) for value in (node.get("controls") or [])]
    aliases = node.get("control_aliases") or {}
    alias_names: list[str] = []
    for values in aliases.values():
        alias_names.extend(str(value) for value in (values or []))
    service = [str(value) for value in (node.get("service_actions") or [])]
    return list(dict.fromkeys(controls)), list(dict.fromkeys(alias_names)), list(dict.fromkeys(service))


def resolve_control_action(
    node: dict[str, Any],
    action: str,
    *,
    force: bool = False,
) -> str:
    node_id = str(node.get("node_id"))
    controls, alias_names, service = allowed_control_names(node)
    operator = [str(value) for value in (node.get("operator_controls") or [])]
    target = action.casefold()

    if target in {value.casefold() for value in operator}:
        allowed = controls + alias_names + service
        raise DomeValidationError(
            f"Действие {action!r} для {node_id} доступно только Оператору "
            f"(operator_controls). Разрешённые действия агента: {allowed or 'нет'}"
        )

    control_map = {value.casefold(): value for value in controls}
    service_map = {value.casefold(): value for value in service}
    alias_map: dict[str, str] = {}
    for canonical, values in (node.get("control_aliases") or {}).items():
        alias_map[str(canonical).casefold()] = str(canonical)
        for alias in values or []:
            alias_map[str(alias).casefold()] = str(canonical)

    if target in control_map:
        return control_map[target]
    if target in alias_map:
        return alias_map[target]
    if target in service_map:
        return service_map[target]

    if force:
        return action

    allowed = controls + alias_names + service
    raise DomeValidationError(
        f"Действие {action!r} для {node_id} недоступно. Разрешено: {allowed or 'нет'}"
    )


def resolve_query_name(node: dict[str, Any], query: str, *, force: bool = False) -> str:
    queries = [str(value) for value in (node.get("queries") or [])]
    target = query.casefold()
    for value in queries:
        if value.casefold() == target:
            return value
    if force:
        return query
    raise DomeValidationError(
        f"Запрос {query!r} для {node.get('node_id')} недоступен. Разрешено: {queries or 'нет'}"
    )


def telemetry_diff(
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if before is None or after is None:
        return None
    changes: dict[str, Any] = {}
    for key in sorted(set(before) | set(after)):
        old = before.get(key)
        new = after.get(key)
        if old != new:
            changes[key] = {"before": old, "after": new}
    return changes
