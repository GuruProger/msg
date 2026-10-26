from __future__ import annotations

import unittest

from dome_control.domain import (
    DomeValidationError,
    filter_catalog,
    find_node,
    resolve_control_action,
    resolve_query_name,
    telemetry_diff,
    telemetry_node,
)


CATALOG = [
    {
        "node_id": "smart_panel_01",
        "code_name": "smart_panel",
        "title": "Smart panel",
        "system": "Power",
        "category": "power",
        "controls": ["line_on", "line_off"],
        "control_aliases": {"line_on": ["enable_line"], "line_off": ["disable_line"]},
        "operator_controls": ["inspect_line"],
        "queries": [],
        "service_actions": ["reconnect"],
    },
    {
        "node_id": "printer_3d_01",
        "code_name": "printer_3d",
        "title": "Printer",
        "system": "Production",
        "category": "production",
        "controls": ["home"],
        "control_aliases": {"home": []},
        "operator_controls": [],
        "queries": ["list_files"],
        "service_actions": [],
    },
]


class DomainTests(unittest.TestCase):
    def test_find_node_by_id_and_code_name(self) -> None:
        self.assertEqual(find_node(CATALOG, "smart_panel_01")["node_id"], "smart_panel_01")
        self.assertEqual(find_node(CATALOG, "SMART_PANEL")["node_id"], "smart_panel_01")
        self.assertIsNone(find_node(CATALOG, "missing"))

    def test_filter_catalog(self) -> None:
        self.assertEqual(len(filter_catalog(CATALOG, system="power")), 1)
        self.assertEqual(len(filter_catalog(CATALOG, search="print")), 1)
        self.assertEqual(len(filter_catalog(CATALOG, system="power", search="printer")), 0)

    def test_resolve_control_alias_and_canonical(self) -> None:
        node = CATALOG[0]
        self.assertEqual(resolve_control_action(node, "line_on"), "line_on")
        self.assertEqual(resolve_control_action(node, "enable_line"), "line_on")
        self.assertEqual(resolve_control_action(node, "disable_line"), "line_off")
        self.assertEqual(resolve_control_action(node, "reconnect"), "reconnect")

    def test_operator_action_is_rejected(self) -> None:
        with self.assertRaises(DomeValidationError) as ctx:
            resolve_control_action(CATALOG[0], "inspect_line")
        self.assertIn("только Оператору", str(ctx.exception))

    def test_unknown_action_and_force(self) -> None:
        with self.assertRaises(DomeValidationError):
            resolve_control_action(CATALOG[0], "bogus")
        self.assertEqual(resolve_control_action(CATALOG[0], "bogus", force=True), "bogus")

    def test_query_resolution(self) -> None:
        self.assertEqual(resolve_query_name(CATALOG[1], "list_files"), "list_files")
        with self.assertRaises(DomeValidationError):
            resolve_query_name(CATALOG[1], "bogus")
        self.assertEqual(resolve_query_name(CATALOG[1], "bogus", force=True), "bogus")

    def test_telemetry_node(self) -> None:
        data = {"nodes": {"smart_panel_01": {"online": True}}}
        self.assertEqual(telemetry_node(data, "smart_panel_01")[0], "smart_panel_01")
        self.assertEqual(telemetry_node(data, "SMART_PANEL_01")[0], "smart_panel_01")
        self.assertIsNone(telemetry_node(data, "missing"))

    def test_telemetry_diff(self) -> None:
        diff = telemetry_diff({"a": 1, "b": 2}, {"a": 1, "b": 3})
        self.assertEqual(diff, {"b": {"before": 2, "after": 3}})
        self.assertIsNone(telemetry_diff(None, {"a": 1}))


if __name__ == "__main__":
    unittest.main()
