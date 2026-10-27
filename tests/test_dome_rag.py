"""Тесты базы знаний: таксономия, карточки, покрытие и мост к агенту.

Тесты, которым нужна собранная база, пропускаются, если ``kb.sqlite`` ещё не
создан: сборка — отдельный шаг (``py kb.py build``), а не побочный эффект
юнит-теста.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KB_DIR = ROOT / "dome_rag"
KB_DB = KB_DIR / "build" / "kb.sqlite"

if str(KB_DIR) not in sys.path:
    sys.path.insert(0, str(KB_DIR))

from dome_agent.control import DomeTool, ModelCommand  # noqa: E402
from dome_agent.knowledge import KnowledgeService  # noqa: E402

BUILT = KB_DB.is_file()
CARD_JSON = KB_DIR / "data" / "cards.json"

requires_db = unittest.skipUnless(BUILT, f"база не собрана: {KB_DB}")


class TaxonomyTests(unittest.TestCase):
    def setUp(self) -> None:
        from dome_rag.builder import coverage, taxonomy

        self.coverage = coverage
        self.taxonomy = taxonomy

    def test_attributes_have_labels_and_synonyms(self) -> None:
        for identifier, item in self.taxonomy.attributes().items():
            self.assertTrue(item.get("label"), identifier)
            if identifier == "section_text":
                # Служебный атрибут для абзацев без привязки: синонимов нет,
                # и он никогда не может дать ANSWERABLE.
                self.assertFalse(item.get("synonyms"))
                continue
            self.assertTrue(item.get("synonyms"), identifier)

    def test_russian_question_maps_to_attribute(self) -> None:
        slots = self.coverage.extract_slots("Какая влажность считается опасной для станков?")
        self.assertIn("threshold_value", slots.attributes)
        self.assertFalse(slots.off_domain)

    def test_off_domain_question_is_recognised(self) -> None:
        slots = self.coverage.extract_slots("Какой пароль у администратора купола?")
        self.assertTrue(slots.off_domain)
        self.assertIn("credential", slots.attributes)

    def test_family_match_is_not_symmetric_blindly(self) -> None:
        # Ответ на «что делать» может лежать в правиле/запрете, но правило
        # не обязано отвечать на вопрос о последствиях.
        self.assertGreaterEqual(
            self.coverage._attribute_match("fault_action", "policy_rule"), 0.65
        )


class CardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from dome_rag.builder.cards import CardBuilder

        cls.cards = CardBuilder().build()

    def test_build_produces_many_cards(self) -> None:
        self.assertGreater(len(self.cards), 1000)

    def test_hold_cards_explain_pause_reason(self) -> None:
        hold = [c for c in self.cards if c.get("canonical_id") == "HOLD-PRINTER"]
        attributes = {c["attribute"] for c in hold}
        self.assertIn("fault_cause", attributes)
        self.assertIn("fault_action", attributes)
        cause = next(c for c in hold if c["attribute"] == "fault_cause")
        self.assertIn("hold_reason", cause["value"])

    def test_climate_threshold_is_bound_to_climate_node(self) -> None:
        # Метрика порога записана латиницей (climate.humidity_pct). Без явной
        # привязки карточка не находилась вопросом «какая влажность опасна».
        card = next(c for c in self.cards if c.get("canonical_id") == "TH_CONDENSATE")
        self.assertIn("climate_sensor_01", card["entities"])

    def test_cards_have_source_and_subject_attribute(self) -> None:
        for card in self.cards[:200]:
            self.assertTrue(card.get("subject"), card["card_id"])
            self.assertTrue(card.get("attribute"), card["card_id"])
            self.assertTrue((card.get("source") or {}).get("doc"), card["card_id"])

    def test_manifest_has_no_secrets(self) -> None:
        manifest = KB_DIR / "corpus_manifest.json"
        if not manifest.is_file():
            self.skipTest("манифест не собран")
        text = manifest.read_text(encoding="utf-8").lower()
        for marker in ("team_1.txt", "team_2.txt", ".env", ".pem", "id_rsa"):
            self.assertNotIn(marker, text)


@requires_db
class AnswerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.service = KnowledgeService(KB_DIR, db_path=KB_DB)

    def test_answerable_question_has_number_and_confidence(self) -> None:
        pack = self.service.pack("Какая влажность считается опасной для станков?")
        self.assertIsNotNone(pack)
        self.assertEqual(pack["verdict"], "ANSWERABLE")
        self.assertIn("90", " ".join(pack["facts"]))
        self.assertGreater(pack["confidence"]["score"], 0.5)
        self.assertTrue(pack["sources"])

    def test_kb_detects_climate_entity_in_question(self) -> None:
        from dome_rag.builder.retrieval import KB

        kb = KB()
        try:
            self.assertIn(
                "climate_sensor_01",
                kb.detect_entities("Какая влажность считается опасной для станков?"),
            )
        finally:
            kb.con.close()

    def test_negative_question_abstains_with_zero_confidence(self) -> None:
        for question in (
            "Какой пароль у администратора купола?",
            "Какая столица Австралии?",
            "Сколько стоит килограмм филамента?",
        ):
            pack = self.service.pack(question)
            self.assertIsNotNone(pack)
            self.assertEqual(pack["verdict"], "NO_RELEVANT_EVIDENCE", question)
            self.assertEqual(pack["confidence"]["score"], 0.0, question)
            self.assertEqual(pack["facts"], [])
            self.assertEqual(pack["sources"], [])
            self.assertTrue(pack["note"])

    def test_pack_respects_character_budget(self) -> None:
        service = KnowledgeService(KB_DIR, db_path=KB_DB, max_chars=900)
        pack = service.pack("Что делать при потере телеметрии узла?")
        self.assertIsNotNone(pack)
        self.assertLessEqual(
            len(json.dumps(pack, ensure_ascii=False, default=str)), 900
        )

    def test_pack_never_grants_permission(self) -> None:
        pack = self.service.pack("Можно ли агенту снять деталь со стола?")
        self.assertIsNotNone(pack)
        self.assertIn("не разрешение", pack["note"].lower())
        self.assertNotIn("allowed", pack)
        self.assertNotIn("permission", pack)


@requires_db
class RegressionTests(unittest.TestCase):
    """Регрессии, найденные при разборе реальных провалов eval.

    Каждый тест соответствует конкретному классу ошибок, а не одной строке
    вопроса: точный код, общее правило без сущности, метрика в скобках,
    ложный ответ на off-domain вопрос, различение датированной и
    недатированной записи журнала, LCS-перекрытие и алиасы возможностей.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.service = KnowledgeService(KB_DIR, db_path=KB_DB)

    def test_exact_code_is_answered_even_below_top_k(self) -> None:
        # «Что означает GAP-001»: карточка кода может не попасть в top-k
        # обычного поиска, но точный код обязан добавлять её в рассмотрение.
        pack = self.service.pack("Что означает GAP-001?")
        self.assertEqual(pack["verdict"], "ANSWERABLE")
        self.assertIn("GAP-001", " ".join(pack["facts"]))

    def test_general_policy_without_entity_still_answers(self) -> None:
        # POL-001 не называет принтер, но прямо отвечает на «можно ли агенту
        # снять деталь». Раньше entity-фильтр делал ответ PARTIAL.
        pack = self.service.pack("Можно ли агенту снять деталь со стола принтера?")
        self.assertEqual(pack["verdict"], "ANSWERABLE")
        self.assertIn("Оператор", " ".join(pack["facts"]))

    def test_threshold_metric_in_parentheses_is_bound(self) -> None:
        # Метрика записана как «попытки resume (…resume_attempts)». Разбор
        # скобок в вариантах субъекта привязывает вопрос к TH_RESUME_LIMIT.
        pack = self.service.pack("Сколько раз можно нажать resume?")
        self.assertEqual(pack["verdict"], "ANSWERABLE")
        self.assertIn("resume_attempts", " ".join(pack["facts"]))

    def test_off_domain_question_abstains(self) -> None:
        for question in ("Какая высота купола?", "Кто написал «Войну и мир»?"):
            pack = self.service.pack(question)
            self.assertEqual(pack["verdict"], "NO_RELEVANT_EVIDENCE", question)
            self.assertEqual(pack["confidence"]["score"], 0.0, question)

    def test_dated_shift_is_answerable_but_undated_journal_is_partial(self) -> None:
        dated = self.service.pack("Что было в смену 2026-10-15?")
        self.assertEqual(dated["verdict"], "ANSWERABLE")
        self.assertIn("2026-10-15", " ".join(dated["facts"]))

        # «Мнение жильца» — слабый источник: без даты нельзя утверждать, о
        # какой смене речь. Это честный PARTIAL, а не ANSWERABLE.
        undated = self.service.pack("Что записал жилец в бортовой журнал?")
        self.assertEqual(undated["verdict"], "PARTIAL")
        self.assertLessEqual(undated["confidence"]["score"], 0.55)

    def test_phrase_overlap_uses_longest_common_subsequence(self) -> None:
        from dome_rag.builder.coverage import phrase_overlap

        # Общие слова идут не подряд: «резкое падение … генерации» против
        # «внезапное падение … генерации». Мешок слов дал бы 0, LCS — 2/3.
        self.assertAlmostEqual(
            phrase_overlap("резкое падение генерации", "внезапное падение солнечной генерации"),
            2 / 3,
            places=3,
        )

    def test_capability_aliases_are_found(self) -> None:
        message = self.service.pack("Как написать Оператору?")
        self.assertEqual(message["verdict"], "ANSWERABLE")
        self.assertIn("post_message", " ".join(message["facts"]).lower())

        files = self.service.pack("Как посмотреть файлы принтера?")
        self.assertEqual(files["verdict"], "ANSWERABLE")
        self.assertIn("list_files", " ".join(files["facts"]).lower())

    def test_collision_alias_and_load_shed_lines(self) -> None:
        # Бытовые формулировки, которых нет в тексте правила дословно.
        collision = self.service.pack("Что произойдёт при столкновении сопла с деталью?")
        self.assertEqual(collision["verdict"], "ANSWERABLE")
        self.assertIn("BED_COLLISION", " ".join(collision["facts"]))

        lines = self.service.pack("Какие линии отключаются при дефиците мощности?")
        self.assertEqual(lines["verdict"], "ANSWERABLE")
        self.assertIn("IL_LOAD_SHED", " ".join(lines["facts"]))

    def test_sensor_overread_is_a_symptom_not_a_live_request(self) -> None:
        # «Показания» раньше включали LIVE и уводили вопрос в live_state.
        pack = self.service.pack("Как понять, что датчик завышает показания?")
        self.assertEqual(pack["verdict"], "ANSWERABLE")
        self.assertIn("завыша", " ".join(pack["facts"] + pack["hypotheses"]).lower())

    def test_crew_is_answerable_but_captain_is_not(self) -> None:
        size = self.service.pack("Сколько человек в экипаже?")
        self.assertEqual(size["verdict"], "ANSWERABLE")
        self.assertIn("2–4", " ".join(size["facts"]))

        roles = self.service.pack("Какие роли есть в экипаже?")
        self.assertEqual(roles["verdict"], "ANSWERABLE")
        self.assertIn("Инженер-оператор", " ".join(roles["facts"]))

        # В корпусе нет капитана экипажа: это по-прежнему честный отказ.
        captain = self.service.pack("Кто капитан экипажа?")
        self.assertEqual(captain["verdict"], "NO_RELEVANT_EVIDENCE")

    def test_undocumented_spec_abstains(self) -> None:
        for question in (
            "Какой цвет корпуса у 3D-принтера?",
            "Какая гарантия на 3D-принтер?",
            "Сколько памяти у Raspberry Pi?",
            "Как часто можно опрашивать шину МС-ТЮК?",
        ):
            pack = self.service.pack(question)
            self.assertEqual(pack["verdict"], "NO_RELEVANT_EVIDENCE", question)
            self.assertEqual(pack["confidence"]["score"], 0.0, question)

    def test_history_shift_closes_secondary_interlock_slot(self) -> None:
        # Запись смены прямо перечисляет срабатывания интерлоков, хотя
        # отдельной карточки interlock_trigger для этой даты может не быть.
        pack = self.service.pack("Какие интерлоки срабатывали в смену 2026-10-16?")
        self.assertEqual(pack["verdict"], "ANSWERABLE")
        self.assertIn("2026-10-16", " ".join(pack["facts"]))

    def test_composition_question_keeps_subject_and_drops_neighbour_faq(self) -> None:
        # «Какие системы входят в Купол» не должно подкрепляться FAQ про зоны
        # доступа или роли экипажа: у них другой субъект.
        pack = self.service.pack("Какие системы входят в купол?")
        self.assertEqual(pack["verdict"], "ANSWERABLE")
        facts = " ".join(pack["facts"])
        self.assertIn("Энергетика", facts)
        self.assertNotIn("Зона общего доступа", facts)

    def test_comparison_presents_both_sides_as_facts(self) -> None:
        # Вопрос-сравнение просит описания обеих сторон, а не гипотезу об
        # одной из них.
        pack = self.service.pack(
            "Чем отличается ложное завышение показаний от ложного занижения?"
        )
        self.assertEqual(pack["verdict"], "ANSWERABLE")
        facts = " ".join(pack["facts"]).lower()
        self.assertIn("завыша", facts)
        self.assertIn("занижа", facts)


class _FakeApi:
    async def close(self) -> None:
        return None


@requires_db
class AgentWiringTests(unittest.IsolatedAsyncioTestCase):
    async def test_agent_attaches_knowledge_pack_and_logs_it(self) -> None:
        from dome_agent.agent import DomeAgent
        from dome_agent.config import AgentConfig

        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs"
            config = AgentConfig(
                dome_url="ws://localhost:1",
                dome_key="test-key",
                log_dir=log_dir,
                state_file=log_dir / "state.json",
                policy_file=ROOT / "dome_policy.json",
                kb_dir=KB_DIR,
                kb_db=KB_DB,
                kb_max_chars=2000,
            )
            agent = DomeAgent(config, api=_FakeApi())
            try:
                pack = await agent._knowledge_pack(
                    "Что такое IL_CO2?", purpose="test"
                )
                self.assertIsNotNone(pack)
                self.assertEqual(pack["verdict"], "ANSWERABLE")
                self.assertEqual(agent.stats["knowledge_queries"], 1)
                self.assertLessEqual(
                    len(json.dumps(pack, ensure_ascii=False, default=str)), 2000
                )
            finally:
                await agent._shutdown()

            records = [
                json.loads(line)
                for path in log_dir.glob("*.jsonl")
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            knowledge = [r for r in records if r["kind"] == "knowledge"]
            self.assertTrue(knowledge)
            self.assertEqual(knowledge[-1]["verdict"], "ANSWERABLE")

    async def test_agent_works_without_built_database(self) -> None:
        from dome_agent.agent import DomeAgent
        from dome_agent.config import AgentConfig

        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs"
            config = AgentConfig(
                dome_url="ws://localhost:1",
                dome_key="test-key",
                log_dir=log_dir,
                state_file=log_dir / "state.json",
                policy_file=ROOT / "dome_policy.json",
                kb_dir=Path(tmp) / "absent",
                kb_db=Path(tmp) / "absent" / "build" / "kb.sqlite",
            )
            agent = DomeAgent(config, api=_FakeApi())
            try:
                self.assertIsNone(
                    await agent._knowledge_pack("Что такое IL_CO2?", purpose="test")
                )
                self.assertEqual(agent.stats["knowledge_queries"], 0)
            finally:
                await agent._shutdown()


class DegradedServiceTests(unittest.TestCase):
    def test_missing_database_is_not_an_exception(self) -> None:
        service = KnowledgeService(
            ROOT / "dome_rag_absent",
            db_path=ROOT / "dome_rag_absent" / "build" / "kb.sqlite",
        )
        self.assertFalse(service.available)
        self.assertIsNone(service.pack("Что такое IL_CO2?"))
        self.assertTrue(service.error)

    def test_disabled_service_does_nothing(self) -> None:
        service = KnowledgeService(KB_DIR, db_path=KB_DB, enabled=False)
        self.assertFalse(service.available)
        self.assertIsNone(service.pack("Что такое IL_CO2?"))


@requires_db
class DomeToolKnowledgeTests(unittest.IsolatedAsyncioTestCase):
    async def test_knowledge_tool_is_read_only(self) -> None:
        tool = DomeTool(object(), KnowledgeService(KB_DIR, db_path=KB_DB))
        result = await tool.execute(
            ModelCommand(tool="knowledge", query="При каком CO2 срабатывает IL_CO2?"),
            force=True,
        )
        self.assertTrue(result["ok"])
        pack = result["result"]
        self.assertEqual(pack["verdict"], "ANSWERABLE")
        self.assertIn("2000", " ".join(pack["facts"]))

    async def test_knowledge_tool_without_service_fails_softly(self) -> None:
        tool = DomeTool(object())
        result = await tool.execute(
            ModelCommand(tool="knowledge", query="Что такое IL_CO2?"),
        )
        self.assertFalse(result["ok"])
        self.assertIn("disabled", result["error"])

    async def test_knowledge_unknown_tool_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ModelCommand(tool="shell", action="rm -rf /")


if __name__ == "__main__":
    unittest.main()
