"""Cross-component regressions. Local corpus and HTTP test client only."""
import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from dome_agent.config import AgentConfig
from dome_agent.agent import DomeAgent
from dome_agent.analysis import Snapshot
from dome_agent.chat import ChatMessage
from dome_agent.control import ModelCommand
from dome_agent.knowledge import KnowledgeService
from dome_agent.rules import RuleResult
from dome_agent.state import StateStore
from dome_rag.builder import db
from dome_rag.builder.answer import UNSAFE_RESET
from dome_rag.builder.ingest import build_nodes
from dome_rag.builder.md import first_sentences
from dome_rag.builder.retrieval import KB
from dome_web.dashboard_backend import create_frontend

ROOT = Path(__file__).resolve().parent.parent


class IntegratedKnowledgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.knowledge = KnowledgeService(ROOT / 'dome_rag')

    @classmethod
    def tearDownClass(cls):
        cls.knowledge.close()

    def test_live_request_cannot_be_answered_by_thresholds(self):
        result = self.knowledge.query('Какова концентрация CO2 в Куполе прямо сейчас?')
        self.assertTrue(result['requires_live'])
        self.assertNotEqual(result['verdict'], 'ANSWERABLE')
        self.assertEqual(result['facts'], [])
        self.assertTrue(result['unknown'])
        self.assertIn('live_telemetry', result['coverage']['missing'])
        self.assertEqual(result['confidence']['score'], 0)

    def test_live_diagnostic_keeps_live_requirement(self):
        result = self.knowledge.query('Почему сейчас принтер встал на паузу?')
        self.assertTrue(result['requires_live'])
        self.assertNotEqual(result['verdict'], 'ANSWERABLE')

    def test_disagreement_does_not_authorize_alarm_reset(self):
        result = self.knowledge.query('Датчик CO2 показывает 2500 ppm, резервный 600 ppm. Этого достаточно, чтобы считать тревогу ложной и сбросить блокировку?')
        self.assertTrue(result['requires_verification'])
        self.assertNotEqual(result['verdict'], 'ANSWERABLE')
        self.assertTrue(any('не доказывает' in x for x in result['unknown']))
        self.assertFalse(any(UNSAFE_RESET.search(x) for x in result['actions']))

    def test_fault_catalog_never_becomes_unconditional_actions(self):
        result = self.knowledge.query('Что делать при F-7.1?')
        self.assertTrue(result['hypotheses'])
        self.assertFalse(any(UNSAFE_RESET.search(x) for x in result['actions']))

    def test_api_identifiers_survive_markdown_extraction(self):
        self.assertEqual(first_sentences('`node_id`: `printer_3d_01`, `query`: `list_files`.'),
                         'node_id: printer_3d_01, query: list_files.')

    def test_read_query_does_not_recommend_starting_a_job(self):
        result = self.knowledge.query('Как узнать список файлов принтера?')
        self.assertIn('list_files', ' '.join(result['facts']))
        self.assertFalse(any('start_job' in x for x in result['actions']))

    def test_missing_database_is_reported_without_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = KnowledgeService(tmp)
            self.assertIsNone(service.query('Что такое CO2?'))
            self.assertIn('не собрана', service.error)

    def test_long_question_also_respects_minimum_prompt_budget(self):
        service = KnowledgeService(ROOT / 'dome_rag', max_chars=800)
        try:
            result = service.pack('Какая влажность опасна для станков? ' * 30)
            self.assertLessEqual(len(json.dumps(result, ensure_ascii=False, default=str)), 800)
            self.assertIn('verdict', result)
            self.assertTrue(result['note'])
        finally:
            service.close()

    def test_schema_does_not_disappear_without_telemetry_sample(self):
        nodes = build_nodes(None)
        self.assertEqual(len(nodes), 37)
        self.assertGreaterEqual(sum(len(n['params']) for n in nodes), 498)

    def test_database_is_read_only(self):
        kb = KB()
        try:
            with self.assertRaises(Exception):
                kb.con.execute("DELETE FROM cards")
        finally:
            kb.close()

    def test_installed_paths_do_not_depend_on_current_directory(self):
        self.assertTrue(AgentConfig.__dataclass_fields__['kb_dir'].default.is_absolute())

    def test_custom_directory_supplies_its_own_database_default(self):
        config = AgentConfig(dome_url='ws://unused.invalid', dome_key='test', kb_dir=ROOT/'custom-kb')
        self.assertEqual(config.kb_db, ROOT/'custom-kb/build/kb.sqlite')

    def test_failed_database_build_preserves_previous_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'kb.sqlite'
            path.write_bytes(b'existing database')
            with patch.object(db, '_load_json', side_effect=RuntimeError('invalid source')):
                with self.assertRaisesRegex(RuntimeError, 'invalid source'):
                    db.build_db(path)
            self.assertEqual(path.read_bytes(), b'existing database')
            self.assertEqual([p.name for p in Path(tmp).iterdir()], ['kb.sqlite'])

    def test_recent_backend_action_reconciliation_is_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = StateStore(Path(tmp) / 'state.json')
            state.begin_action('pause', request={'action': 'pause_job'}, ref=77, expected='PAUSED')
            self.assertTrue(state.get('pending_actions'))
            self.assertTrue(hasattr(state, 'reconcile_pending_actions'))


class FrontendKnowledgeTests(unittest.TestCase):
    def setUp(self):
        self.client = self.enterContext(TestClient(create_frontend(ROOT), base_url='http://127.0.0.1'))
        info = self.client.get('/api/frontend').json()
        self.headers = {'X-Dome-CSRF': info['csrf'], 'Origin': 'http://127.0.0.1'}
        self.post('settings', {'llm_enabled': False})

    def post(self, action, data):
        return self.client.post('/api/frontend/' + action, headers=self.headers, json=data)

    def test_search_and_chat_use_same_evidence(self):
        question = 'Какая влажность опасна для станков?'
        found = self.post('knowledge', {'query': question}).json()
        reply = self.post('chat', {'text': question}).json()['reply']
        self.assertEqual(found['answer']['verdict'], reply['verdict'])
        self.assertIn('90', reply['text'])
        self.assertTrue(reply['sources'])

    def test_unknown_and_live_answers_cannot_be_overridden_by_model(self):
        self.post('settings', {'llm_enabled': True, 'api_key': 'local-test-only'})
        with patch('dome_agent.llm.DeepSeekAdvisor.advise', side_effect=AssertionError('must not call model')):
            reply = self.post('chat', {'text': 'Какова концентрация CO2 прямо сейчас?'}).json()['reply']
            self.assertTrue(reply['requires_live'])
            self.assertNotEqual(reply['verdict'], 'ANSWERABLE')
            self.assertIn('Текущий статус оборудования не подтверждён', reply['text'])

    def test_no_product_version_in_visible_shell(self):
        html = self.client.get('/').text
        self.assertNotIn('Dome Control · 2.0', html)
        self.assertNotIn('v2', html.lower())
        self.assertIn('class="sidebar"', html)

    def test_index_health_is_published_without_credentials(self):
        info = self.client.get('/api/frontend').json()
        self.assertTrue(info['knowledge']['available'])
        self.assertNotIn('api_key', info)


class AgentKnowledgeFlowTests(unittest.IsolatedAsyncioTestCase):
    async def test_controller_receives_rag_but_protection_cannot_be_reset(self):
        class LocalAPI:
            async def catalog(self): return {'nodes': []}
            async def close(self): pass
            async def control_device(self, *a, **kw): raise AssertionError('No hardware call expected')
        with tempfile.TemporaryDirectory() as tmp:
            config = AgentConfig(dome_url='ws://unused.invalid', dome_key='test',
                log_dir=Path(tmp), state_file=Path(tmp)/'state.json',
                policy_file=ROOT/'dome_policy.json', control_enabled=True)
            agent = DomeAgent(config, api=LocalAPI())
            try:
                snapshot = Snapshot(json.loads((ROOT/'tests/fixtures/telemetry.jsonl').read_text(encoding='utf-8').splitlines()[0]))
                decision = RuleResult('Проверка состояния', '', 0.5)
                message = ChatMessage.from_raw({'id': 1, 'text': 'Какая влажность опасна для станков?', 'sender': {'role': 'operator', 'name': 'operator'}})
                context = await agent._control_context(snapshot, decision, [message])
                self.assertIn('90', ' '.join(context['knowledge']['facts']))
                self.assertTrue(context['rule_decision']['rag_refs'])
                read = await agent.tool.execute(ModelCommand(tool='knowledge', query='Что означает IL_CO2?'))
                self.assertTrue(read['ok'])
                blocked = await agent._execute_model_command(ModelCommand(tool='control', node_id='dome_automation_01', action='block_interlock', value='IL_CO2'), force=True, source='test')
                self.assertTrue(blocked['blocked'])
                self.assertIn('hard-denied', blocked['error'])
                unsupported = await agent._execute_model_command(ModelCommand(tool='control', node_id='printer_3d_01', action='stop_job'), force=True, source='test')
                self.assertTrue(unsupported['blocked'])
            finally:
                await agent._shutdown()


if __name__ == '__main__':
    unittest.main()
