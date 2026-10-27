import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from dome_web.dashboard_backend import create_frontend
from dome_web.service import DashboardService

BACKEND=Path(__file__).resolve().parent.parent

PANEL_LINES = [
    {"line": 1, "name": "Производственный отдел", "state": "ON", "protection_state": "ARMED"},
    {"line": 2, "name": "Теплица ФабЛаб", "state": "ON", "protection_state": "ARMED"},
    {"line": 3, "name": "3D-принтер", "state": "ON", "protection_state": "ARMED"},
    {"line": 4, "name": "ЧПУ-фрезер", "state": "ON", "protection_state": "ARMED"},
    {"line": 5, "name": "Теплица Химлаб", "state": "ON", "protection_state": "ARMED"},
    {"line": 6, "name": "Обычное освещение", "state": "ON", "protection_state": "ARMED"},
    {"line": 7, "name": "Аварийное освещение", "state": "OFF", "protection_state": "ARMED"},
    {"line": 8, "name": "Резервная линия", "state": "ON", "protection_state": "ARMED"},
]

class FrontendBackendTests(unittest.TestCase):
    def setUp(self):
        self.client=TestClient(create_frontend(BACKEND),base_url='http://127.0.0.1')
        self.client.__enter__()
        self.addCleanup(self.client.__exit__,None,None,None)
        self.info=self.client.get('/api/frontend').json()
        self.headers={'X-Dome-CSRF':self.info['csrf'],'Origin':'http://127.0.0.1'}
    def post(self,action,data):return self.client.post('/api/frontend/'+action,json=data,headers=self.headers)
    def test_our_original_visual_shell_is_served(self):
        html=self.client.get('/').text
        self.assertIn('class="sidebar"',html)
        self.assertIn('/ui/styles.css',html)
        self.assertNotIn('class="topbar__inner"',html)
    def test_new_backend_contract_is_preserved(self):
        data=self.client.get('/api/demo').json()
        self.assertTrue(data['demo'])
        self.assertIn('observation',data)
        self.assertIn('verifications',data)
    def test_our_assets_are_available(self):
        for name in ['app.js','styles.css','explanations.css','insights.js']:
            self.assertEqual(self.client.get('/ui/'+name).status_code,200)
    def test_mutation_requires_csrf(self):
        self.assertEqual(self.client.post('/api/frontend/settings',json={}).status_code,403)
    def test_foreign_origin_rejected(self):
        self.assertEqual(self.client.post('/api/frontend/settings',json={},headers={**self.headers,'Origin':'https://untrusted.example'}).status_code,403)
    def test_lan_ip_host_and_origin_accepted(self):
        headers={**self.headers,'Host':'192.168.1.50:8080','Origin':'http://192.168.1.50:8080'}
        response=self.client.post('/api/frontend/settings',json={'llm_enabled':False},headers=headers)
        self.assertEqual(response.status_code,200,response.text)
        self.assertTrue(response.json()['ok'])
    def test_foreign_and_public_hosts_rejected_for_reads(self):
        for host in ['untrusted.example','8.8.8.8']:
            response=self.client.get('/api/frontend',headers={'Host':host})
            self.assertEqual(response.status_code,403,host)
            self.assertIn('Недопустимый адрес панели',response.text)
    def test_model_key_stays_on_server(self):
        self.assertEqual(self.post('settings',{'api_key':'private-test-key','llm_enabled':False}).status_code,200)
        text=self.client.get('/api/frontend').text
        self.assertNotIn('private-test-key',text)
    def test_agent_chat_uses_new_demo_without_device_controls(self):
        self.post('settings',{'llm_enabled':False})
        before=self.client.get('/api/demo').json()
        response=self.post('chat',{'demo':True,'text':'Какой заряд АКБ?'})
        self.assertEqual(response.status_code,200,response.text)
        self.assertEqual(response.json()['reply']['engine'],'Справка без модели')
        self.assertIn('battery_01',response.json()['reply']['text'])
        self.assertEqual(self.client.get('/api/demo').json()['metrics'],before['metrics'])
    def test_no_control_route_is_added(self):
        self.assertIn(self.client.post('/api/control',json={}).status_code,[404,405])
        self.assertEqual(self.post('control',{'node_id':'cnc_01','action':'resume'}).status_code,404)
    def test_create_frontend_starts_the_service_loop_without_monitor(self):
        calls=[]
        original=DashboardService.start
        def spy(service,**kwargs):
            calls.append(kwargs)
            return original(service,**kwargs)
        with patch.object(DashboardService,'start',spy):
            client=TestClient(create_frontend(BACKEND),base_url='http://127.0.0.1')
            with client:pass
        self.assertEqual(calls,[{'monitor':False}])
    def test_chat_control_is_wired_through_confirmation(self):
        executed=[]
        async def fake_lines(service):
            service.settings['control_enabled']=True
            return copy.deepcopy(PANEL_LINES)
        async def fake_execute(service,command):
            executed.append(command)
            return {'ok':True,'verified':True,'noop':False,'line':command.line,'label':command.label,'action':command.action,'state':'ON'}
        with patch.object(DashboardService,'panel_lines',fake_lines),patch.object(DashboardService,'execute_line_command',fake_execute):
            client=TestClient(create_frontend(BACKEND),base_url='http://127.0.0.1')
            with client:
                info=client.get('/api/frontend').json()
                headers={'X-Dome-CSRF':info['csrf'],'Origin':'http://127.0.0.1'}
                first=client.post('/api/frontend/chat',json={'demo':True,'text':'включи аварийное освещение'},headers=headers)
                self.assertEqual(first.status_code,200,first.text)
                self.assertEqual(first.json()['reply']['verdict'],'CONTROL')
                self.assertIn('линия 7',first.json()['reply']['text'].lower())
                self.assertEqual(executed,[])
                second=client.post('/api/frontend/chat',json={'demo':True,'text':'да'},headers=headers)
                self.assertEqual(second.status_code,200,second.text)
                self.assertIn('Готово',second.json()['reply']['text'])
                self.assertEqual(len(executed),1)
                self.assertEqual(executed[0].line,7)
                self.assertEqual(executed[0].action,'line_on')

if __name__=='__main__':unittest.main()
