import json
from pathlib import Path
import unittest
from fastapi.testclient import TestClient
from dome_web.dashboard_backend import create_frontend

BACKEND=Path(__file__).resolve().parent.parent

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

if __name__=='__main__':unittest.main()
