"""Serve our frontend over the supplied backend without editing its sources."""
import argparse
import asyncio
from contextlib import asynccontextmanager
import importlib
import json
import os
from pathlib import Path
import secrets
import sys
from types import SimpleNamespace
from urllib.parse import urlsplit


def create_frontend(backend_root: Path | None = None):
    backend_root=Path(backend_root or Path(__file__).resolve().parent.parent).resolve()
    if not (backend_root/'dome_web/journal_api.py').is_file():
        raise RuntimeError('Укажите папку нового бэкенда с dome_web/journal_api.py')
    from fastapi import Request
    from fastapi.responses import FileResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from starlette.routing import Route
    from .journal_api import create_app
    from .service import DashboardService, WebError
    root=Path(__file__).parent/'static'
    log_dir=Path(os.getenv('AGENT_LOG_DIR') or backend_root/'logs')
    state_file=Path(os.getenv('AGENT_STATE_FILE') or log_dir/'agent_state.json')
    app=create_app(log_dir=log_dir,state_file=state_file)
    endpoints={r.path:r.endpoint for r in app.routes if hasattr(r,'endpoint')}
    service=DashboardService(backend_root,env_file='.env')
    service.csrf=secrets.token_urlsafe(32)
    service.connection='journal'
    original_search=service.search
    # Prepared records are also usable when no chunks/index are shipped.
    if not any((backend_root/'dome_rag/chunks').glob('*.jsonl')):
        chunks=[]
        for p in (backend_root/'dome_rag/data').glob('*.json'):
            try: data=json.loads(p.read_text(encoding='utf-8'))
            except (OSError,ValueError):continue
            rows=data if isinstance(data,list) else [{k:v} for k,v in data.items()] if isinstance(data,dict) else []
            for row in rows:
                chunks.append({'title':row.get('title') or row.get('name') or row.get('id') or p.stem,'source':str(p.relative_to(backend_root)),'text':json.dumps(row,ensure_ascii=False,indent=2)})
        service._kb_chunks=chunks

    async def snapshot(demo=False):
        endpoint=endpoints['/api/demo' if demo else '/api/summary']
        return await asyncio.to_thread(endpoint)

    def absorb(data):
        obs=data.get('observation') or {};m=data.get('metrics') or {};nodes={}
        for key,node_id,param in [('soc_pct','battery_01','soc_pct'),('co2_ppm','climate_sensor_01','co2_ppm'),('printer','printer_3d_01','state'),('cnc','cnc_01','state'),('fume','fume_extractor_01','state'),('power_w','power_summary','power_w')]:
            if key=='co2_ppm':node_id=(obs.get('metric_sources') or {}).get(key) or 'co2_observation'
            if m.get(key) is not None:nodes[node_id]={param:m[key],'updated_at':obs.get('ts')}
        service.demo=data.get('demo',False)
        service.telemetry={'nodes':nodes,'game_time':obs.get('dome_time'),'observed_at':obs.get('ts'),'source_kind':'demo' if service.demo else 'journal'}
        service.catalog=[{'node_id':k,'title':k} for k in nodes]
        service.connection='demo' if service.demo else 'journal'
        service.settings['url']='journal://'+str((data.get('run') or {}).get('run_id') or 'waiting')
        service.agent=SimpleNamespace(state=SimpleNamespace(task=(data.get('state') or {}).get('task')))
        service._alerts=lambda:data.get('alerts') or []

    async def index(request):return FileResponse(root/'index.html')
    app.router.routes.insert(0,Route('/',index,methods=['GET']))
    app.mount('/ui',StaticFiles(directory=str(root)),name='own-frontend')

    @app.get('/api/frontend')
    async def frontend_info():
        return service.redact({'csrf':service.csrf,'model':service.settings['model'],'base_url':service.settings['base_url'],'llm_enabled':service.settings['llm_enabled'],'api_key_saved':bool(service.settings['api_key']),'messages':list(service.conversation.messages),'busy':service.conversation.busy,'knowledge':service.knowledge.status(),'backend_root':str(backend_root),'log_dir':str(log_dir),'state_file':str(state_file)})

    async def authorized(request):
        host=request.headers.get('host','')
        if host.split(':')[0] not in {'localhost','127.0.0.1'}:raise WebError('Недопустимый адрес панели',403)
        if request.headers.get('origin') not in {None,'http://'+host}:raise WebError('Запрос с другого сайта запрещён',403)
        if not secrets.compare_digest(request.headers.get('x-dome-csrf',''),service.csrf):raise WebError('Обновите страницу перед отправкой',403)
        body=await request.body()
        if len(body)>16000:raise WebError('Слишком большой запрос',413)
        data=json.loads(body)
        if not isinstance(data,dict):raise WebError('Ожидается объект JSON')
        return data

    @app.post('/api/frontend/{action}')
    async def frontend_action(action:str,request:Request):
        try:
            data=await authorized(request)
            if action=='chat':
                absorb(await snapshot(data.get('demo') is True))
                return await service.conversation.ask(data.get('text',''))
            if action=='knowledge':return await asyncio.to_thread(original_search,str(data.get('query','')))
            if action=='settings':
                if service.conversation.busy:raise WebError('Дождитесь ответа агента',409)
                for key in ['model','base_url']:
                    if key in data:
                        v=str(data[key]).strip()
                        if not v or len(v)>1000:raise WebError('Некорректное значение настройки')
                        if key=='base_url' and urlsplit(v).scheme not in {'https','http'}:raise WebError('Укажите адрес API модели')
                        if key=='base_url' and v!=service.settings[key]:service.settings['api_key']=''
                        service.settings[key]=v
                if data.get('api_key'):service.settings['api_key']=str(data['api_key']).strip()
                if isinstance(data.get('llm_enabled'),bool):service.settings['llm_enabled']=data['llm_enabled']
                return {'ok':True}
            raise WebError('Эта операция отсутствует',404)
        except WebError as e:return JSONResponse({'ok':False,'error':str(e)},status_code=e.status)
        except (ValueError,TypeError):return JSONResponse({'ok':False,'error':'Некорректные данные запроса'},status_code=400)
        except Exception as e:return JSONResponse({'ok':False,'error':service.redact(str(e))},status_code=503)
    original_lifespan=app.router.lifespan_context
    @asynccontextmanager
    async def lifespan(application):
        async with original_lifespan(application):
            try:yield
            finally:service.close()
    app.router.lifespan_context=lifespan
    return app


def main():
    parser=argparse.ArgumentParser(description='Моя панель Купола · новый бэкенд')
    parser.add_argument('--backend-root',type=Path,default=Path(__file__).resolve().parent.parent)
    parser.add_argument('--port',type=int,default=8080)
    args=parser.parse_args()
    import uvicorn
    uvicorn.run(create_frontend(args.backend_root),host='127.0.0.1',port=args.port)
