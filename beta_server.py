"""Website beta alongside the existing WhatsApp service. No API key in the browser."""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import secrets
import sqlite3
import threading
import time

import bot
import server

LOG = logging.getLogger('website-beta')
HERE = Path(__file__).resolve().parent
SESSION_TTL = 24 * 3600
MAX_BODY = 16000


class BetaError(Exception):
    def __init__(self, status, message):
        self.status, self.message = status, message


class LockedEngine:
    """Share corpus/index memory and serialize paid work across both channels."""
    def __init__(self, engine):
        self.engine, self.lock = engine, threading.Lock()

    @property
    def is_ready(self):
        return self.engine.is_ready

    def ready(self):
        with self.lock:
            return self.engine.ready()

    def answer(self, question, history):
        with self.lock:
            return self.engine.answer(question, history)


class BetaApp:
    def __init__(self, service, total_limit=20, session_limit=5):
        if not 1 <= session_limit <= total_limit <= 1000:
            raise ValueError('Invalid beta daily limits')
        self.service, self.total_limit, self.session_limit = service, total_limit, session_limit
        self.path = service.cfg.data_dir / 'website-beta.sqlite3'
        self.stop, self.wake = threading.Event(), threading.Event()
        self.thread = None
        self.html = (HERE / 'beta.html').read_bytes()
        with self.db() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY, expires REAL NOT NULL, history TEXT NOT NULL DEFAULT '[]');
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, session TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    question TEXT NOT NULL, state TEXT NOT NULL, created REAL NOT NULL,
                    result TEXT NOT NULL DEFAULT '');
                CREATE TABLE IF NOT EXISTS quota (
                    day TEXT NOT NULL, session TEXT NOT NULL, count INTEGER NOT NULL,
                    PRIMARY KEY(day,session));
            ''')

    @contextmanager
    def db(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA foreign_keys=ON')
        try:
            with db:
                yield db
        finally:
            db.close()

    def cleanup(self):
        cutoff = datetime.fromtimestamp(time.time()-8*86400,timezone.utc).date().isoformat()
        with self.db() as db:
            db.execute('DELETE FROM sessions WHERE expires<?',(time.time(),))
            db.execute('DELETE FROM quota WHERE day<?',(cutoff,))

    def create_session(self):
        token = secrets.token_urlsafe(32)
        sid = hashlib.sha256(token.encode()).hexdigest()
        expires = time.time()+SESSION_TTL
        with self.db() as db:
            db.execute('BEGIN IMMEDIATE')
            db.execute('DELETE FROM sessions WHERE expires<?',(time.time(),))
            if db.execute('SELECT count(*) FROM sessions').fetchone()[0] >= 200:
                raise BetaError('429 Too Many Requests','גרסת הבטא עמוסה כרגע. נסו שוב בהמשך.')
            db.execute('INSERT INTO sessions(id,expires) VALUES (?,?)',(sid,expires))
        return {'token':token,'expires':expires,'daily_limit':self.session_limit}

    def session(self, environ):
        auth = environ.get('HTTP_AUTHORIZATION','')
        if not auth.startswith('Bearer ') or len(auth)>100:
            raise BetaError('401 Unauthorized','השיחה פגה. יש לפתוח שיחה חדשה.')
        sid = hashlib.sha256(auth[7:].encode()).hexdigest()
        with self.db() as db:
            row = db.execute('SELECT * FROM sessions WHERE id=? AND expires>?',(sid,time.time())).fetchone()
        if row is None:
            raise BetaError('401 Unauthorized','השיחה פגה. יש לפתוח שיחה חדשה.')
        return row

    def enqueue(self, session, body):
        question, request_id = body.get('message'), body.get('request_id')
        if not isinstance(question,str) or not 0<len(question.strip())<=2500:
            raise BetaError('400 Bad Request','יש לכתוב שאלה באורך של עד 2,500 תווים.')
        if not isinstance(request_id,str) or not re.fullmatch(r'[a-zA-Z0-9-]{16,80}',request_id):
            raise BetaError('400 Bad Request','מזהה הבקשה אינו תקין.')
        question = question.strip()
        sid = session['id']
        jid = hashlib.sha256((sid+':'+request_id).encode()).hexdigest()
        local = bot.greeting_reply(question)
        with self.db() as db:
            db.execute('BEGIN IMMEDIATE')
            existing = db.execute('SELECT * FROM jobs WHERE id=?',(jid,)).fetchone()
            if existing:
                if existing['question'] != question:
                    raise BetaError('409 Conflict','מזהה הבקשה כבר שייך לשאלה אחרת.')
                return {'id':jid,'state':existing['state']}
            if db.execute('SELECT count(*) FROM jobs WHERE session=?',(sid,)).fetchone()[0]>=100:
                raise BetaError('429 Too Many Requests','מכסת ההודעות לשיחה הזאת מוצתה.')
            if db.execute("SELECT 1 FROM jobs WHERE session=? AND state IN ('queued','running')",(sid,)).fetchone():
                raise BetaError('409 Conflict','יש להמתין למענה לשאלה הקודמת.')
            if db.execute("SELECT count(*) FROM jobs WHERE state IN ('queued','running')").fetchone()[0]>=8:
                raise BetaError('429 Too Many Requests','התור מלא כרגע. נסו שוב בעוד כמה דקות.')
            if local is None:
                day = datetime.now(timezone.utc).date().isoformat()
                total = db.execute('SELECT coalesce(sum(count),0) FROM quota WHERE day=?',(day,)).fetchone()[0]
                own = db.execute('SELECT count FROM quota WHERE day=? AND session=?',(day,sid)).fetchone()
                if total>=self.total_limit or (own and own[0]>=self.session_limit):
                    raise BetaError('429 Too Many Requests','מכסת הבטא להיום מוצתה. אפשר לחזור מחר.')
                db.execute('INSERT INTO quota VALUES (?,?,1) ON CONFLICT(day,session) DO UPDATE SET count=count+1',(day,sid))
            db.execute('INSERT INTO jobs(id,session,question,state,created,result) VALUES (?,?,?,?,?,?)',
                       (jid,sid,question,'done' if local else 'queued',time.time(),local or ''))
        self.wake.set()
        return {'id':jid,'state':'done' if local else 'queued'}

    def job(self, sid, jid):
        with self.db() as db:
            row = db.execute('SELECT state,result FROM jobs WHERE id=? AND session=?',(jid,sid)).fetchone()
        if row is None:
            raise BetaError('404 Not Found','הבקשה אינה זמינה בשיחה הזאת.')
        return {'id':jid,'state':row['state'],'text':row['result']}

    def reset(self, sid):
        with self.db() as db:
            db.execute('BEGIN IMMEDIATE')
            if db.execute("SELECT 1 FROM jobs WHERE session=? AND state IN ('queued','running')",(sid,)).fetchone():
                raise BetaError('409 Conflict','אפשר למחוק את השיחה אחרי סיום המענה הנוכחי.')
            db.execute('UPDATE sessions SET history=? WHERE id=?',('[]',sid))
            db.execute('DELETE FROM jobs WHERE session=?',(sid,))
        return {'ok':True}

    def once(self):
        with self.db() as db:
            db.execute('BEGIN IMMEDIATE')
            job = db.execute("SELECT j.*,s.history FROM jobs j JOIN sessions s ON s.id=j.session WHERE j.state='queued' AND s.expires>? ORDER BY j.created LIMIT 1",(time.time(),)).fetchone()
            if job is None:
                return False
            db.execute("UPDATE jobs SET state='running' WHERE id=?",(job['id'],))
        try:
            if not self.service.engine.ready():
                raise RuntimeError('corpus_unavailable')
            result = self.service.engine.answer(job['question'],json.loads(job['history']))
            text = server.format_reply(result,self.service.links)
            history = json.loads(job['history'])
            compact = {'text':result['text'],'sources':[
                {k:s[k] for k in ('id','file','page')} for s in result.get('sources',[])]}
            bot.remember(history,job['question'],compact)
            with self.db() as db:
                db.execute("UPDATE jobs SET state='done',result=? WHERE id=?",(text,job['id']))
                db.execute('UPDATE sessions SET history=? WHERE id=?',(json.dumps(history,ensure_ascii=False),job['session']))
        except Exception as exc:
            LOG.warning('beta_answer_failed:%s',type(exc).__name__)
            with self.db() as db:
                db.execute("UPDATE jobs SET state='failed',result=? WHERE id=?",('המענה נקטע. אפשר לנסות שוב; לא בוצע ניסיון חוזר אוטומטי.',job['id']))
        return True

    def start(self):
        with self.db() as db:
            # A restart never replays a possibly billed request.
            db.execute("UPDATE jobs SET state='failed',result=? WHERE state='running'",('המענה נקטע בעקבות אתחול השרת. אפשר לשלוח את השאלה שוב.',))
        def loop():
            last_cleanup = 0
            while not self.stop.is_set():
                try:
                    if time.time()-last_cleanup>60:
                        self.cleanup()
                        last_cleanup=time.time()
                    if not self.once():
                        self.wake.wait(1)
                        self.wake.clear()
                except Exception as exc:
                    LOG.error('beta_worker_failed:%s',type(exc).__name__)
                    return
        self.thread=threading.Thread(target=loop,daemon=True)
        self.thread.start()

    def __call__(self,environ,start_response):
        path,method=environ.get('PATH_INFO',''),environ.get('REQUEST_METHOD','GET')
        if not (path=='/beta' or path.startswith('/beta/')):
            return self.service(environ,start_response)
        def respond(status,data,html=False):
            raw=data if html else json.dumps(data,ensure_ascii=False).encode()
            headers=[('Content-Type','text/html; charset=utf-8' if html else 'application/json; charset=utf-8'),
                     ('Content-Length',str(len(raw))),('Cache-Control','no-store'),
                     ('X-Content-Type-Options','nosniff'),('Referrer-Policy','no-referrer'),
                     ('Content-Security-Policy',"default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; img-src 'self' data:; base-uri 'none'; form-action 'none'; frame-ancestors 'self' https://itamar.club https://www.itamar.club https://orharambam.com https://www.orharambam.com https://*.squarespace.com")]
            start_response(status,headers)
            return [raw]
        try:
            if path in ('/beta','/beta/') and method=='GET':
                return respond('200 OK',self.html,html=True)
            if method=='POST':
                try:
                    length=int(environ.get('CONTENT_LENGTH') or '0')
                    if not 0<length<=MAX_BODY:raise ValueError()
                    if environ.get('CONTENT_TYPE','').split(';')[0].strip()!='application/json':raise ValueError()
                    body=json.loads(environ['wsgi.input'].read(length))
                    if not isinstance(body,dict):raise ValueError()
                except (ValueError,TypeError,UnicodeError):
                    raise BetaError('400 Bad Request','הבקשה אינה תקינה.') from None
            if path=='/beta/api/session' and method=='POST':
                if body.get('consent') is not True:
                    raise BetaError('400 Bad Request','יש לאשר את תנאי השימוש והפרטיות לפני פתיחת שיחה.')
                return respond('201 Created',self.create_session())
            session=self.session(environ)
            if path=='/beta/api/chat' and method=='POST':
                return respond('202 Accepted',self.enqueue(session,body))
            if path=='/beta/api/reset' and method=='POST':
                return respond('200 OK',self.reset(session['id']))
            if path.startswith('/beta/api/jobs/') and method=='GET':
                jid=path.removeprefix('/beta/api/jobs/')
                if re.fullmatch(r'[a-f0-9]{64}',jid):
                    return respond('200 OK',self.job(session['id'],jid))
            raise BetaError('404 Not Found','העמוד אינו קיים.')
        except BetaError as exc:
            return respond(exc.status,{'error':exc.message})
        except (sqlite3.Error,OSError):
            return respond('503 Service Unavailable',{'error':'השירות אינו זמין כרגע. נסו שוב בהמשך.'})


def main():
    os.umask(0o077)
    logging.basicConfig(level=logging.WARNING,format='%(levelname)s %(message)s')
    cfg=server.Config.from_env()
    server.seed_data(cfg.data_dir)
    engine=LockedEngine(server.Engine(cfg))
    service=server.Service(cfg,engine=engine)
    engine.ready()
    service.start()
    beta=BetaApp(service,int(os.environ.get('BETA_DAILY_TOTAL_LIMIT','20')),
                 int(os.environ.get('BETA_DAILY_SESSION_LIMIT','5')))
    beta.start()
    from waitress import serve
    serve(beta,host='0.0.0.0',port=int(os.environ.get('PORT','10000')),threads=4,
          max_request_body_size=server.MAX_BODY,channel_timeout=30,expose_tracebacks=False)


if __name__=='__main__':
    main()
