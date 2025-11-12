# -*- coding: utf-8 -*-
# 🧳 旅遊小管家 Pro（按鈕錄音版 + Google Maps 連結 + Markdown超連結）
# ==========================================
import os, re, base64, tempfile, urllib.parse
from datetime import datetime
import gradio as gr
from openai import OpenAI
from fastapi import FastAPI

URL_RE = re.compile(r'(https?://[^\s)]+)')
def linkify(text:str)->str:
    if not text: return ''
    return URL_RE.sub(lambda m: f'[{m.group(1)}]({m.group(1)})', text)

def gmaps_search_link(q:str)->str:
    return f"https://www.google.com/maps/search/?api=1&query={urllib.parse.quote(q)}"

def needs_map_link(s:str)->bool:
    if not s: return False
    s2=s.lower()
    for k in ['google map','地圖','導航','maps','怎麼走','路線','帶我去']: 
        if k in s2: return True
    return False

def maybe_append_map_link(user:str,bot:str)->str:
    if not needs_map_link(user): return bot
    if URL_RE.search(bot or ''): return bot
    return (bot or '')+f"\n\n➡️ 快速開啟地圖：[{user} · Google 地圖]({gmaps_search_link(user)})"

api_key=os.environ.get('OPENAI_API_KEY')
if not api_key: raise ValueError('❌ 請先設定 OPENAI_API_KEY')
client=OpenAI(api_key=api_key)
VOICE_CHOICES=["alloy","ash","ballad","coral","echo","fable","nova","onyx","sage","shimmer","verse"]
FIXED_TTS_MODEL="gpt-4o-mini-tts"
LANG_CHOICES=["auto","中文(zh)","英文(en)","泰文(th)","日文(ja)","韓文(ko)"]
LANG_MAP={"auto":"auto","中文(zh)":"zh","英文(en)":"en","泰文(th)":"th","日文(ja)":"ja","韓文(ko)":"ko"}
DEFAULT_SYSTEM_PROMPT='你是旅遊小管家，回答精簡、實用，使用繁體中文，允許直接輸出地圖Markdown連結。'

def safe_chat(txt,prompt):
    resp=client.chat.completions.create(model='gpt-4o-mini',messages=[{'role':'system','content':prompt or DEFAULT_SYSTEM_PROMPT},{'role':'user','content':txt}],temperature=0.5)
    return resp.choices[0].message.content.strip(),None

def safe_tts(text,voice):
    if not text.strip(): return None,None
    out=tempfile.NamedTemporaryFile(delete=False,suffix='.mp3').name
    with client.audio.speech.with_streaming_response.create(model=FIXED_TTS_MODEL,voice=voice,input=text) as r: r.stream_to_file(out)
    return out,None

def on_send_text(msg,hist,voice,prompt):
    if not msg.strip(): return hist,hist,'',None,'請輸入訊息',''
    bot,_=safe_chat(msg,prompt)
    bot=maybe_append_map_link(msg,bot)
    bot=linkify(bot)
    audio,_=safe_tts(bot,voice)
    hist=(hist or [])+[(msg,bot)]
    return hist,hist,'',audio,bot,''

CSS=':root{--bg:#0a1120;--panel:#0f1b33cc;--stroke:#1e2b4d;--accent:#54b7ff;--accent-2:#00ffd0;}body{background:#0a1120;}'
with gr.Blocks(title='旅遊小管家 Pro',css=CSS) as demo:
    chatbot=gr.Chatbot(label='對話區',height=520,render_markdown=True)
    state=gr.State([])
    txt=gr.Textbox(label='輸入文字')
    send=gr.Button('🚀 送出')
    clear=gr.Button('🧹 清空')
    audio=gr.Audio(label='語音回覆',type='filepath')
    voice=gr.Dropdown(choices=VOICE_CHOICES,value='alloy',label='語音包')
    prompt=gr.Textbox(value=DEFAULT_SYSTEM_PROMPT,label='系統提示',lines=3)
    send.click(on_send_text,[txt,state,voice,prompt],[chatbot,state,txt,audio])
    clear.click(lambda:([],[]),None,[chatbot,state])

app=gr.mount_gradio_app(FastAPI(),demo,path='/')
if __name__=='__main__':
    demo.launch(server_name='0.0.0.0',server_port=int(os.environ.get('PORT',7860)))
