
# -*- coding: utf-8 -*-
# ==========================================
# 🎤 旅遊語音小管家（外部連結 OK · 科技風 UI · System Prompt 預設摺疊）
# ==========================================
# - 「開始錄音」→「停止並送出」
# - Whisper 轉文字 → GPT 回答 → GPT TTS 語音回覆
# - 自動偵測地圖/導航需求，補上 Google Maps 外部連結（HTML <a>，target=_blank）
# - Chatbot 使用 render_markdown=False（直接渲染 HTML），避免被 Render 路由吃掉
# - System Prompt 以 Accordion 預設摺疊
# - 相容 Gradio 3.41（事件 I/O 與 _js 回傳值對齊）
# ==========================================

import os, re, base64, tempfile, urllib.parse
from datetime import datetime
import gradio as gr
from fastapi import FastAPI
from openai import OpenAI, APIConnectionError, APIStatusError, BadRequestError, AuthenticationError

# -------------------------
# 🔗 連結工具（HTML 模式，避免相對路徑誤判）
# -------------------------
URL_RE = re.compile(r'(https?://[^\s)]+)')

def html_linkify(text: str) -> str:
    """把純網址轉成 HTML 連結，target=_blank，避免 Render 把連結當站內路徑。"""
    if not text:
        return ""
    def _to_a(m):
        u = m.group(1)
        if not u.startswith("http"):
            u = "https://" + u.lstrip("/")
        return f'<a href="{u}" target="_blank" rel="noopener noreferrer">{u}</a>'
    return URL_RE.sub(_to_a, text)

def gmaps_search_link(q: str) -> str:
    return "https://www.google.com/maps/search/?api=1&query=" + urllib.parse.quote(q or "")

def needs_map_link(s: str) -> bool:
    if not s:
        return False
    s2 = s.lower()
    keys = ["google map", "google maps", "地圖", "導航", "怎麼走", "路線", "帶我去", "map", "maps"]
    return any(k in s2 for k in keys)

def maybe_append_map_link(user_utterance: str, bot_text: str) -> str:
    if not needs_map_link(user_utterance):
        return bot_text
    if URL_RE.search(bot_text or ""):
        return bot_text
    url = gmaps_search_link(user_utterance)
    return (bot_text or "") + f'\n\n➡️ 快速開啟地圖：<a href="{url}" target="_blank" rel="noopener noreferrer">{user_utterance} · Google 地圖</a>'

# -------------------------
# 🔑 初始化
# -------------------------
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ 請先設定環境變數 OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

VOICE_CHOICES = ["alloy","ash","ballad","coral","echo","fable","nova","onyx","sage","shimmer","verse"]
FIXED_TTS_MODEL = "gpt-4o-mini-tts"
LANG_CHOICES = ["auto","中文(zh)","英文(en)","泰文(th)","日文(ja)","韓文(ko)","法文(fr)","德文(de)","西班牙文(es)","越南文(vi)"]
LANG_MAP = {"auto":"auto","中文(zh)":"zh","英文(en)":"en","泰文(th)":"th","日文(ja)":"ja","韓文(ko)":"ko","法文(fr)":"fr","德文(de)":"de","西班牙文(es)":"es","越南文(vi)":"vi"}

DEFAULT_SYSTEM_PROMPT = """你是旅遊語音小管家，回答精簡、實用，使用繁體中文。
當使用者索取地圖、位置或導航時，請輸出外部連結（完整 https）並以清楚標題呈現。
若只有地名/關鍵字，請輸出：
  https://www.google.com/maps/search/?api=1&query=<urlencoded_place>
若有起訖點與交通方式，可輸出：
  https://www.google.com/maps/dir/?api=1&origin=<o>&destination=<d>&travelmode=<driving|walking|transit|bicycling>
回覆時可附上 1~3 個關聯景點的地圖連結。
"""

# -------------------------
# 🛡️ OpenAI 包裝與錯誤格式化
# -------------------------
def _fmt_err(e: Exception) -> str:
    if isinstance(e, AuthenticationError):
        return "⚠️ OpenAI 驗證失敗：請確認 OPENAI_API_KEY 是否正確。"
    if isinstance(e, BadRequestError):
        return f"⚠️ 參數錯誤：{getattr(e, 'message', str(e))}"
    if isinstance(e, APIStatusError):
        return f"⚠️ 服務狀態錯誤：{e.status_code} {getattr(e, 'message', str(e))}"
    if isinstance(e, APIConnectionError):
        return "⚠️ 無法連線到 OpenAI 服務，請稍後再試或檢查網路/防火牆。"
    return f"⚠️ 未預期錯誤：{str(e)}"

def safe_chat(user_text: str, system_prompt: str):
    try:
        sys_prompt = (system_prompt or "").strip() or DEFAULT_SYSTEM_PROMPT
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":sys_prompt},{"role":"user","content":user_text}],
            temperature=0.5,
        )
        answer = resp.choices[0].message.content.strip()
        return answer, None
    except Exception as e:
        return None, _fmt_err(e)

def safe_transcribe_file(path: str, lang_label: str):
    try:
        lang = LANG_MAP.get(lang_label, "auto")
        kwargs = {"model": "whisper-1"}
        if lang != "auto":
            kwargs["language"] = lang
        with open(path, "rb") as f:
            result = client.audio.transcriptions.create(file=f, **kwargs)
        return (getattr(result, "text", "") or "").strip(), None
    except Exception as e:
        return None, _fmt_err(e)

def safe_tts(text: str, voice_name: str):
    try:
        if not text.strip():
            return None, None
        out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
        with client.audio.speech.with_streaming_response.create(
            model=FIXED_TTS_MODEL, voice=voice_name, input=text
        ) as r:
            r.stream_to_file(out)
        return out, None
    except Exception as e:
        return None, _fmt_err(e)

# -------------------------
# 🧰 JS 錄音資料處理 → 後端檔案
# -------------------------
def write_dataurl_to_file(data_url: str) -> str:
    if not data_url or "," not in data_url:
        raise ValueError("無效的音訊資料。")
    header, b64 = data_url.split(",", 1)
    ext = ".webm"
    if "audio/ogg" in header: ext = ".ogg"
    elif "audio/mpeg" in header or "audio/mp3" in header: ext = ".mp3"
    elif "audio/wav" in header: ext = ".wav"
    raw = base64.b64decode(b64)
    path = tempfile.NamedTemporaryFile(delete=False, suffix=ext).name
    with open(path, "wb") as f:
        f.write(raw)
    return path

# -------------------------
# 🚀 事件：文字輸入
# -------------------------
def on_send_text(msg, history, voice_name, system_prompt):
    if not (msg or "").strip():
        return history, history, "", None, "請輸入訊息或使用語音～", ""
    bot, err = safe_chat(msg, system_prompt)
    audio_path = None
    if err:
        bot = err
    else:
        bot = maybe_append_map_link(msg, bot)
        bot = html_linkify(bot)
        audio_path, tts_err = safe_tts(bot, voice_name)
        if tts_err:
            bot += f"\n\n（語音產生失敗：{tts_err}）"
    history = (history or []) + [(msg, bot)]
    return history, history, "", audio_path, bot, ""

# -------------------------
# 🛎️ 事件：停止並送出（錄音→Whisper→GPT→TTS）
# -------------------------
def on_audio_dataurl_received(audio_b64_dataurl, history, voice_name, system_prompt, whisper_lang_label):
    if not audio_b64_dataurl:
        return history, history, None, "沒有錄到音，請重試。"
    try:
        path = write_dataurl_to_file(audio_b64_dataurl)
    except Exception as e:
        return history, history, None, _fmt_err(e)

    text, terr = safe_transcribe_file(path, whisper_lang_label)
    if terr:
        return history, history, None, terr

    bot, cerr = safe_chat(text, system_prompt)
    audio_path = None
    if cerr:
        bot = cerr
    else:
        bot = maybe_append_map_link(text, bot)
        bot = html_linkify(bot)
        audio_path, aerr = safe_tts(bot, voice_name)
        if aerr:
            bot += f"\n\n（語音產生失敗：{aerr}）"
    history = (history or []) + [(f"(語音提問)\n{text}", bot)]
    return history, history, audio_path, "已停止並送出"

def export_chat(history):
    lines = [f"旅遊語音小管家對話匯出 - {datetime.now():%Y-%m-%d %H:%M:%S}", "="*60]
    for user, bot in history or []:
        lines += ["使用者：", user or "", "小管家：", bot or "", "-"*40]
    path = tempfile.NamedTemporaryFile(delete=False, suffix=".txt").name
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path

def clear_history_both():
    return [], []

# -------------------------
# 🎨 UI 與 JS（科技風樣式 + 錄音控制）
# -------------------------
CSS_TECH = """
:root{
  --bg:#0a1120; --panel:#0f1b33cc; --stroke:#1e2b4d;
  --text:#e8eefc; --muted:#9bb0d6; --accent:#54b7ff; --accent-2:#00ffd0;
}
.gradio-container{font-family:ui-sans-serif,system-ui,PingFangTC,'Noto Sans TC',Segoe UI,Roboto,Helvetica,Arial;color:var(--text)}
body{background:radial-gradient(1200px 600px at 20% -10%, #11315d55, transparent),linear-gradient(180deg,#0a1120 0%, #0a1120 100%);}
.neon-panel{background:var(--panel);border:1px solid var(--stroke);box-shadow:0 0 0 1px #0e1a33 inset,0 10px 30px #0008;border-radius:16px;padding:16px;}
button.primary{background:linear-gradient(90deg,var(--accent),var(--accent-2));color:#00121d;font-weight:700;border-radius:12px!important}
.badge{background:#112a49;color:#8bd9ff;padding:2px 8px;border:1px solid #1f3c66;border-radius:999px;font-size:12px;margin-left:8px}
a{color:#7fd0ff;text-decoration:underline}
"""

# JS：開始錄音（對齊 1 個輸出 mic_status）
JS_START_RECORD = """
async () => {
  try {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      return ['不支援'];
    }
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mr = new MediaRecorder(stream);
    window.__mr_chunks = [];
    window.__mr_stream = stream;
    window.__mr = mr;
    mr.ondataavailable = (e)=>{ if (e.data && e.data.size) window.__mr_chunks.push(e.data); };
    mr.start();
    return ['錄音中...'];
  } catch (e) {
    return ['權限被拒或裝置不可用'];
  }
}
"""

# JS：停止並送出（回傳陣列以對齊 5 個 *輸入*；僅替換第一個為 dataURL）
JS_STOP_AND_EXPORT = """
async (dataurl_box, history, voice, system_prompt, lang) => {
  try{
    const mr = window.__mr;
    const stream = window.__mr_stream;
    if (!mr) { return [null, history, voice, system_prompt, lang]; }
    return await new Promise(resolve => {
      mr.onstop = async () => {
        try{
          const blob = new Blob(window.__mr_chunks || [], { type: 'audio/webm;codecs=opus' });
          if (stream) { stream.getTracks().forEach(t=>t.stop()); }
          window.__mr = null; window.__mr_stream = null; window.__mr_chunks = null;
          const dataUrl = await new Promise((res,rej)=>{
            const reader = new FileReader();
            reader.onloadend = () => res(reader.result);
            reader.onerror = rej;
            reader.readAsDataURL(blob);
          });
          resolve([dataUrl, history, voice, system_prompt, lang]);
        }catch(err){
          resolve([null, history, voice, system_prompt, lang]);
        }
      };
      mr.stop();
    });
  }catch(e){
    return [null, history, voice, system_prompt, lang];
  }
}
"""

with gr.Blocks(title="旅遊語音小管家", css=CSS_TECH) as demo:
    gr.Markdown("## 🎤 旅遊語音小管家 <span class='badge'>外部連結 OK · 按一下開始錄音 → 再按一下停止並送出</span>")
    with gr.Row(equal_height=True):
        with gr.Column(scale=3, elem_classes=["neon-panel"]):
            chatbot = gr.Chatbot(label="對話區", height=520, render_markdown=False)
            history_state = gr.State([])
            user_text = gr.Textbox(placeholder="輸入文字...", label="文字訊息", lines=2)
            with gr.Row():
                send_btn = gr.Button("🚀 送出文字", elem_classes=["primary"])
                clear_btn = gr.Button("🧹 清空對話")
            tts_output = gr.Audio(label="🔊 回覆語音", type="filepath", interactive=False)
            latest_text = gr.Textbox(label="📝 送出內容（最新輪）", interactive=False)
            error_box = gr.Textbox(label="⚠️ 訊息 / 錯誤提示", interactive=False)
            with gr.Row():
                export_btn = gr.Button("📝 匯出對話（.txt）")
                export_file = gr.File(label="下載檔案", visible=True)

        with gr.Column(scale=2, elem_classes=["neon-panel"]):
            gr.Markdown("### 🎙️ 語音直送（按鈕控制）")
            with gr.Row():
                start_btn = gr.Button("🎙️ 開始錄音", elem_classes=["primary"])
                stop_send_btn = gr.Button("⏹️ 停止並送出")
            mic_status = gr.Textbox(value="尚未錄音", label="狀態", interactive=False)
            audio_dataurl_box = gr.Textbox(visible=False)

            gr.Markdown("---")
            gr.Markdown("### ⚙️ 偏好設定")
            voice_dropdown = gr.Dropdown(choices=VOICE_CHOICES, value="alloy", label="語音包（GPT TTS）")
            whisper_lang_dd = gr.Dropdown(choices=LANG_CHOICES, value="auto", label="Whisper 語言")
            with gr.Accordion("🔒 系統提示詞（開發者設定）", open=False):
                system_prompt_tb = gr.Textbox(value=DEFAULT_SYSTEM_PROMPT, label="System Prompt（系統提示詞）", lines=4)

    # 綁定事件
    send_btn.click(on_send_text, [user_text, history_state, voice_dropdown, system_prompt_tb],
                   [chatbot, history_state, user_text, tts_output, latest_text, error_box])
    clear_btn.click(clear_history_both, None, [chatbot, history_state])
    start_btn.click(fn=lambda: None, inputs=None, outputs=[mic_status], _js=JS_START_RECORD)
    stop_send_btn.click(on_audio_dataurl_received,
                        [audio_dataurl_box, history_state, voice_dropdown, system_prompt_tb, whisper_lang_dd],
                        [chatbot, history_state, tts_output, mic_status],
                        _js=JS_STOP_AND_EXPORT)
    export_btn.click(export_chat, [history_state], [export_file])

# FastAPI mount for Render / uvicorn
fastapi_app = FastAPI()
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=port)
