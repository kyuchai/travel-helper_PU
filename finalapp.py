
# -*- coding: utf-8 -*-
# ==========================================
# 🧳 旅遊小管家 Pro（按鈕錄音版：開始錄音→停止並送出；Render/uvicorn）
# ==========================================
import os
import io
import base64
import tempfile
from datetime import datetime

import gradio as gr
from openai import OpenAI, APIConnectionError, APIStatusError, BadRequestError, AuthenticationError
from fastapi import FastAPI

# -------------------------
# 初始化
# -------------------------
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ 請先設定環境變數 OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

VOICE_CHOICES = ["alloy","ash","ballad","coral","echo","fable","nova","onyx","sage","shimmer","verse"]
FIXED_TTS_MODEL = "gpt-4o-mini-tts"
LANG_CHOICES = ["auto","中文(zh)","英文(en)","泰文(th)","日文(ja)","韓文(ko)","法文(fr)","德文(de)","西班牙文(es)","越南文(vi)"]
LANG_MAP = {"auto":"auto","中文(zh)":"zh","英文(en)":"en","泰文(th)":"th","日文(ja)":"ja","韓文(ko)":"ko","法文(fr)":"fr","德文(de)":"de","西班牙文(es)":"es","越南文(vi)":"vi"}
DEFAULT_SYSTEM_PROMPT = "你是旅遊小管家，回答精簡、實用，使用繁體中文。"

# -------------------------
# 錯誤格式化
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
        return resp.choices[0].message.content.strip(), None
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
# 小工具
# -------------------------
def write_dataurl_to_file(data_url: str) -> str:
    """
    data_url 形如: 'data:audio/webm;codecs=opus;base64,AAAA...' -> 寫到臨時 .webm
    """
    if not data_url or "," not in data_url:
        raise ValueError("無效的音訊資料。")
    header, b64 = data_url.split(",", 1)
    ext = ".webm"
    if "audio/ogg" in header:
        ext = ".ogg"
    elif "audio/mpeg" in header or "audio/mp3" in header:
        ext = ".mp3"
    elif "audio/wav" in header:
        ext = ".wav"
    raw = base64.b64decode(b64)
    path = tempfile.NamedTemporaryFile(delete=False, suffix=ext).name
    with open(path, "wb") as f:
        f.write(raw)
    return path

# -------------------------
# 事件：文字輸入
# -------------------------
def on_send_text(msg, history, voice_name, system_prompt):
    if not (msg or "").strip():
        return history, history, "", None, "請輸入訊息或使用語音～", ""
    bot, err = safe_chat(msg, system_prompt)
    audio_path = None
    if err:
        bot = err
    else:
        audio_path, tts_err = safe_tts(bot, voice_name)
        if tts_err:
            bot += f"\n\n（語音產生失敗：{tts_err}）"
    history = (history or []) + [(msg, bot)]
    return history, history, "", audio_path, bot, ""

# -------------------------
# 事件：停止並送出（從 dataURL -> 檔案 -> Whisper -> Chat -> TTS）
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
        audio_path, aerr = safe_tts(bot, voice_name)
        if aerr:
            bot += f"\n\n（語音產生失敗：{aerr}）"
    history = (history or []) + [(f"(語音提問)\n{text}", bot)]
    return history, history, audio_path, ""

def export_chat(history):
    try:
        if not history:
            return None
        lines = [f"旅遊小管家對話匯出 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "="*60]
        for user, bot in history:
            lines += ["使用者：", user or "", "小管家：", bot or "", "-"*40]
        path = tempfile.NamedTemporaryFile(delete=False, suffix=".txt").name
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return path
    except Exception as e:
        path = tempfile.NamedTemporaryFile(delete=False, suffix=".txt").name
        with open(path, "w", encoding="utf-8") as f:
            f.write(_fmt_err(e))
        return path

def clear_history_both():
    return [], []

# -------------------------
# UI 與 JS
# -------------------------
CSS_TECH = """
:root{--bg:#0a1120;--panel:#0f1b33cc;--stroke:#1e2b4d;--text:#e8eefc;--muted:#9bb0d6;--accent:#54b7ff;--accent-2:#00ffd0;}
.gradio-container{font-family:ui-sans-serif,system-ui,PingFangTC,'Noto Sans TC',Segoe UI,Roboto,Helvetica,Arial;}
body{background:radial-gradient(1200px 600px at 20% -10%, #11315d55, transparent),linear-gradient(180deg,#0a1120 0%, #0a1120 100%);}
.neon-panel{background:var(--panel);border:1px solid var(--stroke);box-shadow:0 0 0 1px #0e1a33 inset,0 10px 30px #0008;border-radius:16px;padding:16px;}
button.primary{background:linear-gradient(90deg,var(--accent),var(--accent-2));color:#00121d;font-weight:700;border-radius:12px!important}
"""

# JS：使用 MediaRecorder 控制開始/停止，並把 DataURL 回傳給 Python
JS_START_RECORD = """
async () => {
  try {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      return ['不支援', null];
    }
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mr = new MediaRecorder(stream);
    window.__mr_chunks = [];
    window.__mr_stream = stream;
    window.__mr = mr;
    mr.ondataavailable = (e)=>{ if (e.data && e.data.size) window.__mr_chunks.push(e.data); };
    mr.start();
    return ['錄音中...', null];
  } catch (e) {
    return ['權限被拒或裝置不可用', null];
  }
}
"""

JS_STOP_AND_EXPORT = """
async () => {
  try{
    const mr = window.__mr;
    const stream = window.__mr_stream;
    if (!mr) { return [ '尚未開始錄音', null ]; }
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
          resolve(['已停止並送出', dataUrl]);
        }catch(err){
          resolve([ '轉檔失敗', null ]);
        }
      };
      mr.stop();
    });
  }catch(e){
    return [ '停止失敗', null ];
  }
}
"""

with gr.Blocks(title="旅遊小管家 Pro（按鈕錄音版）", css=CSS_TECH) as demo:
    gr.Markdown("## 🧳 旅遊小管家 Pro  <span class='badge'>按一下開始錄音 → 再按一下停止並送出</span>")
    with gr.Row(equal_height=True):
        # 左：聊天
        with gr.Column(scale=3, elem_classes=["neon-panel"]):
            chatbot = gr.Chatbot(label="對話區", height=520)
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

        # 右：語音 & 設定
        with gr.Column(scale=2, elem_classes=["neon-panel"]):
            gr.Markdown("### 🎤 語音直送（按鈕控制）")

            # 錄音控制與狀態
            with gr.Row():
                start_btn = gr.Button("🎙️ 開始錄音", elem_classes=["primary"])
                stop_send_btn = gr.Button("⏹️ 停止並送出")
            mic_status = gr.Textbox(value="尚未錄音", label="狀態", interactive=False)

            # 由 JS 送來的 dataURL（隱藏）
            audio_dataurl_box = gr.Textbox(visible=False)

            gr.Markdown("---")
            gr.Markdown("### ⚙️ 偏好設定")
            voice_dropdown = gr.Dropdown(choices=VOICE_CHOICES, value="alloy", label="語音包")
            whisper_lang_dd = gr.Dropdown(choices=LANG_CHOICES, value="auto", label="Whisper 語言")
            system_prompt_tb = gr.Textbox(value=DEFAULT_SYSTEM_PROMPT, label="System Prompt（系統提示詞）", lines=3)

    # 事件：文字聊天
    send_btn.click(on_send_text, [user_text, history_state, voice_dropdown, system_prompt_tb],
                   [chatbot, history_state, user_text, tts_output, latest_text, error_box])
    clear_btn.click(clear_history_both, None, [chatbot, history_state])

    # 事件：開始錄音（純前端）
    start_btn.click(lambda: ("錄音中...", None), None, [mic_status, audio_dataurl_box], _js=JS_START_RECORD)

    # 事件：停止並送出（前端停止並傳 dataURL → 後端轉文字/聊天/TTS）
    stop_send_btn.click(on_audio_dataurl_received,
                        [audio_dataurl_box, history_state, voice_dropdown, system_prompt_tb, whisper_lang_dd],
                        [chatbot, history_state, tts_output, mic_status],
                        _js=JS_STOP_AND_EXPORT)

    # 匯出
    export_btn.click(export_chat, [history_state], [export_file])

# -------------------------
# FastAPI app (for Render / uvicorn)
# -------------------------
fastapi_app = FastAPI()
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=port)
