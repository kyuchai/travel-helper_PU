
# -*- coding: utf-8 -*-
# ==========================================
# 🧳 旅遊小管家 Pro（即時錄製＆上傳：自動轉文字預覽；穩定加強版；Render/uvicorn）
# ==========================================
import os
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

def safe_transcribe(path: str, lang_label: str):
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
# 事件
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

def on_preview_voice(audio_file, whisper_lang_label):
    if not audio_file:
        return "尚未錄音或尚未選擇檔案。", gr.update(visible=True), None
    text, err = safe_transcribe(audio_file, whisper_lang_label)
    if err:
        return err, gr.update(visible=True), None
    return text, gr.update(visible=True), None

def on_send_voice(audio_file, pending_text, history, voice_name, system_prompt):
    text = (pending_text or "").strip()
    if not text and audio_file:
        text, err = safe_transcribe(audio_file, "auto")
        if err:
            return history, history, None, err, gr.update(visible=True), None
    if not text:
        return history, history, None, "沒有可送出的內容。", gr.update(visible=True), None

    bot, err = safe_chat(text, system_prompt)
    audio_path = None
    if err:
        bot = err
    else:
        audio_path, tts_err = safe_tts(bot, voice_name)
        if tts_err:
            bot += f"\n\n（語音產生失敗：{tts_err}）"
    history = (history or []) + [(f"(語音轉文字)\n{text}", bot)]
    return history, history, audio_path, "", gr.update(visible=False), None

def on_clear_pending(_):
    return "", gr.update(visible=False), None

def clear_audio(_):
    return None

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
# UI
# -------------------------
CSS_TECH = """
:root{--bg:#0a1120;--panel:#0f1b33cc;--stroke:#1e2b4d;--text:#e8eefc;--muted:#9bb0d6;--accent:#54b7ff;--accent-2:#00ffd0;}
.gradio-container{font-family:ui-sans-serif,system-ui,PingFangTC,'Noto Sans TC',Segoe UI,Roboto,Helvetica,Arial;}
body{background:radial-gradient(1200px 600px at 20% -10%, #11315d55, transparent),linear-gradient(180deg,#0a1120 0%, #0a1120 100%);}
.neon-panel{background:var(--panel);border:1px solid var(--stroke);box-shadow:0 0 0 1px #0e1a33 inset,0 10px 30px #0008;border-radius:16px;padding:16px;}
button.primary{background:linear-gradient(90deg,var(--accent),var(--accent-2));color:#00121d;font-weight:700;border-radius:12px!important}
"""

with gr.Blocks(title="旅遊小管家 Pro（即時錄製＆上傳）", css=CSS_TECH) as demo:
    gr.Markdown("## 🧳 旅遊小管家 Pro  <span class='badge'>錄音即轉文字 × 檔案上傳即預覽</span>")
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
            gr.Markdown("### 🎤 語音輸入（錄音或上傳後自動轉文字→預覽→再決定送出）")

            mic_audio = gr.Audio(sources=["microphone","upload"], type="filepath",
                                 label="錄音或上傳檔案（停止錄音或選擇檔案後會自動轉文字並顯示於下方）")

            pending_box = gr.Textbox(label="🕒 尚未送出的語音文字（請檢查內容）", visible=False, lines=3)
            with gr.Row():
                send_voice_btn = gr.Button("✅ 送出語音內容", elem_classes=["primary"])
                cancel_pending_btn = gr.Button("❎ 取消（重錄）")
                drop_btn = gr.Button("🗑️ 清除音訊")

            gr.Markdown("---")
            gr.Markdown("### ⚙️ 偏好設定")
            voice_dropdown = gr.Dropdown(choices=VOICE_CHOICES, value="alloy", label="語音包")
            whisper_lang_dd = gr.Dropdown(choices=LANG_CHOICES, value="auto", label="Whisper 語言")
            system_prompt_tb = gr.Textbox(value=DEFAULT_SYSTEM_PROMPT, label="System Prompt（系統提示詞）", lines=3)

    # 綁定事件
    send_btn.click(on_send_text, [user_text, history_state, voice_dropdown, system_prompt_tb],
                   [chatbot, history_state, user_text, tts_output, latest_text, error_box])
    clear_btn.click(clear_history_both, None, [chatbot, history_state])

    # ✅ 「即時錄製＆上傳」：Audio 元件變動就自動做預覽/轉文字
    mic_audio.change(on_preview_voice, [mic_audio, whisper_lang_dd], [pending_box, pending_box, tts_output])

    # 送出 / 取消 / 清除
    send_voice_btn.click(on_send_voice, [mic_audio, pending_box, history_state, voice_dropdown, system_prompt_tb],
                         [chatbot, history_state, tts_output, pending_box, pending_box, mic_audio])
    cancel_pending_btn.click(on_clear_pending, [pending_box], [pending_box, pending_box, mic_audio])
    drop_btn.click(clear_audio, [mic_audio], [mic_audio])

    export_btn.click(export_chat, [history_state], [export_file])

# -------------------------
# FastAPI app (for Render / uvicorn)
# -------------------------
fastapi_app = FastAPI()
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=port)
