
# -*- coding: utf-8 -*-
# ==========================================
# 🧳 旅遊小管家 Pro（穩定版：移除 load JS、修正預聽傳參；Render/uvicorn）
# ==========================================
import os
import tempfile
from datetime import datetime

import gradio as gr
from openai import OpenAI
from fastapi import FastAPI

# -------------------------
# 初始化
# -------------------------
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ 請先設定環境變數 OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

VOICE_CHOICES = ["alloy", "ash", "ballad", "coral", "echo", "fable", "nova", "onyx", "sage", "shimmer", "verse"]
FIXED_TTS_MODEL = "gpt-4o-mini-tts"
LANG_CHOICES = ["auto","中文(zh)","英文(en)","泰文(th)","日文(ja)","韓文(ko)","法文(fr)","德文(de)","西班牙文(es)","越南文(vi)"]
LANG_MAP = {"auto":"auto","中文(zh)":"zh","英文(en)":"en","泰文(th)":"th","日文(ja)":"ja","韓文(ko)":"ko","法文(fr)":"fr","德文(de)":"de","西班牙文(es)":"es","越南文(vi)":"vi"}
DEFAULT_SYSTEM_PROMPT = "你是旅遊小管家，回答精簡、實用，使用繁體中文。"

# -------------------------
# 功能函式
# -------------------------
def chat_answer(user_text: str, system_prompt: str) -> str:
    if not user_text.strip():
        return "請輸入訊息或使用語音～"
    sys_prompt = system_prompt.strip() or DEFAULT_SYSTEM_PROMPT
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"system","content":sys_prompt},{"role":"user","content":user_text}],
        temperature=0.5,
    )
    return resp.choices[0].message.content.strip()

def transcribe_audio_to_text(audio_file_path: str, lang_label: str) -> str:
    lang_code = LANG_MAP.get(lang_label, "auto")
    kwargs = {"model": "whisper-1"}
    if lang_code != "auto":
        kwargs["language"] = lang_code
    with open(audio_file_path, "rb") as f:
        result = client.audio.transcriptions.create(file=f, **kwargs)
    return (getattr(result, "text", "") or "").strip()

def text_to_speech(text: str, voice_name: str):
    if not text.strip():
        return None
    speech_file_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
    with client.audio.speech.with_streaming_response.create(model=FIXED_TTS_MODEL, voice=voice_name, input=text) as r:
        r.stream_to_file(speech_file_path)
    return speech_file_path

def on_send_text(msg, history, voice_name, system_prompt):
    bot_text = chat_answer(msg, system_prompt)
    history = (history or []) + [(msg, bot_text)]
    audio_path = text_to_speech(bot_text, voice_name)
    return history, history, "", audio_path, bot_text

def on_preview_voice(audio_file, whisper_lang_label):
    if not audio_file:
        return "", gr.update(visible=False), None
    text = transcribe_audio_to_text(audio_file, whisper_lang_label)
    return text, gr.update(visible=True), None

def on_send_voice(audio_file, pending_text, history, voice_name, system_prompt):
    text = (pending_text or "").strip()
    if not text and audio_file:
        text = transcribe_audio_to_text(audio_file, "auto")
    if not text:
        return history, history, None, "", gr.update(visible=False), None
    bot_text = chat_answer(text, system_prompt)
    history = (history or []) + [(f"(語音轉文字)\\n{text}", bot_text)]
    audio_path = text_to_speech(bot_text, voice_name)
    return history, history, audio_path, "", gr.update(visible=False), None

def on_clear_pending(_):
    return "", gr.update(visible=False), None

def clear_audio(_):
    return None

def export_chat(history):
    if not history:
        return None
    lines = [f"旅遊小管家對話匯出 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "="*60]
    for user, bot in history:
        lines += ["使用者：", user or "", "小管家：", bot or "", "-"*40]
    path = tempfile.NamedTemporaryFile(delete=False, suffix=".txt").name
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path

def clear_history_both():
    return [], []

def noop_return_none(*args, **kwargs):
    return None

CSS_TECH = """
:root{--bg:#0a1120;--panel:#0f1b33cc;--stroke:#1e2b4d;--text:#e8eefc;--muted:#9bb0d6;--accent:#54b7ff;--accent-2:#00ffd0;}
.gradio-container{font-family:ui-sans-serif,system-ui,PingFangTC,'Noto Sans TC',Segoe UI,Roboto,Helvetica,Arial;}
body{background:radial-gradient(1200px 600px at 20% -10%, #11315d55, transparent),linear-gradient(180deg,#0a1120 0%, #0a1120 100%);}
.neon-panel{background:var(--panel);border:1px solid var(--stroke);box-shadow:0 0 0 1px #0e1a33 inset,0 10px 30px #0008;border-radius:16px;padding:16px;}
button.primary{background:linear-gradient(90deg,var(--accent),var(--accent-2));color:#00121d;font-weight:700;border-radius:12px!important}
"""

# -------------------------
# Gradio 介面
# -------------------------
with gr.Blocks(title="旅遊小管家 Pro（穩定版）", css=CSS_TECH) as demo:
    gr.Markdown("## 🧳 旅遊小管家 Pro  <span class='badge'>語音互動 × 智慧回覆 × 偏好記憶</span>")
    with gr.Row(equal_height=True):
        with gr.Column(scale=3, elem_classes=["neon-panel"]):
            chatbot = gr.Chatbot(label="對話區", height=520)
            history_state = gr.State([])

            user_text = gr.Textbox(placeholder="輸入文字...", label="文字訊息", lines=2)
            with gr.Row():
                send_btn = gr.Button("🚀 送出文字", elem_classes=["primary"])
                clear_btn = gr.Button("🧹 清空對話")

            tts_output = gr.Audio(label="🔊 回覆語音", type="filepath", interactive=False)
            transcript_tb = gr.Textbox(label="📝 送出內容（最新輪）", interactive=False)

            with gr.Row():
                export_btn = gr.Button("📝 匯出對話（.txt）")
                export_file = gr.File(label="下載檔案", visible=True)

        with gr.Column(scale=2, elem_classes=["neon-panel"]):
            gr.Markdown("### 🎤 語音輸入（先錄音 → 預聽 → 再決定送出）")
            mic_audio = gr.Audio(sources=["microphone","upload"], type="filepath", label="錄音或上傳檔案")

            with gr.Row():
                preview_btn = gr.Button("👂 預聽 / 轉文字（尚未送出）", elem_classes=["primary"])
                redo_btn = gr.Button("🔁 重新錄製")
                drop_btn = gr.Button("🗑️ 清除音訊")

            pending_box = gr.Textbox(label="🕒 尚未送出的語音文字（請檢查內容）", visible=False, lines=3)
            with gr.Row():
                send_voice_btn = gr.Button("✅ 送出語音內容", elem_classes=["primary"])
                cancel_pending_btn = gr.Button("❎ 取消（重錄）")

            gr.Markdown("---")
            gr.Markdown("### ⚙️ 偏好設定")
            voice_dropdown = gr.Dropdown(choices=VOICE_CHOICES, value="alloy", label="語音包")
            whisper_lang_dd = gr.Dropdown(choices=LANG_CHOICES, value="auto", label="Whisper 語言")
            system_prompt_tb = gr.Textbox(value=DEFAULT_SYSTEM_PROMPT, label="System Prompt（系統提示詞）", lines=3)

            dummy_store = gr.Textbox(visible=False)

    # 綁定事件
    send_btn.click(on_send_text, [user_text, history_state, voice_dropdown, system_prompt_tb],
                   [chatbot, history_state, user_text, tts_output, transcript_tb])
    clear_btn.click(clear_history_both, None, [chatbot, history_state])

    preview_btn.click(on_preview_voice, [mic_audio, whisper_lang_dd], [pending_box, pending_box, tts_output])
    send_voice_btn.click(on_send_voice, [mic_audio, pending_box, history_state, voice_dropdown, system_prompt_tb],
                         [chatbot, history_state, tts_output, pending_box, pending_box, mic_audio])
    cancel_pending_btn.click(on_clear_pending, [pending_box], [pending_box, pending_box, mic_audio])
    redo_btn.click(clear_audio, [mic_audio], [mic_audio])
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
