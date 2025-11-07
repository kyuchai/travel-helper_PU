# -*- coding: utf-8 -*-
# ==========================================
# 🧳 旅遊小管家 Pro（Render 版：支援 uvicorn 啟動）
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

VOICE_CHOICES = [
    "alloy", "ash", "ballad", "coral", "echo",
    "fable", "nova", "onyx", "sage", "shimmer", "verse",
]
TTS_MODEL_CHOICES = ["gpt-4o-mini-tts", "tts-1", "tts-1-hd"]
WHISPER_LANG_CHOICES = ["auto", "zh", "en", "th", "ja", "ko", "fr", "de", "es", "vi"]
DEFAULT_SYSTEM_PROMPT = "你是旅遊小管家，回答精簡、實用，使用繁體中文。"

# -------------------------
# 函式：聊天
# -------------------------
def chat_answer(user_text: str, system_prompt: str) -> str:
    if not user_text.strip():
        return "請輸入訊息或使用語音～"
    sys_prompt = system_prompt.strip() or DEFAULT_SYSTEM_PROMPT
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_text}
        ],
        temperature=0.5,
    )
    return resp.choices[0].message.content.strip()

# -------------------------
# 函式：Whisper 語音轉文字
# -------------------------
def transcribe_audio_to_text(audio_file_path: str, lang_code: str) -> str:
    kwargs = {"model": "whisper-1"}
    if lang_code and lang_code != "auto":
        kwargs["language"] = lang_code
    with open(audio_file_path, "rb") as f:
        result = client.audio.transcriptions.create(file=f, **kwargs)
    return (getattr(result, "text", "") or "").strip()

# -------------------------
# 函式：TTS 語音合成
# -------------------------
def text_to_speech(text: str, voice_name: str, tts_model: str):
    if not text.strip():
        return None
    tts_model = tts_model or "gpt-4o-mini-tts"
    speech_file_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
    with client.audio.speech.with_streaming_response.create(
        model=tts_model, voice=voice_name, input=text
    ) as response:
        response.stream_to_file(speech_file_path)
    return speech_file_path

# -------------------------
# 聊天互動邏輯
# -------------------------
def on_send_text(msg, history, voice_name, tts_model, system_prompt):
    bot_text = chat_answer(msg, system_prompt)
    history = (history or []) + [(msg, bot_text)]
    audio_path = text_to_speech(bot_text, voice_name, tts_model)
    return history, "", audio_path, bot_text

def on_send_audio(audio_file, history, voice_name, tts_model, whisper_lang, system_prompt):
    if not audio_file:
        return history, None, None, gr.update(visible=True), gr.update(visible=True), ""
    text = transcribe_audio_to_text(audio_file, whisper_lang)
    bot_text = chat_answer(text, system_prompt)
    history = (history or []) + [(f"(語音轉文字)\n{text}", bot_text)]
    audio_path = text_to_speech(bot_text, voice_name, tts_model)
    return history, None, audio_path, gr.update(visible=True), gr.update(visible=True), text

def clear_audio(_):
    return None, gr.update(visible=True), gr.update(visible=True)

def export_chat(history):
    if not history:
        return None
    lines = []
    lines.append(f"旅遊小管家對話匯出 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 60)
    for turn in history:
        user, bot = turn
        lines.append("使用者：\n" + (user or ""))
        lines.append("小管家：\n" + (bot or ""))
        lines.append("-" * 40)
    content = "\n".join(lines)
    path = tempfile.NamedTemporaryFile(delete=False, suffix=".txt").name
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path

def clear_history():
    return None

def noop_return_none(*args, **kwargs):
    return None

# -------------------------
# Gradio Blocks UI
# -------------------------
JS_APPLY_AUDIO_PREFS_AND_AUTOPLAY = """
(audio_path) => {
    try {
        const audios = document.querySelectorAll('audio');
        const audio = audios[audios.length - 1];
        if (!audio) return null;
        const rate = parseFloat(localStorage.getItem('voice_playback_rate') || '1.0');
        const vol  = parseFloat(localStorage.getItem('voice_volume') || '1.0');
        const auto = (localStorage.getItem('voice_autoplay') || 'true') === 'true';
        audio.playbackRate = Math.max(0.5, Math.min(2.0, rate));
        audio.volume = Math.max(0.0, Math.min(1.0, vol));
        if (auto) {
            const p = audio.play();
            if (p && p.catch) { p.catch(() => {}); }
        }
    } catch (e) {}
    return null;
}
"""

with gr.Blocks(title="旅遊小管家 Pro（Render 版）") as demo:
    gr.Markdown("## 🧳 旅遊小管家 Pro\\n支援語音輸入、智慧回覆、TTS 語音切換與偏好記憶。")

    chatbot = gr.Chatbot(label="對話區")
    user_text = gr.Textbox(placeholder="輸入文字或使用語音...", label="文字訊息")
    send_btn = gr.Button("送出文字", variant="primary")

    voice_dropdown = gr.Dropdown(choices=VOICE_CHOICES, value="alloy", label="語音包")
    tts_model_dd = gr.Dropdown(choices=TTS_MODEL_CHOICES, value="gpt-4o-mini-tts", label="TTS 模型")
    whisper_lang_dd = gr.Dropdown(choices=WHISPER_LANG_CHOICES, value="auto", label="Whisper 語言")
    system_prompt_tb = gr.Textbox(value=DEFAULT_SYSTEM_PROMPT, label="System Prompt")

    tts_output = gr.Audio(label="🔊 回覆語音", type="filepath", interactive=False)
    transcript_tb = gr.Textbox(label="🎧 語音辨識文字", interactive=False)
    mic_audio = gr.Audio(sources=["microphone", "upload"], type="filepath", label="錄音後可預聽")

    send_audio_btn = gr.Button("✅ 送出語音")
    delete_audio_btn = gr.Button("🗑️ 刪除音訊")
    rerecord_btn = gr.Button("🔁 重新錄製")

    export_btn = gr.Button("📝 匯出對話")
    clear_btn = gr.Button("🧹 清空對話")

    dummy_store = gr.Textbox(visible=False)

    send_btn.click(on_send_text, [user_text, chatbot, voice_dropdown, tts_model_dd, system_prompt_tb], [chatbot, user_text, tts_output, transcript_tb])
    send_audio_btn.click(on_send_audio, [mic_audio, chatbot, voice_dropdown, tts_model_dd, whisper_lang_dd, system_prompt_tb], [chatbot, mic_audio, tts_output, delete_audio_btn, rerecord_btn, transcript_tb])
    delete_audio_btn.click(clear_audio, [mic_audio], [mic_audio, delete_audio_btn, rerecord_btn])
    rerecord_btn.click(clear_audio, [mic_audio], [mic_audio, delete_audio_btn, rerecord_btn])

    export_file = gr.File(label="下載檔案", visible=True)
    export_btn.click(export_chat, [chatbot], [export_file])
    clear_btn.click(clear_history, None, [chatbot])

    tts_output.change(fn=noop_return_none, inputs=[tts_output], outputs=[dummy_store], _js=JS_APPLY_AUDIO_PREFS_AND_AUTOPLAY)

    gr.Markdown("ℹ️ 若以 Render 部署，請使用 uvicorn finalapp:app")

# -------------------------
# FastAPI app (for Render)
# -------------------------
fastapi_app = FastAPI()
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=port)
