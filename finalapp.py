# -*- coding: utf-8 -*-
# ==========================================
# 🧳 旅遊小管家 Pro（語音切換 + 記憶偏好 + 播放控制 + 匯出）
# 相容：Gradio 3.41.0、openai>=1.0.0
# 需要環境變數：OPENAI_API_KEY
# ==========================================
import os
import tempfile
from datetime import datetime

import gradio as gr
from openai import OpenAI

# -------------------------
# 0) 初始化 OpenAI
# -------------------------
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ 請先設定環境變數 OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# -------------------------
# 1) 常數與選單
# -------------------------
VOICE_CHOICES = [
    "alloy", "ash", "ballad", "coral", "echo",
    "fable", "nova", "onyx", "sage", "shimmer", "verse",
]
TTS_MODEL_CHOICES = ["gpt-4o-mini-tts", "tts-1", "tts-1-hd"]

WHISPER_LANG_CHOICES = [
    "auto", "zh", "en", "th", "ja", "ko", "fr", "de", "es", "vi"
]

DEFAULT_SYSTEM_PROMPT = "你是旅遊小管家，回答精簡、實用，使用繁體中文。"

# -------------------------
# 2) 產生文字回覆（聊天邏輯）
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
# 3) Whisper 語音轉文字（可指定語言）
# -------------------------
def transcribe_audio_to_text(audio_file_path: str, lang_code: str) -> str:
    kwargs = {"model": "whisper-1"}
    if lang_code and lang_code != "auto":
        kwargs["language"] = lang_code
    with open(audio_file_path, "rb") as f:
        result = client.audio.transcriptions.create(file=f, **kwargs)
    return (getattr(result, "text", "") or "").strip()

# -------------------------
# 4) 聲音合成（TTS）
# -------------------------
def text_to_speech(text: str, voice_name: str, tts_model: str):
    if not text.strip():
        return None
    tts_model = tts_model or "gpt-4o-mini-tts"
    speech_file_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
    # 用串流避免大回覆占記憶體
    with client.audio.speech.with_streaming_response.create(
        model=tts_model,
        voice=voice_name,
        input=text
    ) as response:
        response.stream_to_file(speech_file_path)
    return speech_file_path

# -------------------------
# 5) 事件：純文字送出
# -------------------------
def on_send_text(msg, history, voice_name, tts_model, system_prompt):
    bot_text = chat_answer(msg, system_prompt)
    history = (history or []) + [(msg, bot_text)]
    audio_path = text_to_speech(bot_text, voice_name, tts_model)
    return history, "", audio_path, bot_text

# -------------------------
# 6) 事件：語音送出
# -------------------------
def on_send_audio(audio_file, history, voice_name, tts_model, whisper_lang, system_prompt):
    if not audio_file:
        return history, None, None, gr.update(visible=True), gr.update(visible=True), ""
    text = transcribe_audio_to_text(audio_file, whisper_lang)
    bot_text = chat_answer(text, system_prompt)
    history = (history or []) + [(f"(語音轉文字)\\n{text}", bot_text)]
    audio_path = text_to_speech(bot_text, voice_name, tts_model)
    return history, None, audio_path, gr.update(visible=True), gr.update(visible=True), text

# -------------------------
# 7) 刪除 / 重錄（僅清空音訊）
# -------------------------
def clear_audio(_):
    return None, gr.update(visible=True), gr.update(visible=True)

# -------------------------
# 8) 匯出對話（.txt）
# -------------------------
def export_chat(history):
    if not history:
        return None
    lines = []
    lines.append(f"旅遊小管家對話匯出 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 60)
    for turn in history:
        user, bot = turn
        lines.append("使用者：\\n" + (user or ""))
        lines.append("小管家：\\n" + (bot or ""))
        lines.append("-" * 40)
    content = "\\n".join(lines)
    path = tempfile.NamedTemporaryFile(delete=False, suffix=".txt").name
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path

# -------------------------
# 9) 清空對話
# -------------------------
def clear_history():
    return None

# -------------------------
# 10) JS：localStorage 與音訊播放屬性
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

def noop_return_none(*args, **kwargs):
    return None

# -------------------------
# 11) 介面
# -------------------------
with gr.Blocks(title="旅遊小管家 Pro（語音互動＋記憶偏好）") as demo:
    gr.Markdown("## 🧳 旅遊小管家 Pro\\n支援語音輸入、回覆語音、**語音切換**與**偏好記憶**，並能調整**播放速度**與**音量**、匯出對話。")

    # 對話區
    chatbot = gr.Chatbot(label="對話區")
    with gr.Row():
        user_text = gr.Textbox(placeholder="輸入文字或使用語音...", label="文字訊息", scale=4)
        send_btn = gr.Button("送出文字", variant="primary", scale=1)

    # 語音相關設定
    with gr.Row():
        voice_dropdown = gr.Dropdown(choices=VOICE_CHOICES, value="alloy", label="語音包（TTS 聲音）", scale=2)
        tts_model_dd = gr.Dropdown(choices=TTS_MODEL_CHOICES, value="gpt-4o-mini-tts", label="TTS 模型", scale=1)
        whisper_lang_dd = gr.Dropdown(choices=WHISPER_LANG_CHOICES, value="auto", label="Whisper 語言", scale=1)

    with gr.Row():
        autoplay_chk = gr.Checkbox(value=True, label="自動播放回覆語音")
        playback_rate_slider = gr.Slider(minimum=0.5, maximum=2.0, value=1.0, step=0.1, label="播放速度")
        volume_slider = gr.Slider(minimum=0.0, maximum=1.0, value=1.0, step=0.05, label="音量")

    # System prompt（可記憶）
    system_prompt_tb = gr.Textbox(
        value=DEFAULT_SYSTEM_PROMPT,
        label="系統提示詞（System Prompt）",
        lines=3,
        placeholder="可自訂小管家的語氣與行為（此設定會被記住）"
    )

    # 回覆語音與辨識文字顯示
    tts_output = gr.Audio(label="🔊 回覆語音", type="filepath", interactive=False)
    transcript_tb = gr.Textbox(label="🎧 語音辨識文字（使用者錄音）", interactive=False)

    # 語音輸入區（可預聽/重錄/刪除）
    gr.Markdown("### 🎤 語音輸入（錄完可預聽，滿意再送出；或刪除／重錄）")
    mic_audio = gr.Audio(sources=["microphone", "upload"], type="filepath", label="錄音後可預聽、重錄或刪除")
    with gr.Row():
        send_audio_btn = gr.Button("✅ 送出語音（轉文字並回覆）", variant="primary")
        delete_audio_btn = gr.Button("🗑️ 刪除音訊")
        rerecord_btn = gr.Button("🔁 重新錄製")

    # 工具列
    with gr.Row():
        export_btn = gr.Button("📝 匯出對話（.txt）")
        clear_btn = gr.Button("🧹 清空對話")

    # 隱藏 dummy
    dummy_store = gr.Textbox(visible=False)

    # ---- 事件：聊天 ----
    send_btn.click(
        on_send_text,
        inputs=[user_text, chatbot, voice_dropdown, tts_model_dd, system_prompt_tb],
        outputs=[chatbot, user_text, tts_output, transcript_tb],
    )
    send_audio_btn.click(
        on_send_audio,
        inputs=[mic_audio, chatbot, voice_dropdown, tts_model_dd, whisper_lang_dd, system_prompt_tb],
        outputs=[chatbot, mic_audio, tts_output, delete_audio_btn, rerecord_btn, transcript_tb],
    )
    delete_audio_btn.click(clear_audio, inputs=[mic_audio], outputs=[mic_audio, delete_audio_btn, rerecord_btn])
    rerecord_btn.click(clear_audio, inputs=[mic_audio], outputs=[mic_audio, delete_audio_btn, rerecord_btn])

    # ---- 事件：匯出與清除 ----
    export_file = gr.File(label="下載檔案", visible=True)
    export_btn.click(export_chat, inputs=[chatbot], outputs=[export_file])
    clear_btn.click(clear_history, inputs=None, outputs=[chatbot])

    # ---- 事件：記住使用者偏好（localStorage）----
    # 語音包
    voice_dropdown.change(
        fn=noop_return_none, inputs=[voice_dropdown], outputs=[dummy_store],
        _js="(v)=>{ try{localStorage.setItem('voice_choice', v);}catch(e){}; return null; }"
    )
    demo.load(
        fn=lambda: "alloy", inputs=None, outputs=[voice_dropdown],
        _js="()=>{ try{const v=localStorage.getItem('voice_choice'); return v||'alloy';}catch(e){return 'alloy';} }"
    )
    # TTS 模型
    tts_model_dd.change(
        fn=noop_return_none, inputs=[tts_model_dd], outputs=[dummy_store],
        _js="(v)=>{ try{localStorage.setItem('tts_model', v);}catch(e){}; return null; }"
    )
    demo.load(
        fn=lambda: "gpt-4o-mini-tts", inputs=None, outputs=[tts_model_dd],
        _js="()=>{ try{const v=localStorage.getItem('tts_model'); return v||'gpt-4o-mini-tts';}catch(e){return 'gpt-4o-mini-tts';} }"
    )
    # Whisper 語言
    whisper_lang_dd.change(
        fn=noop_return_none, inputs=[whisper_lang_dd], outputs=[dummy_store],
        _js="(v)=>{ try{localStorage.setItem('whisper_lang', v);}catch(e){}; return null; }"
    )
    demo.load(
        fn=lambda: "auto", inputs=None, outputs=[whisper_lang_dd],
        _js="()=>{ try{const v=localStorage.getItem('whisper_lang'); return v||'auto';}catch(e){return 'auto';} }"
    )
    # 自動播放
    autoplay_chk.change(
        fn=noop_return_none, inputs=[autoplay_chk], outputs=[dummy_store],
        _js="(v)=>{ try{localStorage.setItem('voice_autoplay', v ? 'true':'false');}catch(e){}; return null; }"
    )
    demo.load(
        fn=lambda: True, inputs=None, outputs=[autoplay_chk],
        _js="()=>{ try{const v=localStorage.getItem('voice_autoplay'); return v===null? true : (v==='true');}catch(e){return true;} }"
    )
    # 播放速度
    playback_rate_slider.change(
        fn=noop_return_none, inputs=[playback_rate_slider], outputs=[dummy_store],
        _js="(v)=>{ try{localStorage.setItem('voice_playback_rate', String(v));}catch(e){}; return null; }"
    )
    demo.load(
        fn=lambda: 1.0, inputs=None, outputs=[playback_rate_slider],
        _js="()=>{ try{const v=parseFloat(localStorage.getItem('voice_playback_rate')); return isNaN(v)?1.0:v;}catch(e){return 1.0;} }"
    )
    # 音量
    volume_slider.change(
        fn=noop_return_none, inputs=[volume_slider], outputs=[dummy_store],
        _js="(v)=>{ try{localStorage.setItem('voice_volume', String(v));}catch(e){}; return null; }"
    )
    demo.load(
        fn=lambda: 1.0, inputs=None, outputs=[volume_slider],
        _js="()=>{ try{const v=parseFloat(localStorage.getItem('voice_volume')); return isNaN(v)?1.0:v;}catch(e){return 1.0;} }"
    )
    # System Prompt：兩段式，避免 f-string 與 JS 大括號衝突
    demo.load(
        fn=lambda: DEFAULT_SYSTEM_PROMPT,
        inputs=None,
        outputs=[system_prompt_tb],
    )
    demo.load(
        fn=lambda: None,
        inputs=None,
        outputs=[system_prompt_tb],
        _js="""
            () => {
                try {
                    const v = localStorage.getItem('system_prompt');
                    return (v !== null) ? v : undefined;
                } catch (e) {
                    return undefined;
                }
            }
        """,
    )

    # 當音檔變更時套用偏好並自動播放
    tts_output.change(
        fn=noop_return_none,
        inputs=[tts_output],
        outputs=[dummy_store],
        _js=JS_APPLY_AUDIO_PREFS_AND_AUTOPLAY
    )

    gr.Markdown("— 提醒：麥克風權限需在 HTTPS 網址；若嵌入 iframe，請加入 `allow=\\\"microphone\\\"`。偏好設定儲存在瀏覽器 localStorage。")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=port)
