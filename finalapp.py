# -*- coding: utf-8 -*-
# ==========================================
# 🎤 旅遊語音小管家 2.0
# - 語音輸入 / 輸出
# - 年齡層模式（含長者大字）
# - 說話風格
# - 行程規劃表格化（交給 GPT 依提示生成）
# - Google Maps 外部連結
# ==========================================

import os, re, base64, tempfile, urllib.parse
from datetime import datetime
import gradio as gr
from fastapi import FastAPI
from openai import OpenAI, APIConnectionError, APIStatusError, BadRequestError, AuthenticationError

# -------------------------
# 🔗 HTML 連結工具（避免 Render 吃掉 URL）
# -------------------------
URL_RE = re.compile(r'(https?://[^\s)]+)')

def html_linkify(text: str) -> str:
    """將純網址轉為 HTML 超連結，target=_blank。"""
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
    """若使用者問路 / 問地圖，且尚未有連結，則補一條 Google Maps 連結。"""
    if not needs_map_link(user_utterance):
        return bot_text
    if URL_RE.search(bot_text or ""):
        return bot_text
    url = gmaps_search_link(user_utterance)
    return (
        (bot_text or "")
        + '\n\n🗺️ <b>快速開啟地圖：</b> '
        + f'<a href="{url}" target="_blank" rel="noopener noreferrer">'
        + f'{user_utterance} · Google 地圖</a>'
    )

# -------------------------
# 🔑 初始化 OpenAI
# -------------------------
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ 請先設定環境變數 OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

VOICE_CHOICES = [
    "alloy","ash","ballad","coral","echo",
    "fable","nova","onyx","sage","shimmer","verse"
]
FIXED_TTS_MODEL = "gpt-4o-mini-tts"

LANG_CHOICES = [
    "auto",
    "中文(zh)","英文(en)","泰文(th)","日文(ja)","韓文(ko)",
    "法文(fr)","德文(de)","西班牙文(es)","越南文(vi)"
]
LANG_MAP = {
    "auto":"auto",
    "中文(zh)":"zh",
    "英文(en)":"en",
    "泰文(th)":"th",
    "日文(ja)":"ja",
    "韓文(ko)":"ko",
    "法文(fr)":"fr",
    "德文(de)":"de",
    "西班牙文(es)":"es",
    "越南文(vi)":"vi",
}

BASE_SYSTEM_PROMPT = """你是「旅遊語音小管家」，需要同時扮演旅遊顧問與貼心小助手。

【回覆風格】
1. 使用繁體中文。
2. 依照使用者族群與說話風格調整語氣，例如對長者要放慢、清楚、溫和。
3. 優先以 2～3 句簡短說明重點，不要一次講太滿。

【互動方式】
1. 如果使用者沒有說明「地點 / 天數 / 同行對象」等關鍵資訊，請先用 1～2 個問題幫忙釐清需求，再幫忙規劃行程。
2. 問問題時請簡單友善，例如：「想去哪一個縣市呢？」、「大概玩幾天比較合適？」。

【行程規劃呈現方式】
1. 先用一小段話總結整體行程特色（例如：適合長輩慢遊、適合拍照、以美食為主等）。
2. 接著使用「表格」形式，清楚列出每日行程，欄位包含：天數 / 時間 / 地點 / 活動內容 / 交通方式。
3. 最後列出 2～3 點小提醒（例如：攜帶物品、天氣、交通注意事項）。

【Google Maps 使用】
1. 若提到具體地點、景點或使用者詢問怎麼走、路線、導航，請主動提供 Google Maps 外部連結（完整 https URL）。
2. 若只有地名或關鍵字，可使用：
   https://www.google.com/maps/search/?api=1&query=<關鍵字>
"""

def build_system_prompt(base_prompt: str, user_age: str, tone_style: str) -> str:
    age_hint = f"目前服務的對象為：{user_age}。"
    tone_hint = f"請以「{tone_style}」的說話風格來回應。"
    return (base_prompt or BASE_SYSTEM_PROMPT).strip() + "\n\n" + age_hint + "\n" + tone_hint

# -------------------------
# 🛡️ OpenAI 安全封裝
# -------------------------
def _fmt_err(e: Exception) -> str:
    if isinstance(e, AuthenticationError):
        return "⚠️ OpenAI 金鑰驗證失敗，請確認伺服器環境變數設定。"
    if isinstance(e, BadRequestError):
        return f"⚠️ 參數錯誤：{getattr(e, 'message', str(e))}"
    if isinstance(e, APIStatusError):
        return f"⚠️ 服務狀態錯誤：{e.status_code} {getattr(e, 'message', str(e))}"
    if isinstance(e, APIConnectionError):
        return "⚠️ 無法連線至 OpenAI，請稍後再試或檢查網路。"
    return f"⚠️ 未預期錯誤：{str(e)}"

def safe_chat(user_text: str, system_prompt: str):
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role":"system","content":system_prompt.strip()},
                {"role":"user","content":user_text},
            ],
            temperature=0.5,
        )
        ans = resp.choices[0].message.content.strip()
        return ans, None
    except Exception as e:
        return None, _fmt_err(e)

def safe_transcribe(path: str, lang_label: str):
    try:
        lang = LANG_MAP.get(lang_label, "auto")
        kwargs = {"model":"whisper-1"}
        if lang != "auto":
            kwargs["language"] = lang
        with open(path,"rb") as f:
            r = client.audio.transcriptions.create(file=f, **kwargs)
        return (r.text or "").strip(), None
    except Exception as e:
        return None, _fmt_err(e)

def safe_tts(text: str, voice: str):
    try:
        if not text.strip():
            return None, None
        out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
        with client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice=voice,
            input=text,
        ) as r:
            r.stream_to_file(out)
        return out, None
    except Exception as e:
        return None, _fmt_err(e)

# -------------------------
# 🎙️ 錄音資料處理 DataURL → 檔案
# -------------------------
def write_dataurl_to_file(data_url: str) -> str:
    if not data_url or "," not in data_url:
        raise ValueError("無效的音訊資料。")
    header, b64 = data_url.split(",",1)
    ext = ".webm"
    if "audio/ogg" in header:
        ext = ".ogg"
    elif "audio/mpeg" in header or "audio/mp3" in header:
        ext = ".mp3"
    elif "audio/wav" in header:
        ext = ".wav"
    raw = base64.b64decode(b64)
    path = tempfile.NamedTemporaryFile(delete=False, suffix=ext).name
    with open(path,"wb") as f:
        f.write(raw)
    return path

# -------------------------
# 🚀 事件：文字輸入
# -------------------------
def on_send_text(msg, history, voice, system_prompt, user_age, tone_style):
    if not (msg or "").strip():
        return history, history, "", None, "請先輸入問題或使用語音～", "", ""

    sys_prompt = build_system_prompt(system_prompt, user_age, tone_style)
    bot, err = safe_chat(msg, sys_prompt)
    audio_path = None

    if err:
        bot = err
    else:
        bot = maybe_append_map_link(msg, bot)
        bot = html_linkify(bot)
        audio_path, aerr = safe_tts(bot, voice)
        if aerr:
            bot += f"\n\n（語音播放產生失敗：{aerr}）"

    history = (history or []) + [{"role": "user", "content": msg}, {"role": "assistant", "content": bot}]
    # 回傳：chatbot, state, 清空 user, tts_path, 最新回答, STT結果清空, 錯誤訊息清空
    return history, history, "", audio_path, bot, "", ""

# -------------------------
# 🎙️ 事件：停止錄音並送出
# -------------------------
def on_audio_dataurl_received(audio_b64, history, voice, system_prompt, lang_label, user_age, tone_style):
    if not audio_b64:
        return history, history, None, "沒有錄到音", "", "請重新錄音看看～"

    try:
        path = write_dataurl_to_file(audio_b64)
    except Exception as e:
        return history, history, None, "錄音轉檔失敗", "", _fmt_err(e)

    text, terr = safe_transcribe(path, lang_label)
    if terr:
        return history, history, None, "語音辨識失敗", "", terr

    sys_prompt = build_system_prompt(system_prompt, user_age, tone_style)
    bot, cerr = safe_chat(text, sys_prompt)
    audio_path = None

    if cerr:
        bot = cerr
    else:
        bot = maybe_append_map_link(text, bot)
        bot = html_linkify(bot)
        audio_path, aerr = safe_tts(bot, voice)
        if aerr:
            bot += f"\n\n（語音播放產生失敗：{aerr}）"

    # 轉換為 Gradio 6.x 格式
    new_message = [{"role": "user", "content": f"(語音)\n{text}"}, {"role": "assistant", "content": bot}]
    history = (history or []) + new_message
    # 回傳：chatbot, state, tts_path, 錄音狀態, STT結果顯示, 錯誤訊息清空
    return history, history, audio_path, "已停止並送出", text, ""

# -------------------------
# 📝 對話匯出 & 清除
# -------------------------
def export_chat(history):
    lines = ["旅遊語音小管家 對話匯出", "="*60]
    for msg in (history or []):
        if isinstance(msg, dict):
            role = "使用者" if msg.get("role") == "user" else "小管家"
            content = msg.get("content", "")
            lines += [f"{role}：", content, "-"*40]
        else:
            u, b = msg
            lines += ["使用者：", u or "", "小管家：", b or "", "-"*40]
    path = tempfile.NamedTemporaryFile(delete=False, suffix=".txt").name
    with open(path,"w",encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path

def clear_history():
    return [], [], "", "", ""

# -------------------------
# 🎨 CSS（科技風 + 長者大字模式）
# -------------------------
CSS_TECH = """
:root{
  --bg:#0a1120; --panel:#0f1b33cc; --stroke:#1e2b4d;
  --text:#e8eefc; --muted:#9bb0d6; --accent:#54b7ff; --accent-2:#00ffd0;
}
.gradio-container{
  font-family:ui-sans-serif,system-ui,PingFangTC,'Noto Sans TC',Segoe UI,Roboto,Helvetica,Arial;
  color:var(--text);
}
body{
  background:radial-gradient(1200px 600px at 20% -10%, #11315d55, transparent),
             linear-gradient(180deg,#0a1120 0%, #0a1120 100%);
}
.neon-panel{
  background:var(--panel);
  border:1px solid var(--stroke);
  box-shadow:0 0 0 1px #0e1a33 inset,0 10px 30px #0008;
  border-radius:16px;
  padding:16px;
}
button.primary{
  background:linear-gradient(90deg,var(--accent),var(--accent-2));
  color:#00121d;
  font-weight:700;
  border-radius:12px!important;
}
.badge{
  background:#112a49;
  color:#8bd9ff;
  padding:2px 8px;
  border:1px solid #1f3c66;
  border-radius:999px;
  font-size:12px;
  margin-left:8px;
}
a{
  color:#7fd0ff;
  text-decoration:underline;
}
"""

CSS_ELDER = """
.app-root.elder-mode *{
  font-size:1.25em !important;
}
"""

# -------------------------
# 🎙️ JS：開始錄音 / 停止並送出
# -------------------------
JS_START_RECORD = """
async () => {
  try {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      return ['裝置不支援錄音'];
    }
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mr = new MediaRecorder(stream);
    window.__mr_chunks = [];
    window.__mr_stream = stream;
    window.__mr = mr;
    mr.ondataavailable = (e)=>{ if (e.data && e.data.size) window.__mr_chunks.push(e.data); };
    mr.start();
    return ['錄音中…'];
  } catch (e) {
    return ['權限被拒絕或裝置不可用'];
  }
}
"""

JS_STOP_AND_EXPORT = """
async (a,b,c,d,e,f,g)=>{
  try{
    const m = window.__mr;
    const s = window.__mr_stream;
    if(!m){ return [null,b,c,d,e,f,g]; }
    return await new Promise(res=>{
      m.onstop = async ()=>{
        const blob = new Blob(window.__mr_chunks||[],{type:'audio/webm;codecs=opus'});
        if(s){ s.getTracks().forEach(t=>t.stop()); }
        window.__mr=null; window.__mr_stream=null; window.__mr_chunks=null;
        const reader=new FileReader();
        reader.onloadend=()=>res([reader.result,b,c,d,e,f,g]);
        reader.readAsDataURL(blob);
      };
      m.stop();
    });
  }catch(err){
    return [null,b,c,d,e,f,g];
  }
}
"""

# -------------------------
# 🙋‍♂️ 長者模式：切換 CSS class
# -------------------------
def apply_age_mode(age):
    base_classes = ["app-root"]
    if age and "長者" in age:
        base_classes.append("elder-mode")
    return gr.update(elem_classes=base_classes)

# -------------------------
# ✨ 範例問題填入
# -------------------------
def fill_example():
    return "幫我規劃台中一日遊，要包含拍照景點、咖啡廳和交通方式"

# -------------------------
# 🧱 Gradio 介面
# -------------------------
with gr.Blocks(
    title="旅遊語音小管家",
    css=CSS_TECH + CSS_ELDER,
) as demo:

    # 首次載入顯示一次提醒（使用 localStorage 記錄）
    gr.HTML(
        """
        <script>
        window.addEventListener('load', () => {
          try{
            if (!localStorage.getItem('travel_helper_intro_shown')) {
              alert(
                '歡迎使用旅遊語音小管家！\\n\\n' +
                '你可以試著這樣問：\\n' +
                '・幫我規劃台中一日遊\\n' +
                '・我要帶阿公阿嬤去日月潭兩天一夜，幫我安排行程'
              );
              localStorage.setItem('travel_helper_intro_shown','1');
            }
          }catch(e){}
        });
        </script>
        """
    )

    with gr.Column(elem_id="app-root", elem_classes=["app-root"]) as app_root:

        gr.Markdown(
            "## 🎤 旅遊語音小管家 "
            "<span class='badge'>AI 語音旅遊助手</span>"
        )

        gr.Markdown(
            """
> 💡 **小提示：你可以這樣問：**  
> ・「幫我規劃台中一日遊，想走文青咖啡廳路線」  
> ・「我要帶長輩去日月潭兩天一夜，請幫我安排輕鬆的行程」  
> ・「請推薦雲林適合親子去的景點，順便附上 Google 地圖」  
            """
        )

        with gr.Row():

            # 左側：對話與輸出
            with gr.Column(scale=3, elem_classes=["neon-panel"]):
                chatbot = gr.Chatbot(
                    label="對話區",
                    height=520,
                )
                state = gr.State([])

                user = gr.Textbox(
                    label="輸入文字",
                    placeholder="在這裡輸入旅遊問題，或使用右邊語音錄製功能"
                )

                with gr.Row():
                    send = gr.Button("🚀 送出文字", elem_classes=["primary"])
                    clear = gr.Button("🧹 清空對話")
                    example_btn = gr.Button("✨ 插入範例問題")

                tts = gr.Audio(label="🔊 回覆語音", type="filepath")
                latest = gr.Textbox(label="最新回答", interactive=False)
                stt_result = gr.Textbox(label="🗣 你剛剛說了：", lines=3, interactive=False)
                errbox = gr.Textbox(label="訊息 / 錯誤", interactive=False)

                with gr.Row():
                    export = gr.Button("📝 匯出對話")
                    exp_file = gr.File(label="下載檔案", visible=True)

            # 右側：語音與設定
            with gr.Column(scale=2, elem_classes=["neon-panel"], visible=False):
                gr.Markdown("### 🎙️ 語音輸入（按鈕控制）")

                with gr.Row():
                    start = gr.Button("🎙️ 開始錄音", elem_classes=["primary"])
                    stop = gr.Button("⏹️ 停止並送出")

                mic = gr.Textbox(value="尚未錄音", label="錄音狀態", interactive=False)
                audiobox = gr.Textbox(visible=False)

                gr.Markdown("---")
                gr.Markdown("### ⚙️ 使用者設定")

                user_age = gr.Dropdown(
                    label="👥 使用者族群",
                    choices=[
                        "兒童（國小）",
                        "青少年／大學生",
                        "上班族",
                        "長者（大字模式）",
                    ],
                    value="青少年／大學生",
                )

                tone_style = gr.Dropdown(
                    label="🗣 說話風格",
                    choices=["溫柔耐心", "活潑有精神", "專業冷靜", "旅遊網紅風"],
                    value="溫柔耐心",
                )

                voice = gr.Dropdown(
                    choices=VOICE_CHOICES,
                    value="alloy",
                    label="語音包（GPT TTS）"
                )

                lang = gr.Dropdown(
                    choices=LANG_CHOICES,
                    value="auto",
                    label="Whisper 語言"
                )

                with gr.Accordion("🔒 系統提示詞（開發者設定）", open=False):
                    sys_tb = gr.Textbox(
                        value=BASE_SYSTEM_PROMPT,
                        label="System Prompt",
                        lines=6
                    )

    # -------------------------
    # 綁定事件
    # -------------------------
    send.click(
        on_send_text,
        [user, state, voice, sys_tb, user_age, tone_style],
        [chatbot, state, user, tts, latest, stt_result, errbox],
    )

    clear.click(clear_history, None, [chatbot, state, latest, stt_result, errbox])

    example_btn.click(fill_example, None, user)

    start.click(
        fn=lambda: "錄音功能暫時不可用，請使用文字輸入",
        inputs=None,
        outputs=[mic],
    )

    stop.click(
        fn=lambda a,b,c,d,e,f,g: (b,c,None,"錄音功能暫時不可用", "", "請使用文字輸入"),
        inputs=[audiobox, state, voice, sys_tb, lang, user_age, tone_style],
        outputs=[chatbot, state, tts, mic, stt_result, errbox],
    )

    export.click(export_chat, [state], [exp_file])

    # 切換長者模式（大字體）
    user_age.change(apply_age_mode, [user_age], [app_root])

# -------------------------
# 🚪 FastAPI mount（給 Render / uvicorn 用）
# -------------------------
# -------------------------
# 🚪 啟動設定（針對 Render 優化）
# -------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    
    demo.launch(
        server_name="0.0.0.0", 
        server_port=port,
        share=False,
        show_api=False,   # <--- 新增這一行，關閉 API 文檔生成，徹底避開報錯位置
        allowed_paths=["/"]
    )