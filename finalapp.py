# ================================
# 🧳 旅遊小管家（Render-ready：FastAPI + Uvicorn + Gradio）
# ================================
import os
import tempfile
from pathlib import Path

import gradio as gr
from fastapi import FastAPI
from openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# ---------- 0) 讀取 API Key（優先環境變數，其次 .env） ----------
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    try:
        # 本地開發用：自動讀取 .env（Render 不需要 .env）
        from dotenv import load_dotenv
        load_dotenv()
        API_KEY = os.getenv("OPENAI_API_KEY")
    except Exception:
        pass

if not API_KEY:
    # 這段訊息會出現在啟動日志，協助你快速排錯
    raise ValueError(
        "❌ 找不到 OPENAI_API_KEY。\n"
        "請在 Render 的 Dashboard → Environment 里新增環境變數 OPENAI_API_KEY，\n"
        "或在本地專案建立 .env 並寫入：\nOPENAI_API_KEY=sk-xxxxx"
    )

# OpenAI & Embedding
client = OpenAI(api_key=API_KEY)
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=API_KEY,
)

# ---------- 1) FAISS 路徑（允許用環境變數覆寫） ----------
BASE_DIR = Path(__file__).parent.resolve()
FAISS_DIR = Path(os.getenv("FAISS_DIR", BASE_DIR / "faiss1022_db"))

# 延遲載入，避免冷啟動太久/找不到路徑立即崩潰
_db = None
def get_db():
    """延遲載入 FAISS 向量庫（第一次查詢時再讀檔）"""
    global _db
    if _db is None:
        if not FAISS_DIR.exists():
            raise FileNotFoundError(f"❌ 找不到 FAISS 資料夾：{FAISS_DIR}")
        _db = FAISS.load_local(
            str(FAISS_DIR),
            embedding_model,
            allow_dangerous_deserialization=True
        )
        print(f"✅ 已載入 FAISS：{FAISS_DIR}")
    return _db

# ---------- 2) 模型設定 ----------
SYSTEM_PROMPT = (
    "你是一個友善的『旅遊小管家』，幫助使用者解答旅遊相關問題。"
    "如果資料不足或問題與旅遊無關，請回答：『我沒有這方面的資料哦～』"
)
CHAT_MODEL   = "gpt-4o-mini"
WHISPER_MODEL = "whisper-1"
TTS_MODEL    = "gpt-4o-mini-tts"
TTS_VOICE    = "alloy"

# ---------- 3) 業務邏輯 ----------
def query_faiss(user_message: str, k: int = 1):
    try:
        db = get_db()
        docs = db.similarity_search(user_message, k=k)
        return docs[0].page_content if docs else None
    except Exception as e:
        print(f"⚠️ FAISS 查詢失敗: {e}")
        return None

def chat_with_openai(user_message, history):
    if not user_message:
        return "", history, None

    kb = query_faiss(user_message)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for q, a in history:
        messages += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    messages.append({"role": "user", "content": user_message})
    if kb:
        messages.insert(0, {"role": "system", "content": f"資料庫資訊：\n{kb}"})

    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL, messages=messages, temperature=0.6
        )
        reply = resp.choices[0].message.content
    except Exception as e:
        reply = f"⚠️ 無法取得回覆：{e}"

    history.append((user_message, reply))

    # TTS（失敗不阻擋）
    audio_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            with client.audio.speech.with_streaming_response.create(
                model=TTS_MODEL, voice=TTS_VOICE, input=reply
            ) as r:
                r.stream_to_file(tmp.name)
            audio_path = tmp.name
    except Exception as e:
        print(f"🔇 TTS 錯誤：{e}")

    return "", history, audio_path

def audio_to_text(audio_file, history):
    if not audio_file:
        return "", history, None
    try:
        with open(audio_file, "rb") as f:
            result = client.audio.transcriptions.create(
                model=WHISPER_MODEL, file=f, language="zh"
            )
        text = result.text.strip()
    except Exception as e:
        text = f"語音辨識失敗：{e}"
    return chat_with_openai(text, history)

# ---------- 4) Gradio UI ----------
custom_css = """
body { background: linear-gradient(135deg,#f5f7fa,#c3cfe2); font-family:'Microsoft JhengHei',sans-serif; }
.gradio-container { max-width:900px !important; margin:auto; border-radius:20px; box-shadow:0 0 25px rgba(0,0,0,0.1); background:white; }
h1,h2,h3 { text-align:center; color:#333; }
button { border-radius:10px !important; font-weight:bold; transition:.2s; }
button:hover { background-color:#4a90e2 !important; color:white !important; }
"""

with gr.Blocks(title="🧳 旅遊小管家", css=custom_css, theme=gr.themes.Glass()) as demo:
    gr.Markdown("# 🌏 旅遊小管家\n輸入文字或語音提問，小管家會結合知識庫與 GPT 提供建議 ✈️\n---")
    chatbot = gr.Chatbot(label="💬 對話區", height=420)

    with gr.Row():
        msg = gr.Textbox(label="輸入問題", placeholder="例如：幫我安排台南兩天一夜行程", scale=3)
        send_btn = gr.Button("🚀 送出", variant="primary", scale=1)

    with gr.Column():
        audio_in  = gr.Audio(label="🎙️ 語音輸入（錄音或上傳）", type="filepath")
        audio_out = gr.Audio(label="🔊 語音回覆", type="filepath")

    send_btn.click(chat_with_openai, [msg, chatbot], [msg, chatbot, audio_out])
    msg.submit(chat_with_openai, [msg, chatbot], [msg, chatbot, audio_out])
    audio_in.change(audio_to_text, [audio_in, chatbot], [msg, chatbot, audio_out])

# ---------- 5) FastAPI + 掛載 Gradio（Render 用） ----------
app = FastAPI()

@app.get("/health")
def health_check():
    return {"status": "ok"}

# 將 Gradio 掛在根路徑；不要在程式內呼叫 demo.launch()
app = gr.mount_gradio_app(app, demo, path="/")
