# ================================
# ✨ 旅遊小管家（FAISS + GPT + 語音，美化版，Render-ready）
# ================================
import os
import tempfile
from pathlib import Path

import gradio as gr
from fastapi import FastAPI
from openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# ---------- 0) 基本設定 ----------
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ 請先在環境變數設定 OPENAI_API_KEY")

client = OpenAI(api_key=api_key)
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)

# 使用程式所在目錄為基準，避免工作目錄不同造成路徑錯誤
BASE_DIR = Path(__file__).parent.resolve()
faiss_db_path = BASE_DIR / "faiss1022_db"

if not faiss_db_path.exists():
    raise FileNotFoundError(f"❌ 找不到 FAISS 資料夾：{faiss_db_path}")

# 載入 FAISS（允許反序列化）
db = FAISS.load_local(
    str(faiss_db_path),
    embedding_model,
    allow_dangerous_deserialization=True
)
print(f"✅ 已成功載入 FAISS：{faiss_db_path}")

# ---------- 1) 模型參數 ----------
SYSTEM_PROMPT = (
    "你是一個友善的『旅遊小管家』，幫助使用者解答旅遊相關問題。"
    "如果資料不足或問題與旅遊無關，請回答：『我沒有這方面的資料哦～』"
)
CHAT_MODEL = "gpt-4o-mini"
WHISPER_MODEL = "whisper-1"
TTS_MODEL = "gpt-4o-mini-tts"
TTS_VOICE = "alloy"

# ---------- 2) 功能 ----------
def query_faiss(user_message: str, k: int = 1):
    try:
        docs = db.similarity_search(user_message, k=k)
        return docs[0].page_content if docs else None
    except Exception as e:
        print(f"⚠️ FAISS 查詢失敗: {e}")
        return None

def chat_with_openai(user_message, history):
    if not user_message:
        return "", history, None

    faiss_answer = query_faiss(user_message)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for q, a in history:
        messages += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    messages.append({"role": "user", "content": user_message})

    if faiss_answer:
        messages.insert(0, {"role": "system", "content": f"資料庫資訊：\n{faiss_answer}"})

    try:
        resp = client.chat.completions.create(model=CHAT_MODEL, messages=messages, temperature=0.6)
        reply = resp.choices[0].message.content
    except Exception as e:
        reply = f"⚠️ 無法取得回覆：{e}"

    history.append((user_message, reply))

    # TTS（失敗不影響主流程）
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
            result = client.audio.transcriptions.create(model=WHISPER_MODEL, file=f, language="zh")
        text = result.text.strip()
    except Exception as e:
        text = f"語音辨識失敗：{e}"
    return chat_with_openai(text, history)

# ---------- 3) UI（Gradio Blocks） ----------
custom_css = """
body { background: linear-gradient(135deg,#f5f7fa,#c3cfe2); font-family:'Microsoft JhengHei',sans-serif; }
.gradio-container { max-width: 900px !important; margin: auto; border-radius: 20px; box-shadow:0 0 25px rgba(0,0,0,0.1); background:white; }
h1,h2,h3 { text-align:center; color:#333; }
button { border-radius:10px !important; font-weight:bold; transition:.2s; }
button:hover { background-color:#4a90e2 !important; color:white !important; }
"""

with gr.Blocks(title="🧳 旅遊小管家", css=custom_css, theme=gr.themes.Glass()) as demo:
    gr.Markdown("""
    # 🌏 旅遊小管家
    請輸入文字或錄音提問，小管家會結合知識庫與 GPT 提供建議 ✈️
    ---
    """)
    chatbot = gr.Chatbot(label="💬 對話區", height=420)
    with gr.Row():
        msg = gr.Textbox(label="輸入問題", placeholder="例如：幫我安排台南兩天一夜行程", scale=3)
        send_btn = gr.Button("🚀 送出", variant="primary", scale=1)
    with gr.Column():
        audio_in = gr.Audio(label="🎙️ 語音輸入（錄音或上傳）", type="filepath")
        audio_out = gr.Audio(label="🔊 語音回覆", type="filepath")
    send_btn.click(chat_with_openai, [msg, chatbot], [msg, chatbot, audio_out])
    msg.submit(chat_with_openai, [msg, chatbot], [msg, chatbot, audio_out])
    audio_in.change(audio_to_text, [audio_in, chatbot], [msg, chatbot, audio_out])

# ---------- 4) FastAPI + mount Gradio（Render 必備） ----------
app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

# 將 Gradio 掛在根路徑
app = gr.mount_gradio_app(app, demo, path="/")

# 注意：在 Render 上用 uvicorn 啟動：
# Start Command（Render 面板）請設為：
#   uvicorn finalapp:app --host 0.0.0.0 --port $PORT
#
# 不要在容器內再呼叫 demo.launch()（會阻塞/衝突）
