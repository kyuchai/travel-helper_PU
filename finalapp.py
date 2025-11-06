# ================================
# 🧳 旅遊小管家（FAISS + GPT + 語音 美化版 / Render-ready）
# ================================
import os
import tempfile
import gradio as gr

from openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# ================================
# 0️⃣ 初始化環境與模型
# ================================
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ 請先在 Render 的環境變數設定 OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

# 允許開關 TTS（Render 環境偶爾會缺音訊編解碼器）
ENABLE_TTS = os.environ.get("ENABLE_TTS", "1") == "1"

# OpenAI 模型
CHAT_MODEL = "gpt-4o-mini"
WHISPER_MODEL = "whisper-1"
TTS_MODEL = "gpt-4o-mini-tts"
TTS_VOICE = "alloy"

# LangChain 嵌入模型
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=api_key
)

# ================================
# FAISS 載入（可選）
# ================================
faiss_db_path = "faiss1022_db"
db = None

if os.path.isdir(faiss_db_path):
    try:
        db = FAISS.load_local(
            faiss_db_path,
            embedding_model,
            allow_dangerous_deserialization=True
        )
        print("✅ FAISS 資料庫載入成功")
    except Exception as e:
        print(f"⚠️ 無法載入 FAISS：{e}（將以無知識庫模式啟動）")
else:
    print(f"⚠️ 找不到 FAISS 資料夾：{faiss_db_path}，以無知識庫模式啟動")

# ================================
# 系統提示
# ================================
SYSTEM_PROMPT = (
    "你是一個友善的『旅遊小管家』，幫助使用者解答旅遊相關問題。"
    "如果資料不足或問題與旅遊無關，請回答：『我沒有這方面的資料哦～』"
)

# ================================
# 1️⃣ FAISS 查詢
# ================================
def query_faiss(user_message: str, k: int = 1):
    if not db:
        return None
    try:
        docs = db.similarity_search(user_message, k=k)
        return docs[0].page_content if docs else None
    except Exception as e:
        print(f"⚠️ FAISS 查詢失敗: {e}")
        return None

# ================================
# 2️⃣ 與 OpenAI 對話 +（可選）TTS
# ================================
def chat_with_openai(user_message, history):
    if not user_message:
        return "", history, None

    # 取 FAISS 內容補充
    faiss_answer = query_faiss(user_message)

    # 組訊息
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if faiss_answer:
        messages.insert(0, {"role": "system", "content": f"以下是資料庫提供的旅遊資訊:\n{faiss_answer}"})
    for q, a in history:
        messages += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    messages.append({"role": "user", "content": user_message})

    # LLM 回覆
    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0.6
        )
        reply = response.choices[0].message.content
    except Exception as e:
        reply = f"⚠️ 無法取得回覆：{e}"

    history.append((user_message, reply))

    # TTS（可關閉）
    audio_path = None
    if ENABLE_TTS:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                with client.audio.speech.with_streaming_response.create(
                    model=TTS_MODEL, voice=TTS_VOICE, input=reply
                ) as resp:
                    resp.stream_to_file(tmp.name)
                audio_path = tmp.name
        except Exception as e:
            print(f"⚠️ TTS 錯誤：{e}")

    return "", history, audio_path

# ================================
# 3️⃣ 語音輸入 → Whisper
# ================================
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

# ================================
# 4️⃣ 自訂 CSS
# ================================
custom_css = """
body { background: linear-gradient(135deg, #f5f7fa, #c3cfe2); font-family: 'Microsoft JhengHei', sans-serif; }
.gradio-container { max-width: 900px !important; margin: auto; border-radius: 20px; box-shadow: 0 0 25px rgba(0,0,0,0.1); background: white; }
h1,h2,h3 { text-align: center; color: #333; }
button { border-radius: 10px !important; font-weight: bold; transition: 0.2s; }
button:hover { background-color: #4a90e2 !important; color: white !important; }
"""

# ================================
# 5️⃣ Gradio 介面
# ================================
with gr.Blocks(title="🧳 旅遊小管家", css=custom_css, theme=gr.themes.Glass()) as demo:
    gr.Markdown("""
    # 🌏 旅遊小管家
    歡迎使用！輸入文字或語音提問，小管家會根據知識庫與 GPT 提供建議 ✈️
    ---
    """)

    chatbot = gr.Chatbot(label="💬 對話區", height=400)

    with gr.Row():
        msg = gr.Textbox(label="輸入問題", placeholder="例如：幫我安排花蓮兩天一夜行程", scale=3)
        send_btn = gr.Button("🚀 送出", variant="primary", scale=1)

    with gr.Column():
        audio_input = gr.Audio(label="🎙️ 語音輸入（錄音或上傳）", type="filepath")
        audio_output = gr.Audio(label="🔊 語音回覆", type="filepath")

    send_btn.click(chat_with_openai, [msg, chatbot], [msg, chatbot, audio_output])
    msg.submit(chat_with_openai, [msg, chatbot], [msg, chatbot, audio_output])
    audio_input.change(audio_to_text, [audio_input, chatbot], [msg, chatbot, audio_output])

# ================================
# 6️⃣ 啟動（Render-friendly）
# ================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port, show_api=False, debug=False)
