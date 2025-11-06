# ============================================
# 🧳 旅遊小管家（FAISS + GPT + 語音）
# ============================================
import os
import tempfile
import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# ================================
# 0️⃣ 讀取環境變數
# ================================
load_dotenv()  # 讀取 .env 檔
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("❌ 找不到 OPENAI_API_KEY，請在 .env 中設定")

client = OpenAI(api_key=api_key)

# ================================
# 1️⃣ 載入 FAISS 資料庫
# ================================
# ...existing code...
faiss_db_path = "faiss1022_db"
# 嘗試常見位置，如果有把資料夾放在 faiss_1022db/faiss1022_db
if not os.path.exists(faiss_db_path):
    alt_path = os.path.join("faiss_1022db", "faiss1022_db")
    if os.path.exists(alt_path):
        faiss_db_path = alt_path
    else:
        # 嘗試在當前目錄下搜尋名為 faiss1022_db 的子資料夾
        for root, dirs, files in os.walk("."):
            if "faiss1022_db" in dirs:
                faiss_db_path = os.path.join(root, "faiss1022_db")
                break

embedding_model_name = "text-embedding-3-small"
embedding_model = OpenAIEmbeddings(model=embedding_model_name, openai_api_key=api_key)

db = None
if os.path.exists(faiss_db_path):
    try:
        db = FAISS.load_local(faiss_db_path, embedding_model, allow_dangerous_deserialization=True)
        print(f"✅ 已成功載入 FAISS 資料庫：{faiss_db_path}")
    except Exception as e:
        raise RuntimeError(f"❌ 載入 FAISS 資料庫失敗: {e}")
else:
    raise FileNotFoundError(f"❌ 找不到 FAISS 資料庫資料夾（嘗試過：{faiss_db_path}）")
# ...existing code...

# ================================
# 2️⃣ 查詢功能
# ================================
def query_faiss(user_message, k=1):
    if db is None:
        return None
    try:
        docs = db.similarity_search(user_message, k=k)
        if docs:
            return docs[0].page_content
    except Exception as e:
        print(f"⚠️ FAISS 查詢失敗: {e}")
    return None

# ================================
# 3️⃣ 系統設定
# ================================
SYSTEM_PROMPT = (
    "你是一個友善的『旅遊小管家』，幫助使用者解答與旅遊有關的問題。"
    "如果問題與旅遊無關或資料不足，請回答：『我沒有這方面的資料哦～』"
)

CHAT_MODEL = "gpt-4o-mini"
WHISPER_MODEL = "whisper-1"
TTS_MODEL = "gpt-4o-mini-tts"
TTS_VOICE = "alloy"

# ================================
# 4️⃣ GPT + FAISS 回覆邏輯
# ================================
def chat_with_openai(user_message, history):
    if not user_message:
        return "", history, None

    faiss_answer = query_faiss(user_message)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for q, a in history:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": user_message})

    if faiss_answer:
        messages.insert(0, {"role": "system", "content": f"以下是知識庫資訊:\n{faiss_answer}"})

    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0.6,
        )
        reply = response.choices[0].message.content
    except Exception as e:
        reply = f"⚠️ 無法取得回覆：{e}"

    history.append((user_message, reply))

    # TTS 語音輸出
    audio_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            with client.audio.speech.with_streaming_response.create(
                model=TTS_MODEL,
                voice=TTS_VOICE,
                input=reply,
            ) as resp:
                resp.stream_to_file(tmp.name)
            audio_path = tmp.name
    except Exception as e:
        print(f"🔇 TTS 錯誤：{e}")

    return "", history, audio_path

# ================================
# 5️⃣ 語音辨識
# ================================
def audio_to_text(audio_file, history):
    if audio_file is None:
        return "", history, None

    try:
        with open(audio_file, "rb") as f:
            result = client.audio.transcriptions.create(
                model=WHISPER_MODEL,
                file=f,
                language="zh",
            )
        text = result.text.strip()
    except Exception as e:
        text = f"語音辨識失敗：{e}"

    return chat_with_openai(text, history)

# ================================
# 6️⃣ Gradio 網頁介面
# ================================
with gr.Blocks(title="🧳 旅遊小管家（FAISS + GPT）") as demo:
    gr.Markdown("## 🧭 歡迎使用旅遊小管家\n可輸入文字或錄音提問～")

    chatbot = gr.Chatbot(label="旅遊對話區")

    with gr.Column():
        audio_input = gr.Audio(label="🎤 語音輸入", type="filepath")
        audio_output = gr.Audio(label="🔊 語音回覆", type="filepath")

    msg = gr.Textbox(label="輸入你的問題", placeholder="例如：推薦台南旅遊行程")

    msg.submit(chat_with_openai, [msg, chatbot], [msg, chatbot, audio_output])
    audio_input.change(audio_to_text, [audio_input, chatbot], [msg, chatbot, audio_output])

demo.launch(server_name="0.0.0.0", server_port=7860)  # for local test
