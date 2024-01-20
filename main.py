import tkinter as tk
from tkinter import ttk
from ttkthemes import ThemedTk

from datetime import datetime
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.memory import VectorStoreRetrieverMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate

import faiss

from langchain.docstore import InMemoryDocstore
from langchain.vectorstores import FAISS

from memory11 import BufferWindowVectorRetrieverMemory

import os
os.environ["OPENAI_API_KEY"] = ""

llm = OpenAI(temperature=0.2)  # Can be any valid LLM
_DEFAULT_TEMPLATE = """다음은 인간과 개인 AI 비서간의 대화입니다. AI는 사용자의 정보를 기억하여 다양한 맥락에서의 구체적인 세부정보를 제공합니다.
모르는 정보일 경우 모른다고 답해주세요. 

이전 대화 부분 :
{retrieve_history}

(사용자의 말과 관련없는 정보는 사용할 필요가없습니다)

최근 대화 부분: 
{buffer_history}

You: {input}
당신은 비서 AI입니다. 위 질문에 맞는 적절한 답변을 해주세요. """

PROMPT = PromptTemplate(
    input_variables=["retrieve_history", "buffer_history", "input"], template=_DEFAULT_TEMPLATE
)

memory = BufferWindowVectorRetrieverMemory(
)
conversation_with_summary = ConversationChain(
    llm=llm,
    prompt=PROMPT,
    # We set a very low max_token_limit for the purposes of testing.
    memory=memory,
    verbose=False
)

def send_message():
    input_ = entry.get()
    if input_:
        text_box.insert(tk.END, "You: " + input_ + "\n")
        entry.delete(0, tk.END)
        
        if input_ == "end":
            root.destroy()
        elif input_ == "현재시간":
            current_time = datetime.now().strftime("%H:%M")
            current_date = datetime.now().strftime("%Y년 %m월 %d일")
            input_ = f"현재 시간 : {current_date} {current_time}"
            output = conversation_with_summary.predict(input=input_)
            text_box.insert(tk.END,output + "\n")
        else:
            output = conversation_with_summary.predict(input=input_)
            text_box.insert(tk.END,output + "\n")
            text_box.insert(tk.END, "\n\n")

# GUI 생성
root = ThemedTk(theme="arc")  # arc 테마 사용
root.title("Assistant")

# 텍스트 상자 설정
text_box = tk.Text(root, wrap=tk.WORD, bg="white", font=("Helvetica", 12))
text_box.pack(padx=5, pady=5, expand=True, fill=tk.BOTH)

# 입력창 및 스크롤바 설정
entry_frame = ttk.Frame(root)
entry_frame.pack(padx=5, pady=5, fill=tk.X)

entry = ttk.Entry(entry_frame)
entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

# 메시지 전송 버튼 설정
send_button = ttk.Button(entry_frame, text="Send", command=send_message)
send_button.pack(side=tk.RIGHT)

# 메시지 태그 설정
text_box.tag_configure('user', background="yellow", font=("Helvetica", 12, "bold"))
text_box.tag_configure('assistant', background="lightgrey", foreground="black", font=("Helvetica", 12))

# GUI 실행
root.mainloop()