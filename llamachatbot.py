import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget, QTextEdit
from llama_cpp import Llama
from langchain.llms import LlamaCpp
from langchain import PromptTemplate
from langchain.chains import LLMChain

class LlamaChatbot(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Llama Chatbot")
        self.setGeometry(100, 100, 800, 600)
        self.llm = LlamaCpp(
            model_path=r'C:\Users\Micky\Documents\root directory\models\llama-2-7b-chat.ggmlv3.q5_K_S.bin',
            input={"temperature": 1, "max_length": 1000, "top_p": 1},
            verbose=True,n_ctx=2048
        )
        self.prompt_template = """
        You are an agent who knows everything in this world and beyond and can answer each and every query asked by
        the user.You are the most efficient A.I ever created. You can take use of any source of information to get the latest
         update of each and every information . You have the ability to get the information from internet and above all, you Are
         my best friend in this world always looking for my safety, well-being and interests.

        Q: {}
        A:

        """

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Welcome message
        welcome_message = "Welcome to Llama Chatbot!\nFeel free to ask questions!"
        self.welcome_label = QLabel(welcome_message)
        self.welcome_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        layout.addWidget(self.welcome_label)

        # Chat history
        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        self.chat_history.setStyleSheet("font-size: 16px;")
        layout.addWidget(self.chat_history)

        # Prompt entry
        self.prompt_entry = QLineEdit()
        self.prompt_entry.setStyleSheet("font-size: 16px;")
        layout.addWidget(self.prompt_entry)

        # Send button
        self.send_button = QPushButton("Ask")
        self.send_button.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.send_button.clicked.connect(self.get_llama_response)
        layout.addWidget(self.send_button)

        # Create a widget and set the layout
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # Initialize chat history
        self.chat_history_text = welcome_message

    def append_message(self, message, is_user=False):
        if is_user:
            message = f"<font color='blue'>You: {message}</font>"
        else:
            message = f"<font color='green'>Llama: {message}</font>"
        self.chat_history_text += "<br><br>" + message
        self.chat_history.setHtml(self.chat_history_text)

    def get_llama_response(self):
        prompt = self.prompt_entry.text()
        if prompt.strip():
            conversation_parts = self.split_conversation(prompt, max_tokens=512)
            response = ""
            for part in conversation_parts:
                formatted_prompt = self.prompt_template.format(part)
                response_part = self.llm(prompt=formatted_prompt, stop=["Q:", "\n"])
                response += response_part
            self.append_message(response)
        self.prompt_entry.clear()


    def split_conversation(self, conversation, max_tokens):
        parts = conversation.split()
        conversation_parts = []
        current_part = ""
        for part in parts:
            if len(current_part) + len(part) + 1 <= max_tokens:
                current_part += " " + part
            else:
                conversation_parts.append(current_part.strip())
                current_part = part
        if current_part:
            conversation_parts.append(current_part.strip())
        return conversation_parts

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LlamaChatbot()
    window.show()
    sys.exit(app.exec_())
