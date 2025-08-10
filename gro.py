import os
import time
from datetime import datetime
from groq import Groq

# ==== Config ====
api = "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"
MODEL = "llama-3.3-70b-versatile"
ENABLE_EXECUTION = True  # Set to False to skip actual code execution
SAVE_FOLDER = "saved_versions"

# ==== Groq API Setup ====
client = Groq(api_key=api)

def get_response(prompt):
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=MODEL,
    )
    return chat_completion.choices[0].message.content

# ==== LLM Agent Class ====
class LLM_Agent:
    def __init__(self, name, system_prompt="You are a helpful assistant."):
        self.name = name
        self.system_prompt = system_prompt
        self.chat_history = []

    def chat(self, message):
        full_prompt = self.build_prompt(message)
        response = get_response(full_prompt)
        self.chat_history.append(("user", message))
        self.chat_history.append(("assistant", response))
        return response

    def build_prompt(self, new_message):
        prompt = f"{self.system_prompt}\n"
        for role, msg in self.chat_history:
            prefix = "User:" if role == "user" else "Assistant:"
            prompt += f"{prefix} {msg}\n"
        prompt += f"User: {new_message}\nAssistant:"
        return prompt

# ==== Tool: Execute Code Safely ====
def run_code(code_str):
    try:
        exec_globals = {}
        exec(code_str, exec_globals)
        return f"[Executed] Returned globals: {list(exec_globals.keys())}"
    except Exception as e:
        return f"[Error during execution] {e}"

# ==== Tool: Save Code Version ====
def save_code(code_str, critic_feedback=""):
    os.makedirs(SAVE_FOLDER, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = os.path.join(SAVE_FOLDER, f"code_{timestamp}.py")
    with open(filename, "w") as f:
        f.write("# --- Auto-generated code ---\n")
        f.write(code_str)
        f.write("\n\n# --- Critic Feedback ---\n")
        f.write(critic_feedback)
    return filename

# ==== Main Task Flow ====
def main():
    print("🔁 Intelligent Dev Environment (Groq LLM + Tools)")
    task = input("\n🧠 Enter your task (e.g., 'Write a function that parses CSV'): ")

    coder = LLM_Agent("Coder", system_prompt="You are a brilliant Python developer. Return only code unless asked otherwise.")
    critic = LLM_Agent("Critic", system_prompt="You are a strict code reviewer. Review the code for correctness, efficiency, and style.")

    # Step 1: Code Generation
    print("\n[Coder] Thinking...")
    code = coder.chat(task)
    print("\n🧑‍💻 Generated Code:\n", code)

    # Step 2: Critique
    print("\n[Critic] Reviewing...")
    review = critic.chat(code)
    print("\n🧠 Critic Feedback:\n", review)

    # Step 3: Execution (Optional)
    if ENABLE_EXECUTION:
        print("\n⚙️ Executing code...")
        exec_result = run_code(code)
        print("\n💻 Execution Result:\n", exec_result)

    # Step 4: Save version
    saved_file = save_code(code, review)
    print(f"\n📁 Code and feedback saved to: {saved_file}")

# ==== Run ====
if __name__ == "__main__":
    main()