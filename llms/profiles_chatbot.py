import os
from mistralai import Mistral
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()


def get_profiles_details():
    try:
        mongo_uri = "mongodb+srv://hiretalent-dev:Yulwbmn87x92EQ0U@hiretalent.doscksq.mongodb.net/"
        client = MongoClient(mongo_uri)
        db = client["app-dev"]
        collection = db["profiles"]

        profiles = list(collection.find({}, {"_id": 0}))

        if not profiles:
            return "No profiles found."

        context_lines = []

        for profile in profiles:
            name = f"{profile.get('firstName', '')} {profile.get('lastName', '')}".strip()
            summary = profile.get("carrierSummary", "")
            expertise = profile.get("areaOfExpertise", "")
            skills = [s.get("skill", "") for s in profile.get("highlightedSkills", [])]

            experience_lines = []
            for exp in profile.get("experience", []):
                company = exp.get("company", "")
                position = exp.get("position", "")
                responsibility = exp.get("responsibilityDescription", "")
                exp_text = f"{position} at {company}: {responsibility}"
                experience_lines.append(exp_text)

            context_lines.append(
                f"Name: {name}\nExpertise: {expertise}\nSummary: {summary}\n"
                f"Skills: {', '.join(skills)}\nExperience:\n" + "\n".join(experience_lines) + "\n"
            )

        return "\n".join(context_lines)

    except Exception as e:
        return f"Error retrieving profiles: {e}"


class ChatBot:
    def __init__(self, api_key, model="mistral-large-latest"):
        self.api_key = api_key
        self.model = model
        self.conversation_history = []
        self.mistral_client = Mistral(api_key=api_key)

        context = get_profiles_details()
        self.conversation_history.append({
            "role": "system",
            "content": "You are a helpful assistant who answers questions using the following user profiles:\n" + context
        })

    def run(self):
        print("Chatbot is running. Ask a question or type 'exit' to quit.")
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() in ["exit", "quit"]:
                break
            self.send_request(user_input)

    def send_request(self, user_input):
        self.conversation_history.append({"role": "user", "content": user_input})
        buffer = ""
        stream_response = self.mistral_client.chat.stream(
            model=self.model,
            messages=self.conversation_history
        )

        for chunk in stream_response:
            delta = chunk.data.choices[0].delta.content
            if delta:
                buffer += delta
                print(delta, end="", flush=True)

        print("\n")
        self.conversation_history.append({"role": "assistant", "content": buffer})


if __name__ == "__main__":
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        print("Please set the MISTRAL_API_KEY environment variable.")
        exit(1)

    chatbot = ChatBot(api_key=api_key)
    chatbot.run()
