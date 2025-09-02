import boto3
from langchain_aws import ChatBedrock
from typing import List

class SupervisorAgent:
    def __init__(self):
        self.llm = ChatBedrock(
            client=boto3.client(
                "bedrock-runtime",
                region_name="us-east-1",
            ),
            model_id="meta.llama3-8b-instruct-v1:0",
            model_kwargs={"temperature": 0.1, "max_gen_len": 128,},
        )
        self.agent_categories = [
            "Meal Planning and Nutrition",
            "Fitness Coaching",
            "Mental Wellness",
            "Sleep Coach",
            "Workplace Ergonomics",
            "Symptom Checker and Care Navigator",
            "Hydration Reminder",
            "Wellness Analytics",
            "Other Inquiries"
        ]

    def classify(self, query: str) -> List[str]:
        """
        Classifies the user query into one or more of the predefined agent categories.
        """
        category_list = "\n".join([f"- {category}" for category in self.agent_categories])

        prompt = f"""You are an expert at routing user queries to the correct specialized agent(s).
        Based on the user's query, please identify all appropriate agents from the following list.
        Your response MUST be a comma-separated list of the agent names, and nothing else.

        Available Agents:
        {category_list}

        User Query: "{query}"

        Appropriate Agent(s):"""

        response = self.llm.invoke(prompt)
        # The model's raw output is in the 'content' attribute
        raw_classification = response.content.strip()

        # Parse the comma-separated string and validate against the agent_categories list
        classifications = [agent.strip() for agent in raw_classification.split(',')]
        
        # Filter out any invalid or hallucinated agent names
        valid_classifications = [agent for agent in classifications if agent in self.agent_categories]

        if not valid_classifications:
            # Handle cases where the model returns nothing valid
            print(f"Warning: Model returned no valid categories for output: '{raw_classification}'. Defaulting.")
            return ["General Inquiry"]

        return valid_classifications

# Example usage:
if __name__ == "__main__":
    classifier = ClassifierAgent()

    queries = [
        "I have a persistent headache and feel a bit dizzy.",
        "What's a good 15-minute workout I can do during my lunch break?",
        "I've been feeling really stressed out and anxious lately, and I'm not sleeping well because of it.",
        "Can you suggest a healthy recipe for dinner tonight?",
        "Show me my wellness summary for the past week.",
        "My back hurts from sitting all day, what can I do?",
    ]

    for q in queries:
        agent_names = classifier.classify(q)
        print(f"Query: '{q}'\nRouted to: {agent_names}\n" + "-"*30)
