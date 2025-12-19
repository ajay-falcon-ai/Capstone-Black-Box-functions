import json
import numpy as np
import json
# Example: using OpenAI client. Swap out for Azure/Anthropic/etc. if needed.
from openai import OpenAI
from string import Template


class LLMCandidateGenerator:
    """
    Candidate generator that uses an LLM to propose new points
    based on prompt patterns defined in config.yaml under model.llm.
    """

    def __init__(self, llm_config: dict):
        self.provider = llm_config.get("provider", "openai")
        self.model = llm_config.get("model", "gpt-4.1")
        self.temperature = llm_config.get("temperature", 0.7)
        self.max_tokens = llm_config.get("max_tokens", 256)
        
        self.top_p = llm_config.get("top_p", 1.0)   # default: full distribution
        self.top_k = llm_config.get("top_k", None)  # default: disabled

        # With simplified config: one active prompt pattern + template
        self.prompt_pattern = llm_config.get("prompt_pattern", "zero-shot")
        self.template = llm_config.get("template", "")

        self.llm_config = llm_config

        if self.provider == "openai":
            self.client = OpenAI()
        else:
            raise NotImplementedError(f"Provider {self.provider} not supported yet.")
    
    def build_prompt(self, X, y, dim: int, examples=None):
        """
        Build a prompt string based on the active pattern in config.
        """
        pattern = self.prompt_pattern

        if pattern == "zero-shot":
            X_list = X.tolist()
            y_list = y.tolist()
            prompt = self.template.replace("{inputs}", json.dumps(X_list)).replace("{outputs}", json.dumps(y_list))
            return prompt

        elif pattern == "few-shot":
            X_list = X.tolist()
            y_list = y.tolist()
            prompt = self.template.replace("{inputs}", json.dumps(X_list)).replace("{outputs}", json.dumps(y_list))
            return prompt

        elif pattern == "chain-of-thought":
            return self.template or "Reason step by step to propose a candidate."

        elif pattern == "structured":
            return f"Output a JSON object with fields: {self.schema}"

        else:
            raise ValueError(f"Unknown prompt pattern: {pattern}")
    
    def parse_response(self, response_text: str):
        """
        Parse the LLM response into a numpy array candidate.
        """
        pattern = self.prompt_pattern  # comes directly from config: "zero-shot", "few-shot", "structured", etc.

        try:
            if pattern == "structured":
                # Expect a JSON object with fields defined in self.schema
                data = json.loads(response_text)
                return np.array([data[p] for p in self.schema])

            else:
                cleaned = response_text.strip()
                if cleaned.startswith("[") and cleaned.endswith("]"):
                    arr = json.loads(cleaned)
                    if isinstance(arr, list):
                        return np.array(arr)
                # fallback: try to extract numbers
                numbers = [float(x) for x in cleaned.replace("[","").replace("]","").split(",")]
                return np.array(numbers)
        except Exception as e:
            print(f"⚠️ Failed to parse response: {e}")
            return None
    
    def generate(self, X, y, dim: int, examples=None):
        """
        Generate a candidate point using the LLM.
        """
        prompt = self.build_prompt(X, y, dim, examples)
        print("Prompt to LLM", prompt)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an optimisation assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            #top_k=self.top_k,
        )
        print("LLM response", response)
        text = response.choices[0].message.content
        candidate = self.parse_response(text)
        return candidate