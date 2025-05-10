import requests
from typing_extensions import Self
from typing import TypedDict
from promptflow.tracing import trace
import os

class BaseModel:
    """
    Base class for model endpoints.
    """

    def __init__(self, model_deployment_name: str):
        self.model_deployment_name = model_deployment_name

    def __call__(self, query: str):
        """
        Invoke the model with the given query.
        """

        from openai import AzureOpenAI

        client = AzureOpenAI(
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
            api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
        )

        completion = client.chat.completions.create(
            model=self.model_deployment_name, 
            messages=[
                {
                    "role": "user",
                    "content": query,
                },
            ],
        )

        response = completion.choices[0].message.content
        return {"response": response}
