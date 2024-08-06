import os
from langchain_openai import OpenAIEmbeddings
from langchain_cohere import CohereEmbeddings

# "text-embedding-3-large", "text-embedding-ada-002"

class CustomOpenAIEmbeddings(OpenAIEmbeddings):

  def __init__(
    self,
    model: str = "text-embedding-ada-002",
    *args, **kwargs
  ):
    openai_api_key = os.getenv("OPENAI_API_KEY")
    super().__init__(openai_api_key=openai_api_key, *args, **kwargs)

  def _embed_documents(self, texts):
    embeddings = [
        self.client.create(
          input=text, model="text-embedding-ada-002").data[0].embedding
        for text in texts
    ]
    return embeddings

  def __call__(self, input):
    return self._embed_documents(input)
