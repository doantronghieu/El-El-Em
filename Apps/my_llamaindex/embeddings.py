from llama_index.embeddings.openai import OpenAIEmbedding

openai_embeddings = {
  "TEXT_EMBED_ADA_002": OpenAIEmbedding(model="text-embedding-ada-002"),
  "TEXT_EMBED_3_LARGE": OpenAIEmbedding(model="text-embedding-3-large"),
}