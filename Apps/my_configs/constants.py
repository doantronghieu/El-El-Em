MODELS = {
    "OPENAI":
    {
      "GPT-3.5-TURBO-0125": "gpt-3.5-turbo-0125",
      "GPT-4-TURBO-PREVIEW": "gpt-4-turbo-preview",
    },
    "ANTHROPIC":
    {
      "CLAUDE-3-HAIKU-20240307": "claude-3-haiku-20240307",
      "CLAUDE-3-OPUS-20240229": "claude-3-opus-20240229",
      "CLAUDE-3-SONNET-20240229": "claude-3-sonnet-20240229",
    },
}

EMBEDDINGS = {
    "OPENAI":
    {
      "TEXT-EMBEDDING-3-LARGE": "text-embedding-3-large",
      "TEXT-EMBEDDING-ADA-002": "text-embedding-ada-002",
    },
    "COHERE":
    {

    },
    "SIZE":
    {
      "TEXT-EMBEDDING-3-LARGE": 1536,
      "TEXT-EMBEDDING-ADA-002": 3072,
    },
    "DISTANCE":
    {
      "COSINE": "Cosine",
      "EUCLID": "Euclid",
      "DOT": "Dot",
      "MANHATTAN": "Manhattan",
    }
}
