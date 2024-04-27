MODELS = {
  "OPENAI":
  {
    "GPT-3.5-TURBO-0125": "gpt-3.5-turbo-0125",
    "GPT-4-TURBO-PREVIEW": "gpt-4-turbo-preview",
    "GPT-3.5-TURBO-L.INDEX": "gpt-3.5-turbo",
    "GPT-4-L.INDEX": "gpt-4",
  },
  "ANTHROPIC":
  {
    "CLAUDE-3-HAIKU-20240307": "claude-3-haiku-20240307",
    "CLAUDE-3-OPUS-20240229": "claude-3-opus-20240229",
    "CLAUDE-3-SONNET-20240229": "claude-3-sonnet-20240229",
  },
  "COHERE": {
    "COMMAND": "command",
    "COMMAND-R": "command-r",
    "RERANK-ENGLISH-V3.0": "rerank-english-v3.0",
    "RERANK-MULTILINGUAL-V3.0": "rerank-multilingual-v3.0",
  }
}

EMBEDDINGS = {
  "OPENAI":
  {
    "TEXT-EMBEDDING-3-LARGE": "text-embedding-3-large",
    "TEXT-EMBEDDING-ADA-002": "text-embedding-ada-002",
  },
  "COHERE":
  {
    "EMBED-EMBED-ENGLISH-V3.0": "embed-embed-english-v3.0",
    "EMBED-MULTILINGUAL-V3.0": "embed-multilingual-v3.0",
    "EMBED-ENGLISH-LIGHT-V3.0": "embed-english-light-v3.0",
    "EMBED-MULTILINGUAL-LIGHT-V3.0": "embed-multilingual-light-v3.0",
    "EMBED-ENGLISH-V2.0": "embed-english-v2.0",
    "EMBED-ENGLISH-LIGHT-V2.0": "embed-english-light-v2.0",
    "EMBED-MULTILINGUAL-V2.0": "embed-multilingual-v2.0",
  },
  "SIZE":
  {
    "TEXT-EMBEDDING-3-LARGE": 1536,
    "TEXT-EMBEDDING-ADA-002": 3072,
    
    "EMBED-EMBED-ENGLISH-V3.0": 1024,
    "EMBED-MULTILINGUAL-V3.0": 1024,
    "EMBED-ENGLISH-LIGHT-V3.0": 384,
    "EMBED-MULTILINGUAL-LIGHT-V3.0": 384,
    "EMBED-ENGLISH-V2.0": 4096,
    "EMBED-ENGLISH-LIGHT-V2.0": 1024,
    "EMBED-MULTILINGUAL-V2.0": 768,
  },
  "DISTANCE":
  {
    "COSINE": "Cosine",
    "EUCLID": "Euclid",
    "DOT": "Dot",
    "MANHATTAN": "Manhattan",
  }
}