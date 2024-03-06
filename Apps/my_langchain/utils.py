from pprint import pprint

def print_documents(docs, is_pprint=True):
    for doc in docs:
        if is_pprint:
            pprint(doc.page_content)
        else:
            print(doc.page_content)

def print_docs_with_metadata_and_score(docs):
    for doc, score in docs:
        content = getattr(doc, "page_content", None)
        metadata = getattr(doc, "metadata", None)

        print(f"Content:\n{content}")

        if metadata:
            print(f"Metadata:\n{metadata}")

        if score is not None:
            print(f"Score:\n{score}")

        print('-'*80) 
