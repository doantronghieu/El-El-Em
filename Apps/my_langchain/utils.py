import csv
from pprint import pprint
from langchain_core.documents import Document


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


def format_docs(docs: list[Document]):
    return "\n\n".join(doc.page_content for doc in docs)


def update_metadata(docs: list[Document], metadatas: dict):
    for k, v in metadatas.items():
        for doc in docs:
            doc.metadata[k] = v


def remove_metadata(docs: list[Document], key_to_remove: str):
    for doc in docs:
        # Check if the key exists in the metadata
        if key_to_remove in doc.metadata:
            # Remove the specified key from metadata
            del doc.metadata[key_to_remove]


def get_csv_column_names(file_path):
    with open(file_path, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        # Assuming the first row contains the column names
        column_names = next(reader, None)

    # Remove leading spaces from column names
    if column_names:
        column_names = [name.strip() for name in column_names]

    return column_names
