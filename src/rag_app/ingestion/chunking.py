from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter

markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
)

def chunk_maker(filepath: str) -> list:
    with open(filepath, "r", encoding="utf-8") as f:
        data = f.read()
    sections = markdown_splitter.split_text(data)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=200)
    file_chunk = text_splitter.split_documents(sections)
    return file_chunk