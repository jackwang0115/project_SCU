from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from huggingface_hub import snapshot_download
import os
import shutil
from tqdm.asyncio import tqdm
import asyncio

def download_embedding_model():
    model_name = "GanymedeNil/text2vec-large-chinese"
    try:
        snapshot_download(repo_id=model_name, local_dir="./text2vec-large-chinese")
        print(f"模型 '{model_name}' 已下載到 './text2vec-large-chinese'")
    except Exception as e:
        print(f"下載模型時發生錯誤: {e}")
        exit(1)

async def load_text_file(file_path: str) -> list:
    try:
        loader = TextLoader(file_path, encoding='utf-8')
        raw_docs = loader.load()

        # 讀取最後一行作為可能的 URL
        # 這個邏輯保持不變，因為是在創建初始 Document 物件時賦予元數據
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            possible_url = lines[-1].strip() if lines else ''

        return [
            Document(
                page_content=doc.page_content,
                metadata={"source_url": possible_url}
            ) for doc in raw_docs
        ]
    except Exception as e:
        print(f"錯誤：{file_path} - {e}")
        return []

async def load_documents_from_folder(folder_path: str) -> list:
    documents = []
    tasks = []

    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith('.txt'):
                file_path = os.path.join(root, filename)
                tasks.append(load_text_file(file_path))

    for result in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="載入文件"):
        try:
            file_documents = await result
            if file_documents:
                documents.extend(file_documents)
        except Exception as e:
            print(f"處理文件時發生錯誤: {e}")
            continue

    return documents

# ===== 修改後的函數 =====
def split_documents(documents: list, chunk_size: int = 600, chunk_overlap: int = 100) -> list:
    """
    將 Document 物件列表分割成更小的文本塊。
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    # 直接對 Document 物件列表進行分割。
    # 分割器會自動將原始文件的元數據（metadata）正確地傳遞給所有分割後產生的文本塊（chunks）。
    # 這避免了之前手動分配元數據時的邏輯錯誤。
    texts = text_splitter.split_documents(documents)
    return texts
# ===== 修改結束 =====

async def create_embeddings_and_store(texts: list, persist_directory: str):
    try:
        embeddings_client = HuggingFaceEmbeddings(
            model_name="./text2vec-large-chinese",
            model_kwargs={'device': 'cuda'}
        )
    except Exception as e:
        print(f"建立 embeddings 時發生錯誤: {e}")
        exit()

    if os.path.exists(persist_directory):
        try:
            print(f"檢測到舊的向量資料庫，正在刪除 {persist_directory}...")
            shutil.rmtree(persist_directory)
            print(f"已刪除")
        except Exception as e:
            print(f"刪除舊向量資料庫時發生錯誤：{str(e)}")
            exit(1)

    try:
        print("開始建立向量資料庫...")
        vectordb = await Chroma.afrom_documents(
            documents=texts,
            embedding=embeddings_client,
            persist_directory=persist_directory
        )
        print("向量資料庫建立完成")
    except Exception as e:
        print(f"建立向量資料庫時發生錯誤：{str(e)}")
        exit(1)

async def main():
    backup_folder = 'test'
    persist_directory = "chroma_db"

    if not os.path.exists("./text2vec-large-chinese"):
        download_embedding_model()

    if not os.path.exists(backup_folder):
        print(f"錯誤：找不到 {backup_folder} 資料夾")
        exit(1)

    documents = await load_documents_from_folder(backup_folder)
    if not documents:
        print("錯誤：沒有找到可讀取的 txt 文件")
        exit(1)
    
    # 使用修改後的分割函數
    texts = split_documents(documents)
    
    await create_embeddings_and_store(texts, persist_directory)

if __name__ == "__main__":
    asyncio.run(main())