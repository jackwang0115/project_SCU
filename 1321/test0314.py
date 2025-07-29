import logging
from typing import List, Dict, Any
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.docstore.document import Document
import streamlit as st
import asyncio
import httpx

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('qa_system.log')
    ]
)

@st.cache_resource
def get_embeddings() -> HuggingFaceEmbeddings:
    logging.info("Initializing embedding model...")
    return HuggingFaceEmbeddings(model_name="./text2vec-large-chinese", model_kwargs={'device': 'cuda'})

@st.cache_resource
def get_llm() -> Ollama:
    logging.info("Initializing LLaMA model...")
    return Ollama(model="phi4:latest")

class SoochowQASystem:
    # ===== 新增：關鍵字與網址對應表 =====
    # 您可以在這裡不斷擴充，讓系統的建議更精準
    FALLBACK_MAP = {
        ('圖書館', '借書', '還書', '自習', '資料庫'): {
            'name': '東吳大學圖書館',
            'url': 'https://www.library.scu.edu.tw/'
        },
        ('註冊', '選課', '成績', '學分', '畢業', '雙主修', '輔系'): {
            'name': '東吳大學教務處',
            'url': 'http://www.scu.edu.tw/acad/'
        },
        ('宿舍', '住宿', '床位'): {
            'name': '東吳大學學務處住宿組',
            'url': 'https://www.scu.edu.tw/osa/residence-service-section/'
        },
        ('獎學金', '就學貸款', '學貸'): {
            'name': '東吳大學學務處德育中心',
            'url': 'https://www.scu.edu.tw/osa/virtue-cultivation-center/'
        },
        ('校園網路', 'wifi', 'email'): {
            'name': '東吳大學電算中心',
            'url': 'https://www.scu.edu.tw/cc/'
        }
    }
    DEFAULT_FALLBACK = {
        'name': '東吳大學',
        'url': 'http://www.scu.edu.tw/'
    }
    # ===== 修改結束 =====

    def __init__(self):
        self.embeddings = get_embeddings()
        self.llm = get_llm()
        self.vector_store = self._init_vector_store()

    def _init_vector_store(self) -> Chroma:
        if "vector_store" not in st.session_state:
            logging.info("Initializing Chroma vector store...")
            st.session_state["vector_store"] = Chroma(
                persist_directory="chroma_db",
                embedding_function=self.embeddings
            )
        return st.session_state["vector_store"]
        
    # ===== 新增：後備建議方法 =====
    def _get_fallback_suggestion(self, query: str) -> Dict[str, str]:
        """
        根據查詢中的關鍵字，從 FALLBACK_MAP 中查找最相關的建議。
        """
        for keywords, suggestion in self.FALLBACK_MAP.items():
            if any(keyword in query for keyword in keywords):
                logging.info(f"Fallback triggered. Matched keywords: {keywords}")
                return suggestion
        logging.info("Fallback triggered. No specific keywords matched, using default.")
        return self.DEFAULT_FALLBACK
    # ===== 修改結束 =====

    async def retrieve_relevant_documents(self, query: str) -> List[Document]:
        try:
            results = await self.vector_store.amax_marginal_relevance_search(
                query=query,
                k=10,
                fetch_k=50,
                lambda_mult=0.3
            )
            return results
        except Exception as e:
            logging.error(f"Error during document retrieval: {e}", exc_info=True)
            raise

    async def generate_response(self, query: str, context: str) -> str:
        prompt = f"""
===== 東吳大學官方知識引擎 =====

請依據以下檢索到的東吳大學資料，提供符合校方規範的回應：

請基於檢索到的東吳大學機構知識庫資料，遵循最高學術標準與專業規範回應本次查詢。

**執行標準協議：**

1. **絕對資料依從性**：回應必須100%基於提供的東吳大學官方資料。嚴禁生成、推測或引入非提供資料中的任何資訊，無論多麼合理或可能。

2. **資料來源精確引用**：
   - 一級引用：明確標示「根據【具體文件全名】第X條/第X頁...」
   - 二級引用：說明「依據東吳大學【具體部門全稱】...」
   - 引用格式須符合學術規範，每條資訊均需明確溯源

3. **專業術語精確使用**：
   - 所有東吳大學專有名詞（如建築物、部門、制度、計畫名稱）必須使用正式全稱
   - 行政單位與學術單位須以官方全稱表示
   - 人員職稱須嚴格按照東吳大學組織架構規範使用

4. **資訊完整性分級處理**：
   - A級（完整資訊）：資料庫中有完整答案，提供全面回應
   - B級（部分資訊）：資料庫中有相關但不完整資訊，清晰標示已知範圍與未知範圍
   - C級（缺失資訊）：資料庫中無相關資訊，明確告知並提供精確的後續諮詢管道

5. **時效性標記制度**：
   - 動態信息（如活動、截止日期）必須標明「依據【日期/學期】資料」
   - 靜態信息（如建築位置、組織結構）標明「基於現行規定」

6. **層級化回應結構**：
   - 摘要層：20-40字核心答案，直接回應問題本質
   - 主體層：依邏輯順序分段闡述，每段聚焦單一要點
   - 補充層：提供相關規定、例外情況或進階資訊
   - 指引層：後續行動建議、聯絡方式或其他資源

7. **語言與表達規範**：
   - 採用符合高等教育機構水準的學術用語與表達
   - 保持客觀、中立、權威的機構語氣
   - 遵循教育部公文及大學正式文件表達準則
   - 避免使用非正式、口語化或不精確表達

8. **跨領域資訊整合**：在回答涉及多個部門的問題時，需明確區分各部門職責與資訊來源，避免混淆

9. **問題分析與重構**：對複雜問題進行結構化拆解，分項回應各子問題

10. **知識範圍界定**：明確區分東吳大學知識庫涵蓋範圍與非涵蓋範圍

【檢索資料】：
{context}

【使用者提問】：
{query}
"""
        try:
            response = await self.llm.ainvoke(prompt)
            logging.info(f"Generated response of length: {len(response)}")
            return response
        except httpx.HTTPError as e:
            logging.error(f"HTTP error during response generation: {e}", exc_info=True)
            raise
        except Exception as e:
            logging.error(f"Error during response generation: {e}", exc_info=True)
            raise

    async def process_query(self, query: str) -> str:
        try:
            logging.info(f"Processing query: {query}")
            results = await self.retrieve_relevant_documents(query)

            # ===== 修改後的後備（Fallback）邏輯 =====
            if not results:
                logging.warning(f"No documents found for query: '{query}'. Using smart fallback.")
                suggestion = self._get_fallback_suggestion(query)
                
                # 如果匹配到特定主題的關鍵字
                if suggestion['name'] != self.DEFAULT_FALLBACK['name']:
                    return (f"⚠ 系統中查無此問題的明確答案。\n\n"
                            f"不過，您的問題似乎與 **{suggestion['name']}** 相關，"
                            f"建議您參考其官方網站以獲取最準確的資訊：\n"
                            f"🔗 [{suggestion['name']}]({suggestion['url']})")
                else:
                    # 如果沒有匹配到任何關鍵字，提供通用建議
                    return (f"⚠ 系統中查無此問題的明確答案。\n\n"
                            f"建議您嘗試使用不同的關鍵字，或直接前往 **{suggestion['name']}** 官方網站查詢：\n"
                            f"🔗 [{suggestion['name']}]({suggestion['url']})")
            # ===== 修改結束 =====

            # 有搜尋結果時，附上網址
            context = "\n\n".join([
                f"【來源】{doc.metadata.get('source_url', '無網址')}\n{doc.page_content}"
                for doc in results
            ])
            logging.info(f"Retrieved context: {context[:200]}...")
            return await self.generate_response(query, context)

        except Exception as e:
            error_msg = f"處理您的問題時發生錯誤。請檢查日誌檔以獲取更多詳細資訊。"
            logging.error(f"Error processing query: {e}", exc_info=True)
            return error_msg

async def main():
    st.set_page_config(
        page_title="東吳大學智能問答系統",
        page_icon="🎓",
        layout="wide"
    )
    st.title("東吳大學智能問答系統 🎓")

    try:
        qa_system = SoochowQASystem()
        query = st.text_input("請輸入您的問題（例如：東吳大學資料科學系在哪？）")

        if query:
            with st.spinner('正在查詢資料庫並生成回答...'):
                response = await qa_system.process_query(query)
                st.markdown("**回答：**")
                # 使用 st.markdown 讓網址可以被點擊
                st.markdown(response, unsafe_allow_html=True)

    except Exception as e:
        error_msg = f"系統發生未預期的錯誤，請稍後再試。"
        st.error(error_msg)
        logging.error(f"Unexpected error in main: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())