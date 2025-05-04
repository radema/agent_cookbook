import os
from typing import List, Dict, Any, Tuple

# PDF processing and extraction libraries
from docling.document import Document
from docling.extractors import TextExtractor, TableExtractor
from docling.models import Table

# LlamaIndex components
from llama_index import VectorStoreIndex, ServiceContext
from llama_index.node_parser import SimpleNodeParser
from llama_index.schema import TextNode, MetadataMode
from llama_index.embeddings import OpenAIEmbedding
from llama_index.llms import OpenAI
from llama_index.ingestion import IngestionPipeline
from llama_index.retrievers import MultiVectorRetriever
from llama_index.storage.docstore import SimpleDocumentStore

# Configure API keys (put these in environment variables in production)
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

# Initialize LLM and embedding models
llm = OpenAI(model="gpt-4")
embed_model = OpenAIEmbedding()
service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)

def load_pdf(pdf_path: str) -> Document:
    """
    Load PDF using Docling and return a Document object
    """
    print(f"Loading PDF: {pdf_path}")
    doc = Document.from_pdf(pdf_path)
    return doc

def extract_text_and_tables(doc: Document) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Extract text and tables from Document object
    Returns: tuple of (text_blocks, tables)
    """
    print("Extracting text and tables...")
    
    # Extract text
    text_extractor = TextExtractor()
    text_blocks = text_extractor.extract(doc)
    
    # Extract tables
    table_extractor = TableExtractor()
    tables = table_extractor.extract(doc)
    
    return text_blocks, tables

def generate_summaries(text_blocks: List[Dict[str, Any]], 
                       tables: List[Table]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Generate summaries for text blocks and tables
    """
    print("Generating summaries...")
    
    # Text summaries
    text_with_summaries = []
    for i, block in enumerate(text_blocks):
        text_content = block['text']
        # Use GPT to create a summary
        if len(text_content.split()) > 20:  # Only summarize longer text blocks
            summary_prompt = f"Summarize the following text in 1-2 sentences:\n\n{text_content}"
            summary = llm.complete(summary_prompt).text
        else:
            summary = text_content
            
        text_with_summaries.append({
            "text": text_content,
            "summary": summary,
            "page": block.get('page', 0),
            "bbox": block.get('bbox', None),
            "type": "text",
            "id": f"text-{i}"
        })
    
    # Table summaries
    tables_with_summaries = []
    for i, table in enumerate(tables):
        # Convert table to string representation
        table_str = table.to_dataframe().to_string()
        
        # Use GPT to create a summary
        summary_prompt = f"Summarize the information in this table concisely:\n\n{table_str}"
        summary = llm.complete(summary_prompt).text
        
        tables_with_summaries.append({
            "table": table,
            "table_str": table_str,
            "summary": summary,
            "page": table.page,
            "bbox": table.bbox,
            "type": "table",
            "id": f"table-{i}"
        })
    
    return text_with_summaries, tables_with_summaries

def create_llama_nodes(text_blocks: List[Dict[str, Any]], 
                      tables: List[Dict[str, Any]]) -> List[TextNode]:
    """
    Convert Docling text blocks and tables to LlamaIndex nodes
    """
    print("Creating LlamaIndex nodes...")
    
    nodes = []
    
    # Create nodes for text blocks
    for block in text_blocks:
        # Create a TextNode
        node = TextNode(
            text=block['text'],
            metadata={
                "page": block.get('page', 0),
                "bbox": str(block.get('bbox', "")),
                "type": "text",
                "id": block.get('id', ""),
                "summary": block.get('summary', "")
            }
        )
        nodes.append(node)
    
    # Create nodes for tables
    for table_info in tables:
        # Create a TextNode for the table
        node = TextNode(
            text=table_info['table_str'],
            metadata={
                "page": table_info.get('page', 0),
                "bbox": str(table_info.get('bbox', "")),
                "type": "table",
                "id": table_info.get('id', ""),
                "summary": table_info.get('summary', "")
            }
        )
        nodes.append(node)
    
    return nodes

def build_vector_store_index(nodes: List[TextNode]) -> VectorStoreIndex:
    """
    Build a VectorStoreIndex from the nodes
    """
    print("Building VectorStoreIndex...")
    
    # Create the index
    index = VectorStoreIndex(nodes, service_context=service_context)
    return index

def build_multi_vector_retriever(index: VectorStoreIndex, nodes: List[TextNode]) -> MultiVectorRetriever:
    """
    Build a MultiVectorRetriever that retrieves based on both content and summaries
    """
    print("Building MultiVectorRetriever...")
    
    # Create a document store to store mapping between summary and original content
    docstore = SimpleDocumentStore()
    
    # Create a separate vector index for the summaries
    summary_nodes = []
    
    for node in nodes:
        # Get the summary from the metadata
        summary = node.metadata.get("summary", "")
        
        # Create a node with the summary as the content
        summary_node = TextNode(
            text=summary,
            metadata={
                "doc_id": node.doc_id,  # Link to the original node
                "page": node.metadata.get("page", 0),
                "type": node.metadata.get("type", ""),
                "id": node.metadata.get("id", "")
            }
        )
        
        # Add to summary nodes
        summary_nodes.append(summary_node)
        
        # Add original node to docstore
        docstore.add_documents([node])
    
    # Create a vector index for summaries
    summary_index = VectorStoreIndex(summary_nodes, service_context=service_context)
    
    # Create the multi-vector retriever
    retriever = MultiVectorRetriever(
        vector_retriever=summary_index.as_retriever(),
        docstore=docstore,
        # This is the ID of the child node (summary) to the parent node (original)
        id_to_doc_id_fn=lambda doc_id: doc_id
    )
    
    return retriever

def run_pipeline(pdf_path: str) -> MultiVectorRetriever:
    """
    Run the complete pipeline from PDF to MultiVectorRetriever
    """
    # Step 1: Load PDF
    doc = load_pdf(pdf_path)
    
    # Step 2-3: Extract text and tables
    text_blocks, tables = extract_text_and_tables(doc)
    
    # Step 4-5: Generate summaries for text and tables
    text_with_summaries, tables_with_summaries = generate_summaries(text_blocks, tables)
    
    # Step 6: Create LlamaIndex nodes preserving metadata
    nodes = create_llama_nodes(text_with_summaries, tables_with_summaries)
    
    # Step 7: Build the VectorStoreIndex
    index = build_vector_store_index(nodes)
    
    # Step 8: Build MultiVectorRetriever
    retriever = build_multi_vector_retriever(index, nodes)
    
    print(f"Pipeline complete! Created a retriever with {len(nodes)} nodes")
    return retriever

def query_system(retriever: MultiVectorRetriever, query: str, top_k: int = 3) -> List[TextNode]:
    """
    Query the system with a question
    """
    print(f"Querying system with: {query}")
    
    # Get top k results
    retrieval_results = retriever.retrieve(query, similarity_top_k=top_k)
    
    print(f"Found {len(retrieval_results)} relevant documents")
    
    return retrieval_results

# Demo usage
if __name__ == "__main__":
    pdf_path = "example_document.pdf"
    
    # Run the pipeline
    retriever = run_pipeline(pdf_path)
    
    # Example query
    query = "What information is in the tables on page 3?"
    results = query_system(retriever, query)
    
    # Display results
    for i, node in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"Type: {node.metadata.get('type')}")
        print(f"Page: {node.metadata.get('page')}")
        print(f"Summary: {node.metadata.get('summary')}")
        print(f"Content: {node.text[:150]}...")
