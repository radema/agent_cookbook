import os
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

# PDF processing and extraction libraries
from docling.document import Document
from docling.extractors import TextExtractor, TableExtractor
from docling.models import Table

# LlamaIndex components
from llama_index import VectorStoreIndex, ServiceContext
from llama_index.node_parser import SimpleNodeParser
from llama_index.schema import TextNode, NodeRelationship, RelatedNodeInfo, MetadataMode
from llama_index.embeddings import OpenAIEmbedding
from llama_index.llms import OpenAI
from llama_index.ingestion import IngestionPipeline
from llama_index.retrievers import RecursiveRetriever
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

def collapse_small_text_chunks(text_blocks: List[Dict[str, Any]], 
                              min_words: int = 20,
                              max_distance: float = 100) -> List[Dict[str, Any]]:
    """
    Collapse small adjacent text blocks that are close to each other
    
    Args:
        text_blocks: List of text blocks from extraction
        min_words: Minimum number of words to consider a block as "small"
        max_distance: Maximum distance (in pixels) to consider blocks as "close"
        
    Returns:
        List of text blocks with small blocks merged
    """
    print("Collapsing small text chunks...")
    
    if not text_blocks:
        return []
    
    # Sort text blocks by page and then by y-coordinate (top to bottom)
    sorted_blocks = sorted(text_blocks, 
                          key=lambda b: (b.get('page', 0), 
                                        b.get('bbox', [0, 0, 0, 0])[1]))
    
    collapsed_blocks = []
    current_block = sorted_blocks[0].copy()
    current_text = current_block['text']
    
    for next_block in sorted_blocks[1:]:
        next_text = next_block['text']
        
        # Check if both blocks are on the same page
        same_page = current_block.get('page', 0) == next_block.get('page', 0)
        
        # Calculate distance between blocks if they have bboxes
        distance = float('inf')
        if same_page and current_block.get('bbox') and next_block.get('bbox'):
            # Distance between bottom of current and top of next
            curr_bbox = current_block['bbox']
            next_bbox = next_block['bbox']
            distance = next_bbox[1] - curr_bbox[3]
        
        # Check if current block is small or next block is small and they're close
        current_is_small = len(current_text.split()) < min_words
        next_is_small = len(next_text.split()) < min_words
        
        if same_page and (current_is_small or next_is_small) and distance <= max_distance:
            # Merge blocks
            current_text = current_text + " " + next_text
            current_block['text'] = current_text
            
            # Update bounding box to cover both blocks
            if current_block.get('bbox') and next_block.get('bbox'):
                x0 = min(current_block['bbox'][0], next_block['bbox'][0])
                y0in = min(current_block['bbox'][1], next_block['bbox'][1])
                x1 = max(current_block['bbox'][2], next_block['bbox'][2])
                y1 = max(current_block['bbox'][3], next_block['bbox'][3])
                current_block['bbox'] = [x0, y0in, x1, y1]
        else:
            # Add current block to results and start a new one
            collapsed_blocks.append(current_block)
            current_block = next_block.copy()
            current_text = current_block['text']
    
    # Add the last block
    collapsed_blocks.append(current_block)
    
    print(f"Collapsed {len(text_blocks) - len(collapsed_blocks)} blocks")
    return collapsed_blocks

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

def build_recursive_retriever(nodes: List[TextNode]) -> Tuple[RecursiveRetriever, SimpleDocumentStore, VectorStoreIndex]:
    """
    Build a RecursiveRetriever with parent-child relationships
    """
    print("Building RecursiveRetriever...")
    
    # Create document store
    docstore = SimpleDocumentStore()
    
    # Add all nodes to docstore
    docstore.add_documents(nodes)
    
    # Create summary nodes that will be indexed
    summary_nodes = []
    
    # Create relationships between nodes on the same page
    for i, node in enumerate(nodes):
        node_page = node.metadata.get("page", 0)
        node_type = node.metadata.get("type", "")
        
        # Create a summary node
        summary = node.metadata.get("summary", "")
        summary_node = TextNode(
            text=summary,
            metadata={
                "page": node_page,
                "type": node_type,
                "original_id": node.id_,
                "is_summary": True
            }
        )
        
        # Add relationship from summary to original
        summary_node.relationships[NodeRelationship.CHILD] = [
            RelatedNodeInfo(node_id=node.id_, relationship=NodeRelationship.CHILD)
        ]
        
        summary_nodes.append(summary_node)
        
        # Add spatial relationships for nodes on the same page
        # Connect text blocks that are close to each other or tables
        for j, other_node in enumerate(nodes):
            if i == j:  # Skip self
                continue
                
            other_page = other_node.metadata.get("page", 0)
            other_type = other_node.metadata.get("type", "")
            
            # Only connect nodes on same page
            if node_page == other_page:
                # Always connect text blocks to tables on same page
                if (node_type == "text" and other_type == "table") or \
                   (node_type == "table" and other_type == "text"):
                    node.relationships.setdefault(NodeRelationship.NEXT, []).append(
                        RelatedNodeInfo(node_id=other_node.id_, relationship=NodeRelationship.NEXT)
                    )
                    
                # For text blocks, connect if they're sequential
                elif node_type == "text" and other_type == "text":
                    # Parse bboxes if available
                    try:
                        node_bbox = eval(node.metadata.get("bbox", "[]"))
                        other_bbox = eval(other_node.metadata.get("bbox", "[]"))
                        
                        if node_bbox and other_bbox:
                            # Connect if other block is directly below (sequential reading)
                            if abs(node_bbox[0] - other_bbox[0]) < 50 and \
                               node_bbox[3] < other_bbox[1] and \
                               node_bbox[3] + 100 > other_bbox[1]:
                                node.relationships.setdefault(NodeRelationship.NEXT, []).append(
                                    RelatedNodeInfo(node_id=other_node.id_, relationship=NodeRelationship.NEXT)
                                )
                    except:
                        pass  # Skip if bbox parsing fails
    
    # Add summary nodes to docstore
    docstore.add_documents(summary_nodes)
    
    # Build vector index from summary nodes
    summary_index = VectorStoreIndex(summary_nodes, service_context=service_context)
    
    # Build recursive retriever
    retriever = RecursiveRetriever(
        docstore=docstore,
        # The root retriever is the vector index of summaries
        root_id="vector",
        retriever_dict={
            "vector": summary_index.as_retriever(similarity_top_k=5)
        },
        node_dict={}  # No additional retrievers needed
    )
    
    return retriever, docstore, summary_index

def run_pipeline(pdf_path: str) -> RecursiveRetriever:
    """
    Run the complete pipeline from PDF to RecursiveRetriever
    """
    # Step 1: Load PDF
    doc = load_pdf(pdf_path)
    
    # Step 2-3: Extract text and tables
    text_blocks, tables = extract_text_and_tables(doc)
    
    # New Step: Collapse small text chunks
    collapsed_text_blocks = collapse_small_text_chunks(text_blocks)
    
    # Step 4-5: Generate summaries for text and tables
    text_with_summaries, tables_with_summaries = generate_summaries(collapsed_text_blocks, tables)
    
    # Step 6: Create LlamaIndex nodes preserving metadata
    nodes = create_llama_nodes(text_with_summaries, tables_with_summaries)
    
    # Step 7-8: Build RecursiveRetriever with relationships
    retriever, docstore, summary_index = build_recursive_retriever(nodes)
    
    print(f"Pipeline complete! Created a retriever with {len(nodes)} nodes")
    return retriever

def query_system(retriever: RecursiveRetriever, query: str, top_k: int = 3) -> List[TextNode]:
    """
    Query the system with a question using the RecursiveRetriever
    """
    print(f"Querying system with: {query}")
    
    # Get retrieval results
    retrieval_results = retriever.retrieve(query)
    
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
    for i, result in enumerate(results):
        node = result.node
        print(f"\nResult {i+1}:")
        print(f"Type: {node.metadata.get('type')}")
        print(f"Page: {node.metadata.get('page')}")
        print(f"Summary: {node.metadata.get('summary', '')}")
        print(f"Content: {node.text[:150]}...")
