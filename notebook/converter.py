from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
    TableFormerMode
)
from docling_core.types.doc.document import TableItem

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from sentence_transformers import SentenceTransformer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

from llama_index.core.schema import IndexNode, TextNode, NodeRelationship, RelatedNodeInfo
from docling_core.types.doc.labels import DocItemLabel

from typing import List, Dict, Any, Union
from tqdm import tqdm

class DocumentProcessor:
    def __init__(self, **kwargs):

        self.setup_converter()
        self.setup_models(**kwargs)

    def setup_converter(self):
        "Setup the document converter with predefined options."
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.do_cell_matching = True
        pipeline_options.ocr_options.lang = ["en","it"]
        pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
        pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=8, device=AcceleratorDevice.MPS
            )

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options
                )
            }
        )

    def setup_models(self, **kwargs):
        "Initialize models for text processing."
        self.embed_model = HuggingFaceEmbedding(kwargs.get('embed_model','sentence-transformers/all-MiniLM-L6-v2'))
        self.chunker = HybridChunker(
            tokenizer=kwargs.get('tokenizer_model',"jinaai/jina-embeddings-v3")
            )
        self.summarizer = Ollama(model=kwargs.get('summarizer_model',"gemma3:4b"), request_timeout=60.0, temperature=0.01)
        

    def convert(self, sources: Union[str, List[str]]) -> List[Dict[str, Any]]:
        """
        Convert documents to a structured format.
        Args:
            sources (Union[str, List[str]]): Path(s) to the document(s).
        Returns:
            List[Dict[str, Any]]: Converted documents.
        """
        if isinstance(sources, str):
            sources = [sources]
        
        # Process each source document
        results = {}
        for source in tqdm(sources):
            results[source] = self.converter.convert(source).document
        
        return results


    def extract_chunk_metadata(self, chunk) -> Dict[str, Any]:
        """Extract essential metadata from a chunk"""
        metadata = {
            "text": chunk.text,
            "headings": [],
            "page_info": None,
            "content_type": None
        }
        
        if hasattr(chunk, 'meta'):
            # Extract headings
            if hasattr(chunk.meta, 'headings') and chunk.meta.headings:
                metadata["headings"] = chunk.meta.headings
            
            # Extract page information and content type
            if hasattr(chunk.meta, 'doc_items'):
                for item in chunk.meta.doc_items:
                    if hasattr(item, 'label'):
                        metadata["content_type"] = str(item.label)
                    
                    if hasattr(item, 'prov') and item.prov:
                        for prov in item.prov:
                            if hasattr(prov, 'page_no'):
                                metadata["page_info"] = prov.page_no
        
        return metadata

    def chunk_documents(self, conversions: Dict[str, Any]) -> List[TextNode]:
        doc_id = 0
        node_id = 0
        nodes = []
        node_mapping = {}

        for source, docling_document in conversions.items():

            tables = {table.get_ref().cref: table for table in docling_document.tables}

            for chunk in self.chunker.chunk(docling_document):
        
                items = chunk.meta.doc_items
                if len(items) == 1 and isinstance(items[0], TableItem):

                    continue # we will process tables later

                refs = " ".join(map(lambda item: item.get_ref().cref, items))
                table_list = [
                    item.get_ref().cref for item in items if item.label in [DocItemLabel.TABLE]
                ]

                text = chunk.text
                chunk_metadata = self.extract_chunk_metadata(chunk)

                node = TextNode(
                    id_='text_'+str(node_id+1),
                    text=text,
                    #embedding = embed_model.encode(text),
                    metadata ={
                        'source':source,
                        'ref':refs,
                        'tables': table_list,
                        "page_info": chunk_metadata["page_info"],
                        "content_type": chunk_metadata["content_type"],
                        "headings": chunk_metadata["headings"],
                        "doc_id": doc_id
                    }
                )

                node_id += 1
                
                # Add relationships
                if len(nodes)>0:
                    if nodes[-1].metadata['source'] == source:
                        node.relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(node_id=nodes[-1].id_)
                        nodes[-1].relationships[NodeRelationship.NEXT] = RelatedNodeInfo(node_id=node.id_)
                nodes.append(node)

                if len(table_list) > 0:
                    for table_id in table_list:
                        if table_id in tables.keys():
                            table = tables[table_id]
                            ref = table.get_ref().cref
                            # create node with tables
                            summary = self.summarizer.complete(f'''
                            You are a summarizer model. Summarize the following table considering that is extracted from a document related to an invoice.\nTable {table.export_to_markdown()}.\n\nSummary:
                            '''
                            )
                            node_table = IndexNode(
                                index_id='table_'+str(node_id+1),
                                #obj=table.export_to_dataframe(),
                                text=summary.text,
                                metadata ={
                                    'source':source,
                                    'ref':ref,
                                    "page_info": chunk_metadata["page_info"],
                                    "content_type": 'TABLE',
                                    "headings": chunk_metadata["headings"],
                                    "doc_id": doc_id,
                                    #"data": table.export_to_dataframe()
                                }
                            )
                            node_mapping['table_'+str(node_id+1)] = table.export_to_dataframe()
                            node_id += 1

                            node.relationships[NodeRelationship.CHILD] = RelatedNodeInfo(node_id=node_table.index_id)
                            node_table.relationships[NodeRelationship.PARENT] = RelatedNodeInfo(node_id=node.id_)

                            nodes.append(node_table)

            doc_id += 1
        return nodes, node_mapping

    def add_embeddings(self, nodes: List[TextNode], method: str = None) -> List[TextNode]:
        """
        Add embeddings to the nodes.
        Args:
            nodes (List[TextNode]): List of nodes to embed.
        Returns:
            List[TextNode]: Nodes with embeddings.
        """
        for node in tqdm(nodes):
            content = node.get_content(
                metadata_mode = (None if node.metadata['content_type']=='TABLE' else "embed")
                )
            if (method == 'summary')&(node.metadata['content_type']!='TABLE'):
                
                summary = self.summarizer.complete(f'''
                You are a summarizer model. Summarize the following document chunk, with related metadata, considering that is a document related to an invoice.
                Optimize the summary considering that it has to be used for embedding. Do not include your reasoning or fake information. Preserve original language and format.
                \nChunk with metadata: {content}\n\nSummary:
                ''')
            embed_text = '\nHeading:'+' '.join(node.metadata['headings'])+\
            f'\n\nOriginal Text:\n{content}\n\n' + ('' if method != 'summary' else f"Summary:\n{summary.text}\n\n")
            node.embedding = self.embed_model.get_text_embedding(embed_text)
                
        return nodes