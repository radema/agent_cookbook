from llama_index.core.response_synthesizers import CompactAndRefine
from llama_index.core.postprocessor.llm_rerank import LLMRerank
from llama_index.core.workflow import (
    Context,
    Workflow,
    StartEvent,
    StopEvent,
    step,
)

from llama_index.core.workflow import Event
from llama_index.core.schema import NodeWithScore

class RetrieverEvent(Event):
    """Result of running retrieval"""

    nodes: list[NodeWithScore]


class RerankEvent(Event):
    """Result of running reranking on retrieved nodes"""

    nodes: list[NodeWithScore]

class RAGWorkflow(Workflow):
    @step
    async def retrieve(
        self, ctx: Context, ev: StartEvent,
    ) -> RetrieverEvent | None:
        "Entry point for RAG, triggered by a StartEvent with `query`."
        query = ev.get("query")
        index = ev.get("index")

        if not query:
            return None

        print(f"Query the database with: {query}")

        # store the query in the global context
        await ctx.set("query", query)

        # get the index from the global context
        if index is None:
            print("Index is empty, load some documents before querying!")
            return None

        retriever = index.as_retriever(similarity_top_k=5, response_mode="tree_summarize")
        nodes = await retriever.aretrieve(query)
        print(f"Retrieved {len(nodes)} nodes.")
        return RetrieverEvent(nodes=nodes)
    @step
    async def rerank(self, ctx: Context, ev: RetrieverEvent) -> RerankEvent:
        # Rerank the nodes
        processor = ctx.get("processor")
        ranker = LLMRerank(
            choice_batch_size=5, top_n=3, llm=processor.summarizer
        )
        print(await ctx.get("query", default=None), flush=True)
        new_nodes = ranker.postprocess_nodes(
            ev.nodes, query_str=await ctx.get("query", default=None)
        )
        print(f"Reranked nodes to {len(new_nodes)}")
        return RerankEvent(nodes=new_nodes)
    
    @step
    async def synthesize(self, ctx: Context, ev: RerankEvent) -> StopEvent:
        """Return a streaming response using reranked nodes."""
        processor = ctx.get("processor")
        llm = processor.summarizer
        summarizer = CompactAndRefine(llm=llm, streaming=True, verbose=True)
        query = await ctx.get("query", default=None)
        response = await summarizer.asynthesize(query, nodes=ev.nodes)

        return StopEvent(result=response)