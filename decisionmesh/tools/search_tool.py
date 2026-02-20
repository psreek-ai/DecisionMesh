"""
DecisionMesh — Search Tool
Tavily web search wrapper. Only used when web_monitoring_enabled=True per decision.
"""
from typing import Optional
from pydantic import BaseModel, Field


class SearchInput(BaseModel):
    query: str = Field(description="The search query to find information about a premise or condition.")
    max_results: int = Field(default=5, ge=1, le=10, description="Maximum number of results to return.")
    search_depth: str = Field(
        default="basic",
        description="Search depth: 'basic' for quick results, 'advanced' for deeper research.",
    )


class SearchResult(BaseModel):
    title: str
    url: str
    content: str
    score: float = 0.0
    published_date: Optional[str] = None


SEARCH_TOOL: dict = {
    "name": "web_search",
    "description": (
        "Search the web for information relevant to a decision premise. "
        "ONLY use this tool when web_monitoring_enabled=True for the current decision. "
        "Always cite the returned source URLs in your observations. "
        "Never fabricate search results."
    ),
    "input_schema": SearchInput.model_json_schema(),
}


async def execute_web_search(query: str, max_results: int = 5, search_depth: str = "basic") -> list[SearchResult]:
    """Execute a Tavily web search and return structured results."""
    try:
        from tavily import AsyncTavilyClient
        from decisionmesh.config import settings

        if not settings.TAVILY_API_KEY:
            return [SearchResult(
                title="Web search unavailable",
                url="",
                content="TAVILY_API_KEY is not configured. Set it in your .env file to enable web monitoring.",
                score=0.0,
            )]

        client = AsyncTavilyClient(api_key=settings.TAVILY_API_KEY)
        response = await client.search(
            query=query,
            max_results=max_results,
            search_depth=search_depth,
        )

        results = []
        for r in response.get("results", []):
            results.append(SearchResult(
                title=r.get("title", ""),
                url=r.get("url", ""),
                content=r.get("content", ""),
                score=r.get("score", 0.0),
                published_date=r.get("published_date"),
            ))
        return results

    except ImportError:
        return [SearchResult(
            title="Tavily not installed",
            url="",
            content="Install tavily-python to enable web search: pip install tavily-python",
            score=0.0,
        )]
    except Exception as e:
        return [SearchResult(
            title="Search error",
            url="",
            content=f"Search failed: {str(e)}",
            score=0.0,
        )]
