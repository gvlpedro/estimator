from fastapi import APIRouter, Depends, HTTPException
from pydantic import ValidationError

from app.dependencies import get_ai_service_client
from app.schemas.search import SearchRequest, SearchResponse
from app.services.ai_client import AIServiceClient, AIServiceError

router = APIRouter(prefix="/api/v1", tags=["search"])


@router.post("/search", response_model=SearchResponse)
async def semantic_search(
    request: SearchRequest,
    client: AIServiceClient = Depends(get_ai_service_client),
) -> SearchResponse:
    """Return the k historical budget chunks most similar to the query.

    Delegates to the AI service's ``POST /search``; a 4xx from the AI
    service is forwarded as-is, anything else (unreachable, 5xx) maps to 502.
    """
    try:
        raw = await client.search(query=request.query, k=request.k)
    except AIServiceError as exc:
        if exc.status_code is not None and 400 <= exc.status_code < 500:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
        raise HTTPException(status_code=502, detail="AI service call failed") from exc
    try:
        return SearchResponse.model_validate(raw)
    except ValidationError as exc:
        raise HTTPException(
            status_code=502, detail="AI service response did not match contract"
        ) from exc
