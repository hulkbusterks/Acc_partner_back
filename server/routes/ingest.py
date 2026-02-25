from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Header
from pydantic import BaseModel
from typing import Optional, List
from server.services.ingest_manager import ingest_manager
from server.services.file_extract import extract_text_from_file
from server.auth import get_current_user
from server.db import get_session
from server.models import User, Book, Topic
from sqlmodel import select

router = APIRouter(prefix="/ingest")

MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB


class BookIn(BaseModel):
    title: str
    authors: Optional[str] = None
    raw_text: Optional[str] = None


class IngestBookOut(BaseModel):
    book_id: str


class IngestFileOut(BaseModel):
    book_id: str
    chars: int


class TopicSummary(BaseModel):
    id: str
    title: str


class CreateTopicsOut(BaseModel):
    created: int
    topics: List[TopicSummary]


@router.post("/book", response_model=IngestBookOut)
def ingest_book(data: BookIn, current_user=Depends(get_current_user)):
    b = ingest_manager.create_book(data.title, data.authors, data.raw_text)
    return {"book_id": b.id}


@router.post("/file", response_model=IngestFileOut)
async def ingest_file(
    title: str,
    file: UploadFile = File(...),
    authors: Optional[str] = None,
    current_user=Depends(get_current_user),
):
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large (max {MAX_UPLOAD_BYTES // (1024*1024)} MB)")

    try:
        text = extract_text_from_file(file.filename or "", data, file.content_type)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if not text:
        raise HTTPException(status_code=400, detail="Could not extract readable text from uploaded file")

    b = ingest_manager.create_book(title=title, authors=authors, raw_text=text)
    return {"book_id": b.id, "chars": len(text)}


@router.post("/books/{book_id}/topics", response_model=CreateTopicsOut)
def create_topics(book_id: str, mode: str = "rag", current_user=Depends(get_current_user)):
    topics = ingest_manager.create_topics_from_book(book_id, mode=mode)
    return {"created": len(topics), "topics": [{"id": t.id, "title": t.title} for t in topics]}


class BookOut(BaseModel):
    id: str
    title: str
    authors: Optional[str] = None


class TopicOut(BaseModel):
    id: str
    book_id: str
    title: str
    start_page: Optional[int] = None
    end_page: Optional[int] = None


@router.get("/books", response_model=List[BookOut], summary="List books", description="List all ingested books.")
def list_books(limit: int = 50, current_user=Depends(get_current_user)):
    with get_session() as db:
        q = select(Book).limit(limit)
        books = db.exec(q).all()
        result = []
        for b in books:
            result.append(BookOut(id=b.id, title=b.title, authors=b.authors))
        return result


@router.get("/books/{book_id}/topics", response_model=List[TopicOut], summary="List topics for a book", description="List all topics generated for a given book.")
def list_topics(book_id: str, current_user=Depends(get_current_user)):
    with get_session() as db:
        q = select(Topic).where(Topic.book_id == book_id)
        topics = db.exec(q).all()
        result = []
        for t in topics:
            result.append(TopicOut(id=t.id, book_id=t.book_id, title=t.title, start_page=t.start_page, end_page=t.end_page))
        return result
