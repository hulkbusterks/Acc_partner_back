from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional, List
from server.db import get_session
from server.models import User
import bcrypt
from server.utils.token import create_token, verify_token
from sqlmodel import select
import re

router = APIRouter(prefix="/auth")

_bearer_scheme = HTTPBearer()

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer_scheme),
):
    """Shared authentication dependency. Use via Depends(get_current_user)."""
    token = credentials.credentials
    payload = verify_token(token)
    if not payload or "user_id" not in payload:
        raise HTTPException(status_code=401, detail="Invalid token")
    with get_session() as db:
        user = db.get(User, str(payload["user_id"]))
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        db.expunge(user)
        return user


class RegisterIn(BaseModel):
    email: str
    password: str
    display_name: str | None = None


class RegisterOut(BaseModel):
    user_id: str


class LoginIn(BaseModel):
    email: str
    password: str


class LoginOut(BaseModel):
    token: str
    user_id: str


@router.post("/register", response_model=RegisterOut)
def register(data: RegisterIn):
    if not _EMAIL_RE.match(data.email):
        raise HTTPException(status_code=400, detail="Invalid email format")
    if len(data.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")
    with get_session() as db:
        existing = db.exec(select(User).where(User.email == data.email)).first()
        if existing:
            raise HTTPException(status_code=400, detail="Email already registered")
        hashed = bcrypt.hashpw(data.password.encode(), bcrypt.gensalt())
        user = User(email=data.email, display_name=data.display_name, password_hash=hashed.decode())
        db.add(user)
        db.commit()
        db.refresh(user)
        return {"user_id": user.id}


@router.post("/login", response_model=LoginOut)
def login(data: LoginIn):
    with get_session() as db:
        user = db.exec(select(User).where(User.email == data.email)).first()
        if not user:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        if not bcrypt.checkpw(data.password.encode(), user.password_hash.encode()):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        token = create_token({"user_id": user.id, "email": user.email})
        return {"token": token, "user_id": user.id}


class UserOut(BaseModel):
    id: str
    email: str
    display_name: Optional[str] = None
    created_at: Optional[str] = None


@router.get("/users", response_model=List[UserOut], summary="List users", description="List all registered users (id, email, display_name). Requires authentication.")
def list_users(limit: int = 50, current_user=Depends(get_current_user)):
    with get_session() as db:
        q = select(User).limit(limit)
        users = db.exec(q).all()
        result = []
        for u in users:
            result.append(UserOut(
                id=u.id,
                email=u.email,
                display_name=u.display_name,
                created_at=str(u.created_at) if u.created_at else None,
            ))
        return result


@router.get("/users/me", response_model=UserOut, summary="Get current user", description="Get the authenticated user's own profile.")
def get_me(current_user=Depends(get_current_user)):
    return UserOut(
        id=current_user.id,
        email=current_user.email,
        display_name=current_user.display_name,
        created_at=str(current_user.created_at) if current_user.created_at else None,
    )


@router.get("/users/{user_id}", response_model=UserOut, summary="Get user by ID", description="Get a specific user's profile by their ID. Requires authentication.")
def get_user_by_id(user_id: str, current_user=Depends(get_current_user)):
    with get_session() as db:
        user = db.get(User, user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        return UserOut(
            id=user.id,
            email=user.email,
            display_name=user.display_name,
            created_at=str(user.created_at) if user.created_at else None,
        )
