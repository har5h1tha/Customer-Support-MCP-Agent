import sqlite3
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


DB_PATH = "tickets.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_database():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tickets (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                subject TEXT NOT NULL,
                description TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)


@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_database()
    yield


app = FastAPI(
    title="Customer Support Ticket API",
    lifespan=lifespan
)


class TicketCreate(BaseModel):
    user_id: str = Field(min_length=1, max_length=100)
    subject: str = Field(min_length=3, max_length=200)
    description: str = Field(min_length=5, max_length=2000)


@app.post("/tickets", status_code=201)
def create_ticket(ticket: TicketCreate):
    ticket_id = f"TKT-{uuid.uuid4().hex[:8].upper()}"

    created_at = datetime.now(timezone.utc).isoformat()

    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO tickets
            (id, user_id, subject, description, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                ticket_id,
                ticket.user_id,
                ticket.subject,
                ticket.description,
                "open",
                created_at
            )
        )

    return {
        "ticket_id": ticket_id,
        "status": "open",
        "message": "Support ticket created successfully."
    }


@app.get("/tickets/{ticket_id}")
def get_ticket(ticket_id: str, user_id: str):
    with get_connection() as conn:
        ticket = conn.execute(
            """
            SELECT id, subject, description, status, created_at
            FROM tickets
            WHERE id = ? AND user_id = ?
            """,
            (ticket_id, user_id)
        ).fetchone()

    if ticket is None:
        # Do not disclose whether another user's ticket exists.
        raise HTTPException(
            status_code=404,
            detail="Ticket not found."
        )

    return dict(ticket)