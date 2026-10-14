import sqlite3

import chromadb
import ollama
from fastmcp import FastMCP


mcp = FastMCP("Customer Support MCP Server")


# Connect to the existing ChromaDB collection
chroma_client = chromadb.PersistentClient(path="./chroma_db")

collection = chroma_client.get_collection(
    name="support_knowledge"
)


def get_connection():
    conn = sqlite3.connect("tickets.db")
    conn.row_factory = sqlite3.Row
    return conn


@mcp.tool()
def search_knowledge_base(query: str) -> dict:
    """Search troubleshooting documentation for relevant information."""

    if not query.strip():
        return {"error": "Query cannot be empty."}

    response = ollama.embed(
        model="nomic-embed-text",
        input=query
    )

    query_embedding = response["embeddings"][0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(3, collection.count())
    )

    documents = results["documents"][0] if results["documents"] else []

    return {
        "query": query,
        "results": documents
    }


@mcp.tool()
def get_ticket_status(ticket_id: str, user_id: str) -> dict:
    """Retrieve a ticket belonging to the specified user."""

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
        return {"error": "Ticket not found."}

    return dict(ticket)


@mcp.tool()
def create_support_ticket(
    user_id: str,
    subject: str,
    description: str
) -> dict:
    """Create a support ticket for a user."""

    subject = subject.strip()
    description = description.strip()

    if not user_id.strip():
        return {"error": "User ID cannot be empty."}

    if len(subject) < 3 or len(subject) > 200:
        return {"error": "Subject must contain 3-200 characters."}

    if len(description) < 5 or len(description) > 2000:
        return {"error": "Description must contain 5-2000 characters."}

    # Reuse the ticket creation API's existing validation rules.
    # The database is the same persistent SQLite database.

    import uuid
    from datetime import datetime, timezone

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
                user_id,
                subject,
                description,
                "open",
                created_at
            )
        )

    return {
        "ticket_id": ticket_id,
        "status": "open",
        "message": "Support ticket created successfully."
    }


if __name__ == "__main__":
    mcp.run()