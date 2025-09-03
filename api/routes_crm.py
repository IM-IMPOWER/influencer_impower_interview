from fastapi import APIRouter, HTTPException, Query
from datetime import datetime
from pydantic import BaseModel, Field
from typing import List, Optional
import os, psycopg
from psycopg.rows import dict_row

router = APIRouter(prefix="/api", tags=["crm"])

def get_conn():
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        raise RuntimeError("DATABASE_URL not set")
    return psycopg.connect(dsn, row_factory=dict_row)

# ---------- Schemas ----------

class ConversationCreate(BaseModel):
    kol_id: int
    channel: str = Field(default="dm", pattern="^(dm|email|line)$")

class ConversationOut(BaseModel):
    id: int
    kol_id: int
    status: str
    channel: str
    last_message_at: Optional[datetime] = None
    created_at: datetime
    # denormalized KOL info (nice for UI)
    kol_username: Optional[str] = None
    kol_display_name: Optional[str] = None
    kol_followers: Optional[int] = None
    proposed_deliverables: Optional[str] = None
    proposed_timeline: Optional[str] = None
    proposed_price_integer: Optional[int] = None
    agreed_deliverables: Optional[str] = None
    agreed_timeline: Optional[str] = None
    agreed_price_integer: Optional[int] = None

class ConversationUpdate(BaseModel):
    status: Optional[str] = Field(default=None, pattern="^(contacted|negotiating|confirmed|closed)$")
    # proposals (while negotiating)
    proposed_deliverables: Optional[str] = None
    proposed_timeline: Optional[str] = None
    proposed_price_integer: Optional[int] = None
    # final agreement (on confirmed)
    agreed_deliverables: Optional[str] = None
    agreed_timeline: Optional[str] = None
    agreed_price_integer: Optional[int] = None


class MessageCreate(BaseModel):
    direction: str = Field(pattern="^(out|in)$")
    body: str = Field(min_length=1)

class MessageOut(BaseModel):
    id: int
    conversation_id: int
    direction: str
    body: str
    created_at: datetime

class ConversationWithMessages(BaseModel):
    conversation: ConversationOut
    messages: List[MessageOut]

# ---------- Routes ----------

@router.post("/conversations", response_model=ConversationOut)
@router.post("/conversations", response_model=ConversationOut)
def create_conversation(payload: ConversationCreate):
    with get_conn() as conn, conn.cursor() as cur:
        # ensure KOL exists
        cur.execute("SELECT id, username, display_name, followers FROM kols WHERE id=%s", (payload.kol_id,))
        k = cur.fetchone()
        if not k:
            raise HTTPException(404, "KOL not found")

        # get-or-create (one conversation per KOL)
        cur.execute("""
            INSERT INTO conversations (kol_id, channel)
            VALUES (%s, %s)
            ON CONFLICT (kol_id)
            DO UPDATE SET channel = EXCLUDED.channel  -- or DO NOTHING if you don't want to change channel
            RETURNING id, kol_id, status, channel, last_message_at, created_at;
        """, (payload.kol_id, payload.channel))
        c = cur.fetchone()

        return {
            **c,
            "kol_username": k["username"],
            "kol_display_name": k["display_name"],
            "kol_followers": k["followers"],
        }


@router.get("/conversations", response_model=List[ConversationOut])
def list_conversations(
    status: Optional[str] = Query(None, pattern="^(contacted|negotiating|confirmed|closed)$"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    wh, args = [], {}
    if status:
        wh.append("c.status = %(status)s")
        args["status"] = status
    where = ("WHERE " + " AND ".join(wh)) if wh else ""
    args.update({"limit": limit, "offset": offset})

    sql = f"""
      SELECT
        c.id, c.kol_id, c.status, c.channel, c.last_message_at, c.created_at,
        k.username AS kol_username, k.display_name AS kol_display_name, k.followers AS kol_followers
      FROM conversations c
      JOIN kols k ON k.id = c.kol_id
      {where}
      ORDER BY COALESCE(c.last_message_at, c.created_at) DESC
      LIMIT %(limit)s OFFSET %(offset)s
    """
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, args)
        rows = cur.fetchall()
    return rows

@router.get("/conversations/{conv_id}", response_model=ConversationWithMessages)
def get_conversation(conv_id: int):
    with get_conn() as conn, conn.cursor() as cur:
        # cur.execute("""
        #   SELECT c.id, c.kol_id, c.status, c.channel, c.last_message_at, c.created_at,
        #          k.username AS kol_username, k.display_name AS kol_display_name, k.followers AS kol_followers
        #   FROM conversations c
        #   JOIN kols k ON k.id = c.kol_id
        #   WHERE c.id = %s
        # """, (conv_id,))
        cur.execute("""
            SELECT c.id, c.kol_id, c.status, c.channel, c.last_message_at, c.created_at,
                c.proposed_deliverables, c.proposed_timeline, c.proposed_price_integer,
                c.agreed_deliverables,  c.agreed_timeline,  c.agreed_price_integer,
                k.username AS kol_username, k.display_name AS kol_display_name, k.followers AS kol_followers
            FROM conversations c
            JOIN kols k ON k.id = c.kol_id
            WHERE c.id = %s
            """, (conv_id,))
        c = cur.fetchone()
        if not c:
            raise HTTPException(404, "Conversation not found")

        cur.execute("""
          SELECT id, conversation_id, direction, body, created_at
          FROM messages
          WHERE conversation_id = %s
          ORDER BY created_at ASC, id ASC
        """, (conv_id,))
        msgs = cur.fetchall()

    return {"conversation": c, "messages": msgs}

# @router.patch("/conversations/{conv_id}", response_model=ConversationOut)
# def update_conversation(conv_id: int, payload: ConversationUpdate):
#     with get_conn() as conn, conn.cursor() as cur:
#         cur.execute("UPDATE conversations SET status=%s WHERE id=%s RETURNING kol_id, status, channel, last_message_at, created_at",
#                     (payload.status, conv_id))
#         row = cur.fetchone()
#         if not row:
#             raise HTTPException(404, "Conversation not found")
#         cur.execute("SELECT username, display_name, followers FROM kols WHERE id=%s", (row["kol_id"],))
#         k = cur.fetchone()
#         return {
#             "id": conv_id,
#             **row,
#             "kol_username": k["username"],
#             "kol_display_name": k["display_name"],
#             "kol_followers": k["followers"],
#         }

# routes_crm.py (add to models)

# inside update_conversation()
@router.patch("/conversations/{conv_id}", response_model=ConversationOut)
def update_conversation(conv_id: int, payload: ConversationUpdate):
    sets, args = [], {"id": conv_id}

    if payload.status is not None:
        sets.append("status = %(status)s"); args["status"] = payload.status

    for col in ("proposed_deliverables","proposed_timeline","proposed_price_integer",
                "agreed_deliverables","agreed_timeline","agreed_price_integer"):
        val = getattr(payload, col)
        if val is not None:
            sets.append(f"{col} = %({col})s"); args[col] = val

    if not sets:
        raise HTTPException(400, "No fields to update")

    sql = f"""
      UPDATE conversations
      SET {", ".join(sets)}
      WHERE id = %(id)s
      RETURNING id, kol_id, status, channel, last_message_at, created_at,
                proposed_deliverables, proposed_timeline, proposed_price_integer,
                agreed_deliverables, agreed_timeline, agreed_price_integer
    """
    with get_conn() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, args)
        row = cur.fetchone()
        if not row: raise HTTPException(404, "Conversation not found")
        # routes_crm.py – after successful PATCH:
        if payload.proposed_deliverables or payload.proposed_timeline or payload.proposed_price_integer:
            cur.execute("""
            INSERT INTO messages (conversation_id, direction, body)
            VALUES (%s, 'out',
                'Offer proposed:\\n• Deliverables: ' || coalesce(%s,'—') ||
                '\\n• Timeline: ' || coalesce(%s,'—') ||
                '\\n• Price: THB ' || coalesce(%s::text,'—')
            )
            """, (conv_id, payload.proposed_deliverables, payload.proposed_timeline, payload.proposed_price_integer))

        if payload.agreed_deliverables or payload.agreed_timeline or payload.agreed_price_integer:
            cur.execute("""
            INSERT INTO messages (conversation_id, direction, body)
            VALUES (%s, 'out',
                'Confirmed:\\n• Deliverables: ' || coalesce(%s,'—') ||
                '\\n• Timeline: ' || coalesce(%s,'—') ||
                '\\n• Price: THB ' || coalesce(%s::text,'—')
            )
            """, (conv_id, payload.agreed_deliverables, payload.agreed_timeline, payload.agreed_price_integer))

        cur.execute("SELECT username, display_name, followers FROM kols WHERE id=%s", (row["kol_id"],))
        k = cur.fetchone()
        return {
            **row,
            "kol_username": k["username"],
            "kol_display_name": k["display_name"],
            "kol_followers": k["followers"],
        }


@router.post("/conversations/{conv_id}/messages", response_model=MessageOut)
def add_message(conv_id: int, payload: MessageCreate):
    with get_conn() as conn, conn.cursor() as cur:
        # ensure conversation exists
        cur.execute("SELECT 1 FROM conversations WHERE id=%s", (conv_id,))
        if not cur.fetchone():
            raise HTTPException(404, "Conversation not found")

        cur.execute("""
          INSERT INTO messages (conversation_id, direction, body)
          VALUES (%s, %s, %s)
          RETURNING id, conversation_id, direction, body, created_at
        """, (conv_id, payload.direction, payload.body))
        m = cur.fetchone()

        # update last_message_at
        cur.execute("UPDATE conversations SET last_message_at = %s WHERE id = %s",
                    (m["created_at"], conv_id))

    return m
