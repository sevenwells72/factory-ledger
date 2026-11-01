"""Dashboard Notes / To-Dos / Reminders: the five /dashboard/api/notes routes.

Moved unchanged from main.py (modularization PR1). main.py registers these
routes at the spot the section used to occupy, before the /dashboard static
mount, so route order is unchanged.

Auth and attribution live in main and are reached through `api` at call time:
every route needs a key (`api.verify_api_key`, Notes are on
DASHBOARD_KEY_ALLOWLIST) and every write calls `api._record_actor_write`, which
attributes named people and does nothing for the shared keys. See
docs/security/notes-endpoints.md.

Notes are annotations; no route here touches transactions, lots or allocations.
"""
from datetime import date
from typing import Optional

from fastapi import Depends, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator


class NoteCreate(BaseModel):
    category: str  # 'note', 'todo', 'reminder'
    title: str
    body: Optional[str] = ""
    priority: Optional[str] = "normal"  # 'low', 'normal', 'high'
    due_date: Optional[str] = None  # YYYY-MM-DD
    entity_type: Optional[str] = None  # 'product', 'lot', 'customer', 'supplier'
    entity_id: Optional[str] = None

    @validator("category")
    def validate_category(cls, v):
        if v not in ("note", "todo", "reminder"):
            raise ValueError("category must be 'note', 'todo', or 'reminder'")
        return v

    @validator("priority")
    def validate_priority(cls, v):
        if v not in ("low", "normal", "high"):
            raise ValueError("priority must be 'low', 'normal', or 'high'")
        return v

class NoteUpdate(BaseModel):
    title: Optional[str] = None
    body: Optional[str] = None
    priority: Optional[str] = None
    status: Optional[str] = None  # 'open', 'done', 'dismissed'
    due_date: Optional[str] = None  # YYYY-MM-DD or empty string to clear
    entity_type: Optional[str] = None
    entity_id: Optional[str] = None

    @validator("priority")
    def validate_priority(cls, v):
        if v is not None and v not in ("low", "normal", "high"):
            raise ValueError("priority must be 'low', 'normal', or 'high'")
        return v

    @validator("status")
    def validate_status(cls, v):
        if v is not None and v not in ("open", "done", "dismissed"):
            raise ValueError("status must be 'open', 'done', or 'dismissed'")
        return v


def register_routes(app, api):
    def _note_row_to_dict(row):
        """Convert a notes DB row to a JSON-safe dict."""
        d = dict(row)
        for key in ("created_at", "updated_at"):
            if d.get(key):
                date_str, time_str = api.format_timestamp(d[key])
                d[key] = f"{date_str} {time_str}"
        if d.get("due_date"):
            d["due_date"] = str(d["due_date"])
        return d


    @app.get("/dashboard/api/notes")
    def dashboard_api_notes(
        category: Optional[str] = Query(default=None),
        status: Optional[str] = Query(default=None),
        entity_type: Optional[str] = Query(default=None),
        entity_id: Optional[str] = Query(default=None),
        _: bool = Depends(api.verify_api_key),
    ):
        """List notes/todos/reminders with optional filters."""
        try:
            with api.get_transaction() as cur:
                clauses = []
                params = []
                if category:
                    clauses.append("category = %s")
                    params.append(category)
                if status:
                    clauses.append("status = %s")
                    params.append(status)
                if entity_type:
                    clauses.append("entity_type = %s")
                    params.append(entity_type)
                if entity_id:
                    clauses.append("entity_id = %s")
                    params.append(entity_id)

                where = ""
                if clauses:
                    where = "WHERE " + " AND ".join(clauses)

                cur.execute(f"""
                    SELECT * FROM notes
                    {where}
                    ORDER BY
                        CASE WHEN status = 'open' THEN 0 ELSE 1 END,
                        CASE priority WHEN 'high' THEN 0 WHEN 'normal' THEN 1 ELSE 2 END,
                        due_date ASC NULLS LAST,
                        created_at DESC
                """, params)
                rows = [_note_row_to_dict(r) for r in cur.fetchall()]
                return {"notes": rows}
        except Exception as e:
            api.logger.error(f"Dashboard notes list failed: {e}")
            return JSONResponse(status_code=500, content={"error": str(e)})


    @app.post("/dashboard/api/notes")
    def dashboard_api_notes_create(req: NoteCreate, request: Request, _: bool = Depends(api.verify_api_key)):
        """Create a note/todo/reminder."""
        try:
            with api.get_transaction() as cur:
                due = None
                if req.due_date:
                    due = date.fromisoformat(req.due_date)

                cur.execute("""
                    INSERT INTO notes (category, title, body, priority, due_date, entity_type, entity_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING *
                """, (req.category, req.title, req.body or "", req.priority, due,
                      req.entity_type, req.entity_id))
                row = cur.fetchone()
                api._record_actor_write(cur, request, "notes", row['id'])
                return _note_row_to_dict(row)
        except ValueError as e:
            return JSONResponse(status_code=400, content={"error": str(e)})
        except Exception as e:
            if api._is_readonly_error(e): raise
            api.logger.error(f"Dashboard notes create failed: {e}")
            return JSONResponse(status_code=500, content={"error": str(e)})


    @app.put("/dashboard/api/notes/{note_id}")
    def dashboard_api_notes_update(note_id: int, req: NoteUpdate, request: Request, _: bool = Depends(api.verify_api_key)):
        """Update a note/todo/reminder."""
        try:
            with api.get_transaction() as cur:
                # Build SET clause dynamically from provided fields
                sets = []
                params = []
                data = req.dict(exclude_unset=True)
                if not data:
                    return JSONResponse(status_code=400, content={"error": "No fields to update"})

                for field, value in data.items():
                    if field == "due_date":
                        if value == "" or value is None:
                            sets.append("due_date = NULL")
                        else:
                            sets.append("due_date = %s")
                            params.append(date.fromisoformat(value))
                    else:
                        sets.append(f"{field} = %s")
                        params.append(value)

                sets.append("updated_at = NOW()")
                params.append(note_id)

                cur.execute(f"""
                    UPDATE notes SET {', '.join(sets)}
                    WHERE id = %s
                    RETURNING *
                """, params)
                row = cur.fetchone()
                if not row:
                    return JSONResponse(status_code=404, content={"error": "Note not found"})
                api._record_actor_write(cur, request, "notes", row['id'])
                return _note_row_to_dict(row)
        except ValueError as e:
            return JSONResponse(status_code=400, content={"error": str(e)})
        except Exception as e:
            if api._is_readonly_error(e): raise
            api.logger.error(f"Dashboard notes update failed: {e}")
            return JSONResponse(status_code=500, content={"error": str(e)})


    @app.delete("/dashboard/api/notes/{note_id}")
    def dashboard_api_notes_delete(note_id: int, request: Request, _: bool = Depends(api.verify_api_key)):
        """Delete a note/todo/reminder."""
        try:
            with api.get_transaction() as cur:
                cur.execute("DELETE FROM notes WHERE id = %s RETURNING id", (note_id,))
                row = cur.fetchone()
                if not row:
                    return JSONResponse(status_code=404, content={"error": "Note not found"})
                api._record_actor_write(cur, request, "notes", note_id)
                return {"deleted": True, "id": note_id}
        except Exception as e:
            if api._is_readonly_error(e): raise
            api.logger.error(f"Dashboard notes delete failed: {e}")
            return JSONResponse(status_code=500, content={"error": str(e)})


    @app.put("/dashboard/api/notes/{note_id}/toggle")
    def dashboard_api_notes_toggle(note_id: int, request: Request, _: bool = Depends(api.verify_api_key)):
        """Toggle a note's status between open and done."""
        try:
            with api.get_transaction() as cur:
                cur.execute("SELECT status FROM notes WHERE id = %s", (note_id,))
                row = cur.fetchone()
                if not row:
                    return JSONResponse(status_code=404, content={"error": "Note not found"})
                new_status = "done" if row["status"] == "open" else "open"
                cur.execute("""
                    UPDATE notes SET status = %s, updated_at = NOW()
                    WHERE id = %s RETURNING *
                """, (new_status, note_id))
                row = cur.fetchone()
                api._record_actor_write(cur, request, "notes", row['id'])
                return _note_row_to_dict(row)
        except Exception as e:
            if api._is_readonly_error(e): raise
            api.logger.error(f"Dashboard notes toggle failed: {e}")
            return JSONResponse(status_code=500, content={"error": str(e)})
