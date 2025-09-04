"""File upload REST endpoints."""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import os
import uuid
from pathlib import Path
import aiofiles
import structlog

from kol_api.config import settings

router = APIRouter(prefix="/upload", tags=["File Upload"])
logger = structlog.get_logger()


class UploadResponse(BaseModel):
    """File upload response."""
    file_id: str
    filename: str
    file_path: str
    file_size: int
    content_type: str


@router.post("/", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    category: Optional[str] = Form("general")
):
    """Upload a file with validation and storage."""
    
    # AIDEV-NOTE: Validate file type
    if file.content_type not in settings.allowed_file_types:
        raise HTTPException(
            status_code=400,
            detail=f"File type {file.content_type} not allowed"
        )
    
    # AIDEV-NOTE: Validate file size
    content = await file.read()
    if len(content) > settings.max_file_size:
        raise HTTPException(
            status_code=400,
            detail=f"File size exceeds maximum of {settings.max_file_size} bytes"
        )
    
    # AIDEV-NOTE: Generate unique filename
    file_id = str(uuid.uuid4())
    file_extension = Path(file.filename or "").suffix
    safe_filename = f"{file_id}{file_extension}"
    
    # AIDEV-NOTE: Create category directory
    upload_dir = settings.upload_path / category
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    # AIDEV-NOTE: Save file
    file_path = upload_dir / safe_filename
    async with aiofiles.open(file_path, 'wb') as f:
        await f.write(content)
    
    logger.info(
        "File uploaded successfully",
        file_id=file_id,
        filename=file.filename,
        size=len(content),
        category=category
    )
    
    return UploadResponse(
        file_id=file_id,
        filename=file.filename or safe_filename,
        file_path=str(file_path.relative_to(settings.upload_path)),
        file_size=len(content),
        content_type=file.content_type or "application/octet-stream"
    )


@router.post("/bulk", response_model=List[UploadResponse])
async def upload_multiple_files(
    files: List[UploadFile] = File(...),
    category: Optional[str] = Form("general")
):
    """Upload multiple files."""
    
    if len(files) > 10:  # Limit bulk uploads
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 files allowed in bulk upload"
        )
    
    responses = []
    
    for file in files:
        try:
            response = await upload_file(file, category)
            responses.append(response)
        except Exception as e:
            logger.error(
                "Failed to upload file in bulk",
                filename=file.filename,
                error=str(e)
            )
            continue
    
    return responses


@router.delete("/{file_id}")
async def delete_file(file_id: str):
    """Delete uploaded file."""
    # AIDEV-NOTE: Implement file deletion with security checks
    # In production, verify user has permission to delete the file
    
    # AIDEV-NOTE: Search for file in upload directories
    for category_dir in settings.upload_path.iterdir():
        if category_dir.is_dir():
            for file_path in category_dir.glob(f"{file_id}.*"):
                try:
                    file_path.unlink()
                    logger.info("File deleted", file_id=file_id, path=str(file_path))
                    return {"message": "File deleted successfully", "file_id": file_id}
                except Exception as e:
                    logger.error("Failed to delete file", file_id=file_id, error=str(e))
                    raise HTTPException(status_code=500, detail="Failed to delete file")
    
    raise HTTPException(status_code=404, detail="File not found")


@router.get("/info/{file_id}")
async def get_file_info(file_id: str):
    """Get file information."""
    # AIDEV-NOTE: Implement file info retrieval
    # In production, this would query a file metadata database
    
    return {"file_id": file_id, "status": "File info endpoint not implemented"}