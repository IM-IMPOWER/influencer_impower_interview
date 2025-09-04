"use client";

import React, { useCallback, useState } from "react";
import { Upload, FileText, X, Check, AlertCircle } from "lucide-react";
import { Button } from "./button";
import { Progress } from "./progress";
import { cn } from "@/lib/utils";

// AIDEV-NOTE: 250903170000 - Reusable file upload component with drag-and-drop support

export interface FileUploadProps {
  onFileSelect: (file: File) => void;
  onFileRemove?: () => void;
  accept?: string;
  maxSize?: number; // in bytes
  className?: string;
  disabled?: boolean;
  multiple?: boolean;
  uploadProgress?: number;
  uploadStatus?: "idle" | "uploading" | "success" | "error";
  errorMessage?: string;
  placeholder?: string;
  description?: string;
}

export function FileUpload({
  onFileSelect,
  onFileRemove,
  accept = ".md,.markdown",
  maxSize = 10 * 1024 * 1024, // 10MB default
  className,
  disabled = false,
  multiple = false,
  uploadProgress,
  uploadStatus = "idle",
  errorMessage,
  placeholder = "Drop your file here",
  description = "or click to browse files",
  ...props
}: FileUploadProps) {
  const [isDragOver, setIsDragOver] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
  };

  const validateFile = (file: File): string | null => {
    if (maxSize && file.size > maxSize) {
      return `File size must be less than ${formatFileSize(maxSize)}`;
    }

    if (accept) {
      const acceptedTypes = accept.split(",").map(type => type.trim());
      const fileExtension = "." + file.name.split(".").pop()?.toLowerCase();
      const isValidType = acceptedTypes.some(type => {
        if (type.startsWith(".")) {
          return type === fileExtension;
        }
        return file.type.match(type);
      });

      if (!isValidType) {
        return `File type not supported. Accepted types: ${accept}`;
      }
    }

    return null;
  };

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    if (!disabled) {
      setIsDragOver(true);
    }
  }, [disabled]);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);

    if (disabled) return;

    const files = Array.from(e.dataTransfer.files);
    const file = files[0]; // Take first file if multiple not allowed

    if (file) {
      const error = validateFile(file);
      if (error) {
        // Could emit error event or show toast here
        return;
      }

      setSelectedFile(file);
      onFileSelect(file);
    }
  }, [disabled, onFileSelect, maxSize, accept]);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const error = validateFile(file);
      if (error) {
        // Could emit error event or show toast here
        return;
      }

      setSelectedFile(file);
      onFileSelect(file);
    }
    // Reset input value to allow selecting the same file again
    e.target.value = "";
  }, [onFileSelect, maxSize, accept]);

  const handleRemoveFile = useCallback(() => {
    setSelectedFile(null);
    onFileRemove?.();
  }, [onFileRemove]);

  const getStatusIcon = () => {
    switch (uploadStatus) {
      case "uploading":
        return <Upload className="h-6 w-6 animate-pulse" />;
      case "success":
        return <Check className="h-6 w-6 text-green-600" />;
      case "error":
        return <AlertCircle className="h-6 w-6 text-red-600" />;
      default:
        return <Upload className="h-6 w-6" />;
    }
  };

  const getStatusMessage = () => {
    switch (uploadStatus) {
      case "uploading":
        return "Uploading...";
      case "success":
        return "Upload successful";
      case "error":
        return errorMessage || "Upload failed";
      default:
        return placeholder;
    }
  };

  const getBorderColor = () => {
    if (disabled) return "border-muted-foreground/25";
    if (uploadStatus === "error") return "border-red-500";
    if (uploadStatus === "success") return "border-green-500";
    if (isDragOver) return "border-primary bg-primary/10";
    return "border-muted-foreground/25 hover:border-primary/50";
  };

  return (
    <div className={cn("w-full", className)} {...props}>
      <div
        className={cn(
          "border-2 border-dashed rounded-lg p-8 text-center transition-all",
          getBorderColor(),
          disabled && "opacity-50 cursor-not-allowed"
        )}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        {selectedFile && uploadStatus !== "error" ? (
          <div className="space-y-4">
            <div className="flex items-center justify-center">
              {getStatusIcon()}
            </div>

            <div className="bg-muted/50 rounded-lg p-4 max-w-md mx-auto">
              <div className="flex items-start justify-between gap-3">
                <FileText className="h-5 w-5 text-muted-foreground flex-shrink-0 mt-0.5" />
                <div className="flex-1 min-w-0">
                  <p className="font-medium text-sm truncate">{selectedFile.name}</p>
                  <p className="text-xs text-muted-foreground">
                    {formatFileSize(selectedFile.size)}
                  </p>
                </div>
                {uploadStatus === "idle" && (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={handleRemoveFile}
                    className="h-auto p-1 hover:bg-destructive/10"
                  >
                    <X className="h-4 w-4" />
                  </Button>
                )}
              </div>
            </div>

            {uploadStatus === "uploading" && uploadProgress !== undefined && (
              <div className="max-w-md mx-auto">
                <Progress value={uploadProgress} className="h-2" />
                <p className="text-xs text-muted-foreground mt-1">
                  {uploadProgress.toFixed(0)}% complete
                </p>
              </div>
            )}

            <p className="text-sm text-muted-foreground">
              {getStatusMessage()}
            </p>
          </div>
        ) : (
          <div className="space-y-4">
            <div className="flex items-center justify-center">
              {getStatusIcon()}
            </div>

            <div>
              <p className="text-lg font-medium mb-2">
                {getStatusMessage()}
              </p>
              <p className="text-muted-foreground mb-4">
                {description}
              </p>
              
              <input
                type="file"
                accept={accept}
                multiple={multiple}
                onChange={handleFileSelect}
                disabled={disabled}
                className="hidden"
                id="file-upload-input"
              />
              <label htmlFor="file-upload-input">
                <Button 
                  asChild 
                  variant="outline"
                  disabled={disabled}
                  className={uploadStatus === "error" ? "border-red-500 text-red-600" : ""}
                >
                  <span>Choose File</span>
                </Button>
              </label>
            </div>

            {errorMessage && uploadStatus === "error" && (
              <p className="text-sm text-red-600 bg-red-50 dark:bg-red-900/10 p-2 rounded">
                {errorMessage}
              </p>
            )}

            <p className="text-xs text-muted-foreground">
              Maximum file size: {formatFileSize(maxSize)}
              {accept && (
                <>
                  <br />
                  Accepted formats: {accept}
                </>
              )}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}