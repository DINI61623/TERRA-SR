#!/usr/bin/env python3
"""
TERRA-SR Structured Error Handling & Response Envelopes
Provides consistent error structures across backend endpoints and processing pipelines.
"""

from typing import Dict, Any, Optional


def create_error_response(
    stage: str,
    message: str,
    suggestion: Optional[str] = None,
    retryable: bool = False,
    details: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Creates a standardized structured error dictionary.
    """
    resp = {
        "status": "ERROR",
        "stage": stage,
        "message": message,
        "retryable": retryable
    }
    if suggestion:
        resp["suggestion"] = suggestion
    if details:
        resp["details"] = details
    return resp


def create_success_response(
    stage: str,
    data: Dict[str, Any],
    message: Optional[str] = None
) -> Dict[str, Any]:
    """
    Creates a standardized success dictionary.
    """
    resp = {
        "status": "SUCCESS",
        "stage": stage,
        **data
    }
    if message:
        resp["message"] = message
    return resp
