"""
GraphQL security extensions and query complexity analysis for sophisticated operations.

This module provides security controls for the enhanced KOL platform GraphQL API,
including query complexity analysis, rate limiting, and field-level authorization.
"""

from typing import Dict, Any, Optional, List
from decimal import Decimal
import time
from functools import wraps

import strawberry
from strawberry.extensions import SchemaExtension
from strawberry.types import Info
import structlog

logger = structlog.get_logger()


class QueryComplexityExtension(SchemaExtension):
    """
    Analyze query complexity to prevent abuse of sophisticated AI operations.
    
    POC2 and POC4 operations are computationally expensive and need protection.
    """
    
    def __init__(self, max_complexity: int = 2000):
        self.max_complexity = max_complexity
        
        # AIDEV-NOTE: Complexity scores for sophisticated operations
        self.field_complexity_scores = {
            # Basic operations (low complexity)
            "kols": 1,
            "campaigns": 1,
            "budgetPlans": 1,
            
            # Enhanced KOL operations (medium complexity)
            "kol": 5,
            "similarKols": 10,
            
            # Sophisticated AI operations (high complexity)
            "matchKolsForCampaign": 50,          # POC2 multi-factor scoring
            "matchKolsWithRequirements": 60,     # POC2 with full requirements
            "optimizeBudget": 80,                # POC4 optimization
            "budgetOptimizationScenarios": 120,  # POC4 multiple scenarios
            
            # Analytics and reporting (medium-high complexity)
            "kolPerformanceAnalytics": 20,
            "campaignPerformanceSummary": 25,
            "kolMarketAnalysis": 40,
            "dataQualityReport": 15,
            
            # Mutations (variable complexity based on operation)
            "createBudgetPlan": 30,
            "createOptimizedBudgetPlan": 40,
            "triggerKolDataRefresh": 25,
            "parseCampaignBrief": 35,
            "exportKolData": 20,
            "bulkUpdateBrandSafety": 15,
            
            # Nested field complexity multipliers
            "scoreComponents": 5,
            "semanticMatching": 8,
            "performancePrediction": 6,
            "allocations": 3,
            "alternativePlans": 10,
        }
    
    def on_operation(self):
        """Calculate and validate query complexity before execution."""
        
        def complexity_analyzer(next_):
            def wrapper(root, info: Info, **kwargs):
                # AIDEV-NOTE: Calculate query complexity
                complexity = self._calculate_query_complexity(info.field_nodes)
                
                logger.info(
                    "GraphQL query complexity analysis",
                    complexity=complexity,
                    max_complexity=self.max_complexity,
                    operation_name=info.operation.name.value if info.operation.name else None,
                    user_id=info.context.get("user", {}).get("id")
                )
                
                # AIDEV-NOTE: Reject overly complex queries
                if complexity > self.max_complexity:
                    logger.warning(
                        "Query complexity limit exceeded",
                        complexity=complexity,
                        limit=self.max_complexity,
                        user_id=info.context.get("user", {}).get("id")
                    )
                    raise Exception(
                        f"Query complexity ({complexity}) exceeds maximum allowed ({self.max_complexity}). "
                        f"Please simplify your query or break it into smaller operations."
                    )
                
                # AIDEV-NOTE: Add complexity info to context for monitoring
                info.context["query_complexity"] = complexity
                
                return next_(root, info, **kwargs)
            
            return wrapper
        
        return complexity_analyzer
    
    def _calculate_query_complexity(self, field_nodes, depth: int = 0) -> int:
        """
        Calculate complexity score for query field nodes.
        
        Args:
            field_nodes: GraphQL field nodes to analyze
            depth: Current nesting depth
            
        Returns:
            Total complexity score
        """
        if depth > 15:  # Prevent excessive nesting
            return 1000  # High penalty for deep nesting
        
        total_complexity = 0
        
        for field_node in field_nodes:
            field_name = field_node.name.value
            
            # AIDEV-NOTE: Base complexity for field
            base_complexity = self.field_complexity_scores.get(field_name, 1)
            
            # AIDEV-NOTE: Depth penalty for nested operations
            depth_penalty = 1 + (depth * 0.5)
            
            field_complexity = base_complexity * depth_penalty
            
            # AIDEV-NOTE: Handle field arguments that increase complexity
            if field_node.arguments:
                for arg in field_node.arguments:
                    arg_name = arg.name.value
                    
                    # Increase complexity for sophisticated parameters
                    if arg_name in ["enableSemanticMatching", "generateAlternatives", 
                                  "includeRiskAnalysis", "includePredictions"]:
                        if self._get_argument_value(arg):
                            field_complexity *= 1.5
                    
                    elif arg_name == "limit" and hasattr(arg.value, 'value'):
                        # Higher limits increase complexity
                        limit_value = int(arg.value.value)
                        if limit_value > 50:
                            field_complexity *= (1 + (limit_value - 50) / 100)
            
            # AIDEV-NOTE: Recursively calculate nested field complexity
            if field_node.selection_set:
                nested_complexity = self._calculate_query_complexity(
                    field_node.selection_set.selections, 
                    depth + 1
                )
                field_complexity += nested_complexity
            
            total_complexity += field_complexity
        
        return int(total_complexity)
    
    def _get_argument_value(self, argument) -> Any:
        """Extract value from GraphQL argument node."""
        if hasattr(argument.value, 'value'):
            return argument.value.value
        return None


class FieldAuthorizationExtension(SchemaExtension):
    """
    Provide field-level authorization for sensitive operations.
    
    Ensures users have appropriate permissions for sophisticated AI operations.
    """
    
    def __init__(self):
        # AIDEV-NOTE: Define permission requirements for sophisticated operations
        self.field_permissions = {
            # POC2 operations - require analyst+ role
            "matchKolsForCampaign": ["analyst", "manager", "admin"],
            "matchKolsWithRequirements": ["analyst", "manager", "admin"],
            
            # POC4 operations - require manager+ role
            "optimizeBudget": ["manager", "admin"],
            "budgetOptimizationScenarios": ["manager", "admin"],
            "createOptimizedBudgetPlan": ["manager", "admin"],
            
            # Data management - require manager+ role
            "triggerKolDataRefresh": ["manager", "admin"],
            "bulkUpdateBrandSafety": ["manager", "admin"],
            "exportKolData": ["analyst", "manager", "admin"],
            
            # Admin-only operations
            "dataQualityReport": ["admin"],
            "kolMarketAnalysis": ["manager", "admin"],
        }
    
    def on_operation(self):
        """Check field-level permissions during query execution."""
        
        def authorization_checker(next_):
            def wrapper(root, info: Info, **kwargs):
                field_name = info.field_name
                user = info.context.get("user")
                
                # AIDEV-NOTE: Check if field requires authorization
                required_roles = self.field_permissions.get(field_name)
                if required_roles and user:
                    user_role = user.get("role", "viewer").lower()
                    
                    if user_role not in required_roles:
                        logger.warning(
                            "Insufficient permissions for GraphQL field",
                            field_name=field_name,
                            user_role=user_role,
                            required_roles=required_roles,
                            user_id=user.get("id")
                        )
                        raise Exception(
                            f"Insufficient permissions to access '{field_name}'. "
                            f"Required role: {', '.join(required_roles)}"
                        )
                
                return next_(root, info, **kwargs)
            
            return wrapper
        
        return authorization_checker


class RateLimitingExtension(SchemaExtension):
    """
    Rate limiting for expensive AI operations to prevent resource abuse.
    """
    
    def __init__(self):
        self.rate_limits = {
            # AIDEV-NOTE: Rate limits per user per minute
            "matchKolsForCampaign": 10,
            "matchKolsWithRequirements": 5,
            "optimizeBudget": 3,
            "budgetOptimizationScenarios": 1,
            "triggerKolDataRefresh": 5,
            "parseCampaignBrief": 10,
        }
        
        # AIDEV-NOTE: In-memory rate limiting (use Redis in production)
        self.request_counts = {}
    
    def on_operation(self):
        """Apply rate limiting to expensive operations."""
        
        def rate_limiter(next_):
            def wrapper(root, info: Info, **kwargs):
                field_name = info.field_name
                user = info.context.get("user")
                
                if field_name in self.rate_limits and user:
                    user_id = user.get("id")
                    current_time = int(time.time() / 60)  # Per-minute buckets
                    
                    key = f"{user_id}:{field_name}:{current_time}"
                    current_count = self.request_counts.get(key, 0)
                    
                    if current_count >= self.rate_limits[field_name]:
                        logger.warning(
                            "Rate limit exceeded for GraphQL field",
                            field_name=field_name,
                            user_id=user_id,
                            current_count=current_count,
                            limit=self.rate_limits[field_name]
                        )
                        raise Exception(
                            f"Rate limit exceeded for '{field_name}'. "
                            f"Limit: {self.rate_limits[field_name]} requests per minute."
                        )
                    
                    self.request_counts[key] = current_count + 1
                    
                    # AIDEV-NOTE: Clean up old entries periodically
                    self._cleanup_old_entries(current_time)
                
                return next_(root, info, **kwargs)
            
            return wrapper
        
        return rate_limiter
    
    def _cleanup_old_entries(self, current_time: int):
        """Remove old rate limiting entries to prevent memory leaks."""
        keys_to_remove = []
        
        for key in self.request_counts:
            try:
                _, _, timestamp = key.split(":")
                if int(timestamp) < current_time - 5:  # Keep last 5 minutes
                    keys_to_remove.append(key)
            except (ValueError, IndexError):
                keys_to_remove.append(key)  # Remove malformed keys
        
        for key in keys_to_remove:
            self.request_counts.pop(key, None)


def require_role(*allowed_roles):
    """
    Decorator for resolver functions to enforce role-based access.
    
    Usage:
        @require_role("manager", "admin")
        async def sensitive_operation(self, info: Info, ...):
            ...
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # AIDEV-NOTE: Extract info parameter (usually second arg)
            info = None
            for arg in args:
                if isinstance(arg, Info):
                    info = arg
                    break
            
            if not info:
                raise Exception("Could not find Info parameter for role checking")
            
            user = info.context.get("user")
            if not user:
                raise Exception("Authentication required")
            
            user_role = user.get("role", "viewer").lower()
            if user_role not in [role.lower() for role in allowed_roles]:
                raise Exception(
                    f"Insufficient permissions. Required: {', '.join(allowed_roles)}"
                )
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator


def log_performance(operation_name: str):
    """
    Decorator to log performance metrics for sophisticated operations.
    
    Usage:
        @log_performance("POC2_KOL_MATCHING")
        async def match_kols(...):
            ...
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                
                processing_time = time.time() - start_time
                logger.info(
                    f"GraphQL operation completed: {operation_name}",
                    processing_time_seconds=processing_time,
                    success=True
                )
                
                return result
            
            except Exception as e:
                processing_time = time.time() - start_time
                logger.error(
                    f"GraphQL operation failed: {operation_name}",
                    processing_time_seconds=processing_time,
                    error=str(e),
                    success=False
                )
                raise
        
        return wrapper
    return decorator


# AIDEV-NOTE: Usage example in schema.py
"""
from kol_api.graphql.security import (
    QueryComplexityExtension, FieldAuthorizationExtension, 
    RateLimitingExtension, require_role, log_performance
)

# In schema creation
schema = strawberry.Schema(
    query=Query,
    mutation=Mutation,
    extensions=[
        QueryComplexityExtension(max_complexity=2000),
        FieldAuthorizationExtension(),
        RateLimitingExtension(),
    ]
)

# In resolver methods
class BudgetResolvers(BaseResolver):
    
    @staticmethod
    @require_role("manager", "admin")
    @log_performance("POC4_BUDGET_OPTIMIZATION")
    async def optimize_campaign_budget(context, ...):
        # Sophisticated budget optimization implementation
        pass
"""