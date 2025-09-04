"""GraphQL resolvers for budget optimization operations."""

from typing import List, Dict, Any, Optional
import json
from decimal import Decimal
import structlog

from kol_api.graphql.resolvers.base import BaseResolver
from kol_api.graphql.types import BudgetPlan, BudgetPlanCreateInput, BudgetOptimizationResult, OperationResult
from kol_api.services.budget_optimizer import BudgetOptimizerService
from kol_api.database.models.budget import (
    BudgetPlan, BudgetAllocation, BudgetStatus, 
    OptimizationObjective, AllocationStrategy
)
from kol_api.database.models.campaign import Campaign
from kol_api.utils.converters import (
    convert_budget_plan_to_graphql, convert_optimization_result_to_graphql
)
from sqlalchemy import select, and_, desc, func
from sqlalchemy.orm import selectinload
from datetime import datetime

logger = structlog.get_logger()


class BudgetResolvers(BaseResolver):
    """GraphQL resolvers for budget optimization operations."""
    
    @staticmethod
    async def get_budget_plans(
        context: Dict[str, Any],
        campaign_id: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> List[BudgetPlan]:
        """Get budget plans, optionally filtered by campaign."""
        try:
            user = BudgetResolvers.require_authentication(context)
            db_session = BudgetResolvers.get_database_session(context)
            
            # AIDEV-NOTE: Build base query with joins
            query = (
                select(BudgetPlan)
                .options(
                    selectinload(BudgetPlan.allocations),
                    selectinload(BudgetPlan.campaign)
                )
                .order_by(desc(BudgetPlan.created_at))
            )
            
            # AIDEV-NOTE: Apply campaign filter
            if campaign_id:
                query = query.where(BudgetPlan.campaign_id == campaign_id)
            
            # AIDEV-NOTE: Apply pagination
            query = query.offset(offset).limit(limit)
            
            # AIDEV-NOTE: Execute query
            result = await db_session.execute(query)
            budget_plans = result.scalars().all()
            
            # AIDEV-NOTE: Convert to GraphQL types
            graphql_plans = []
            for plan in budget_plans:
                allocations = list(plan.allocations) if plan.allocations else []
                graphql_plan = convert_budget_plan_to_graphql(plan, allocations)
                graphql_plans.append(graphql_plan)
            
            logger.info(
                "Budget plans queried successfully",
                user_id=user["id"],
                campaign_id=campaign_id,
                count=len(graphql_plans)
            )
            
            return graphql_plans
            
        except Exception as e:
            BudgetResolvers.log_resolver_error("get_budget_plans", e, context)
            raise
    
    @staticmethod
    async def get_budget_plan_by_id(
        context: Dict[str, Any],
        plan_id: str
    ) -> Optional[BudgetPlan]:
        """Get single budget plan by ID."""
        try:
            user = BudgetResolvers.require_authentication(context)
            db_session = BudgetResolvers.get_database_session(context)
            
            # AIDEV-NOTE: Query budget plan with allocations
            query = (
                select(BudgetPlan)
                .options(
                    selectinload(BudgetPlan.allocations),
                    selectinload(BudgetPlan.campaign)
                )
                .where(BudgetPlan.id == plan_id)
            )
            
            result = await db_session.execute(query)
            budget_plan = result.scalar_one_or_none()
            
            if not budget_plan:
                logger.info("Budget plan not found", user_id=user["id"], plan_id=plan_id)
                return None
            
            # AIDEV-NOTE: Convert to GraphQL type with allocations
            allocations = list(budget_plan.allocations) if budget_plan.allocations else []
            graphql_plan = convert_budget_plan_to_graphql(budget_plan, allocations)
            
            logger.info(
                "Budget plan retrieved successfully",
                user_id=user["id"],
                plan_id=plan_id,
                allocation_count=len(allocations)
            )
            
            return graphql_plan
            
        except Exception as e:
            BudgetResolvers.log_resolver_error(
                "get_budget_plan_by_id", 
                e, 
                context,
                plan_id=plan_id
            )
            raise
    
    @staticmethod
    async def optimize_campaign_budget(
        context: Dict[str, Any],
        campaign_id: str,
        optimization_objective: str,
        constraints: Optional[str] = None,
    ) -> BudgetOptimizationResult:
        """Generate optimized budget allocation."""
        try:
            user = BudgetResolvers.require_role(context, ["admin", "manager", "analyst"])
            db_session = BudgetResolvers.get_database_session(context)
            
            # AIDEV-NOTE: Get campaign to determine budget and constraints
            campaign_query = select(Campaign).where(Campaign.id == campaign_id)
            campaign_result = await db_session.execute(campaign_query)
            campaign = campaign_result.scalar_one_or_none()
            
            if not campaign:
                raise ValueError(f"Campaign {campaign_id} not found")
            
            # AIDEV-NOTE: Parse constraints JSON
            parsed_constraints = {}
            if constraints:
                try:
                    parsed_constraints = json.loads(constraints)
                except json.JSONDecodeError as e:
                    logger.warning("Invalid constraints JSON", constraints=constraints, error=str(e))
                    parsed_constraints = {}
            
            # AIDEV-NOTE: Use campaign budget
            total_budget = campaign.total_budget
            
            # AIDEV-NOTE: Initialize budget optimizer service
            optimizer = BudgetOptimizerService(db_session)
            
            # AIDEV-NOTE: Run optimization algorithm
            optimization_result = await optimizer.optimize_campaign_budget(
                campaign_id=campaign_id,
                total_budget=total_budget,
                optimization_objective=OptimizationObjective(optimization_objective),
                allocation_strategy=AllocationStrategy.PERFORMANCE_WEIGHTED,
                constraints=parsed_constraints
            )
            
            logger.info(
                "Budget optimization completed successfully",
                user_id=user["id"],
                campaign_id=campaign_id,
                objective=optimization_objective,
                selected_kols=len(optimization_result.selected_kols),
                total_cost=float(optimization_result.total_cost),
                optimization_score=float(optimization_result.optimization_score)
            )
            
            # AIDEV-NOTE: Convert result to GraphQL type
            return convert_optimization_result_to_graphql(optimization_result, campaign_id)
            
        except Exception as e:
            BudgetResolvers.log_resolver_error(
                "optimize_campaign_budget", 
                e, 
                context,
                campaign_id=campaign_id,
                objective=optimization_objective
            )
            raise
    
    @staticmethod
    async def create_budget_plan(
        context: Dict[str, Any],
        input: BudgetPlanCreateInput,
    ) -> OperationResult:
        """Create new budget plan with optimization."""
        try:
            user = BudgetResolvers.require_role(context, ["admin", "manager"])
            db_session = BudgetResolvers.get_database_session(context)
            
            # AIDEV-NOTE: Validate campaign exists
            campaign_query = select(Campaign).where(Campaign.id == input.campaign_id)
            campaign_result = await db_session.execute(campaign_query)
            campaign = campaign_result.scalar_one_or_none()
            
            if not campaign:
                raise ValueError(f"Campaign {input.campaign_id} not found")
            
            # AIDEV-NOTE: Parse JSON constraints
            parsed_constraints = {}
            if input.tier_requirements:
                try:
                    parsed_constraints["tier_requirements"] = json.loads(input.tier_requirements)
                except json.JSONDecodeError as e:
                    logger.warning("Invalid tier requirements JSON", error=str(e))
                    parsed_constraints["tier_requirements"] = {}
            
            if input.objective_weights:
                try:
                    parsed_constraints["objective_weights"] = json.loads(input.objective_weights)
                except json.JSONDecodeError as e:
                    logger.warning("Invalid objective weights JSON", error=str(e))
            
            # AIDEV-NOTE: Add additional constraints from input
            if input.min_kols_required:
                parsed_constraints["min_kols"] = input.min_kols_required
            if input.max_kols_allowed:
                parsed_constraints["max_kols"] = input.max_kols_allowed
            if input.reserved_buffer:
                parsed_constraints["reserved_buffer"] = input.reserved_buffer
            
            # AIDEV-NOTE: Initialize budget optimizer service
            optimizer = BudgetOptimizerService(db_session)
            
            # AIDEV-NOTE: Run optimization algorithm
            optimization_result = await optimizer.optimize_campaign_budget(
                campaign_id=input.campaign_id,
                total_budget=Decimal(str(input.total_budget)),
                optimization_objective=OptimizationObjective(input.optimization_objective),
                allocation_strategy=AllocationStrategy(input.allocation_strategy),
                constraints=parsed_constraints
            )
            
            # AIDEV-NOTE: Create formal budget plan from optimization results
            budget_plan = await optimizer.create_budget_plan_from_optimization(
                campaign_id=input.campaign_id,
                optimization_result=optimization_result,
                plan_name=input.name,
                user_id=user["id"]
            )
            
            # AIDEV-NOTE: Set description if provided
            if input.description:
                budget_plan.description = input.description
                await db_session.commit()
            
            logger.info(
                "Budget plan created successfully",
                user_id=user["id"],
                plan_id=budget_plan.id,
                campaign_id=input.campaign_id,
                total_budget=float(input.total_budget),
                optimization_score=float(optimization_result.optimization_score)
            )
            
            return OperationResult(
                success=True,
                message=f"Budget plan '{input.name}' created successfully with {len(optimization_result.selected_kols)} KOL allocations",
                data=str(budget_plan.id)
            )
            
        except Exception as e:
            BudgetResolvers.log_resolver_error("create_budget_plan", e, context)
            raise
    
    @staticmethod
    async def approve_budget_plan(
        context: Dict[str, Any],
        plan_id: str,
    ) -> OperationResult:
        """Approve and activate budget plan."""
        try:
            user = BudgetResolvers.require_role(context, ["admin", "manager"])
            db_session = BudgetResolvers.get_database_session(context)
            
            # AIDEV-NOTE: Get budget plan and validate
            plan_query = select(BudgetPlan).where(BudgetPlan.id == plan_id)
            result = await db_session.execute(plan_query)
            budget_plan = result.scalar_one_or_none()
            
            if not budget_plan:
                raise ValueError(f"Budget plan {plan_id} not found")
            
            if budget_plan.status == BudgetStatus.APPROVED:
                return OperationResult(
                    success=False,
                    message="Budget plan is already approved",
                    data=None
                )
            
            # AIDEV-NOTE: Update plan status and approval metadata
            old_status = budget_plan.status
            budget_plan.status = BudgetStatus.APPROVED
            budget_plan.approved_at = datetime.utcnow()
            budget_plan.approved_by = user["id"]
            
            # AIDEV-NOTE: Create approval audit entry
            approval_metadata = {
                "approved_by": user["id"],
                "approved_at": datetime.utcnow().isoformat(),
                "previous_status": old_status.value,
                "approval_notes": "Approved via GraphQL API"
            }
            
            # AIDEV-NOTE: In production, would store audit entry and notify stakeholders
            # await audit_service.log_approval(plan_id, approval_metadata)
            # await notification_service.notify_plan_approval(budget_plan)
            
            await db_session.commit()
            
            logger.info(
                "Budget plan approved successfully",
                user_id=user["id"],
                plan_id=plan_id,
                previous_status=old_status.value,
                total_budget=float(budget_plan.total_budget)
            )
            
            return OperationResult(
                success=True,
                message=f"Budget plan '{budget_plan.name}' approved successfully",
                data=str(plan_id)
            )
            
        except Exception as e:
            BudgetResolvers.log_resolver_error(
                "approve_budget_plan", 
                e, 
                context,
                plan_id=plan_id
            )
            raise
    
    @staticmethod
    async def execute_budget_plan(
        context: Dict[str, Any],
        plan_id: str,
    ) -> OperationResult:
        """Execute approved budget plan."""
        try:
            user = BudgetResolvers.require_role(context, ["admin", "manager"])
            db_session = BudgetResolvers.get_database_session(context)
            
            # AIDEV-NOTE: Get budget plan with allocations
            plan_query = (
                select(BudgetPlan)
                .options(selectinload(BudgetPlan.allocations))
                .where(BudgetPlan.id == plan_id)
            )
            result = await db_session.execute(plan_query)
            budget_plan = result.scalar_one_or_none()
            
            if not budget_plan:
                raise ValueError(f"Budget plan {plan_id} not found")
            
            if budget_plan.status != BudgetStatus.APPROVED:
                return OperationResult(
                    success=False,
                    message="Budget plan must be approved before execution",
                    data=None
                )
            
            if budget_plan.status == BudgetStatus.EXECUTED:
                return OperationResult(
                    success=False,
                    message="Budget plan has already been executed",
                    data=None
                )
            
            # AIDEV-NOTE: Execute plan - commit allocations and update status
            old_status = budget_plan.status
            budget_plan.status = BudgetStatus.EXECUTED
            budget_plan.executed_at = datetime.utcnow()
            budget_plan.executed_by = user["id"]
            
            # AIDEV-NOTE: Mark all allocations as committed
            allocation_count = 0
            if budget_plan.allocations:
                for allocation in budget_plan.allocations:
                    allocation.is_committed = True
                    allocation.committed_at = datetime.utcnow()
                    allocation_count += 1
            
            # AIDEV-NOTE: Update campaign allocated budget
            if budget_plan.campaign:
                campaign = budget_plan.campaign
                campaign.allocated_budget += budget_plan.total_budget
            
            # AIDEV-NOTE: Create execution audit entry
            execution_metadata = {
                "executed_by": user["id"],
                "executed_at": datetime.utcnow().isoformat(),
                "previous_status": old_status.value,
                "allocations_committed": allocation_count,
                "total_amount": float(budget_plan.total_budget)
            }
            
            # AIDEV-NOTE: In production, would trigger KOL invitations and notifications
            # await kol_invitation_service.send_invitations(budget_plan)
            # await notification_service.notify_plan_execution(budget_plan)
            
            await db_session.commit()
            
            logger.info(
                "Budget plan executed successfully",
                user_id=user["id"],
                plan_id=plan_id,
                allocations_committed=allocation_count,
                total_budget=float(budget_plan.total_budget)
            )
            
            return OperationResult(
                success=True,
                message=f"Budget plan '{budget_plan.name}' executed successfully. {allocation_count} allocations committed.",
                data=str(plan_id)
            )
            
        except Exception as e:
            BudgetResolvers.log_resolver_error(
                "execute_budget_plan", 
                e, 
                context,
                plan_id=plan_id
            )
            raise