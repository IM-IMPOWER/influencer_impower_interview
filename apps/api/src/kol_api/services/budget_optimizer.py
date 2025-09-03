"""Budget optimization service for POC4 - Algorithmic budget allocation."""

import asyncio
import json
from typing import List, Dict, Any, Optional, Tuple
from decimal import Decimal
from dataclasses import dataclass
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession
import structlog

from kol_api.database.models.campaign import Campaign
from kol_api.database.models.kol import KOL, KOLMetrics
from kol_api.database.models.budget import (
    BudgetPlan, BudgetAllocation, BudgetStatus,
    OptimizationObjective, AllocationStrategy
)
from kol_api.config import settings

logger = structlog.get_logger()


@dataclass
class KOLCandidate:
    """Data class for KOL candidate with performance predictions."""
    kol: KOL
    metrics: KOLMetrics
    estimated_cost: Decimal
    predicted_reach: int
    predicted_engagement: int
    predicted_conversions: int
    efficiency_score: Decimal
    risk_score: Decimal


@dataclass  
class OptimizationResult:
    """Result of budget optimization algorithm."""
    selected_kols: List[KOLCandidate]
    total_cost: Decimal
    predicted_performance: Dict[str, int]
    optimization_score: Decimal
    alternative_allocations: List[Dict[str, Any]]
    constraints_met: bool
    optimization_metadata: Dict[str, Any]


class BudgetOptimizerService:
    """Service for algorithmic budget optimization and KOL selection."""
    
    def __init__(self, db_session: AsyncSession):
        self.db_session = db_session
        self.budget_tiers = settings.budget_tiers
    
    async def optimize_campaign_budget(
        self,
        campaign_id: str,
        total_budget: Decimal,
        optimization_objective: OptimizationObjective,
        allocation_strategy: AllocationStrategy,
        constraints: Optional[Dict[str, Any]] = None
    ) -> OptimizationResult:
        """
        Optimize budget allocation for campaign using algorithmic approach.
        
        Args:
            campaign_id: Target campaign ID
            total_budget: Available budget amount
            optimization_objective: Primary optimization goal
            allocation_strategy: Budget allocation approach
            constraints: Additional constraints (tier requirements, etc.)
            
        Returns:
            OptimizationResult with recommended KOL selection and allocation
        """
        
        # AIDEV-NOTE: Get campaign and validate requirements
        campaign = await self._get_campaign_requirements(campaign_id)
        if not campaign:
            raise ValueError(f"Campaign {campaign_id} not found")
        
        # AIDEV-NOTE: Get candidate KOLs with performance data
        candidates = await self._get_kol_candidates(campaign)
        if not candidates:
            raise ValueError("No suitable KOL candidates found for campaign")
        
        # AIDEV-NOTE: Apply optimization algorithm based on objective
        optimization_result = await self._run_optimization_algorithm(
            candidates=candidates,
            total_budget=total_budget,
            objective=optimization_objective,
            strategy=allocation_strategy,
            constraints=constraints or {},
            campaign=campaign
        )
        
        logger.info(
            "Budget optimization completed",
            campaign_id=campaign_id,
            total_budget=float(total_budget),
            selected_kols=len(optimization_result.selected_kols),
            optimization_score=float(optimization_result.optimization_score)
        )
        
        return optimization_result
    
    async def create_budget_plan_from_optimization(
        self,
        campaign_id: str,
        optimization_result: OptimizationResult,
        plan_name: str,
        user_id: str
    ) -> BudgetPlan:
        """
        Create a formal budget plan from optimization results.
        
        Args:
            campaign_id: Target campaign ID
            optimization_result: Result from optimization algorithm
            plan_name: Name for the budget plan
            user_id: User creating the plan
            
        Returns:
            Created BudgetPlan instance
        """
        
        # AIDEV-NOTE: Create the main budget plan
        budget_plan = BudgetPlan(
            campaign_id=campaign_id,
            name=plan_name,
            status=BudgetStatus.DRAFT,
            optimization_objective=optimization_result.optimization_metadata["objective"],
            allocation_strategy=optimization_result.optimization_metadata["strategy"],
            total_budget=optimization_result.total_cost,
            available_budget=optimization_result.total_cost,
            predicted_reach=optimization_result.predicted_performance.get("reach", 0),
            predicted_engagement=optimization_result.predicted_performance.get("engagement", 0),
            predicted_conversions=optimization_result.predicted_performance.get("conversions", 0),
            optimization_score=optimization_result.optimization_score,
            optimization_algorithm="linear_programming_v1",
            algorithm_version="1.0",
            objective_weights=optimization_result.optimization_metadata.get("weights", {}),
        )
        
        self.db_session.add(budget_plan)
        await self.db_session.flush()  # Get the ID
        
        # AIDEV-NOTE: Create individual allocations for each selected KOL
        for candidate in optimization_result.selected_kols:
            allocation = BudgetAllocation(
                budget_plan_id=budget_plan.id,
                kol_id=candidate.kol.id,
                allocation_name=f"KOL: {candidate.kol.display_name}",
                allocation_type="kol_payment",
                allocated_amount=candidate.estimated_cost,
                target_tier=candidate.kol.tier.value,
                target_category=candidate.kol.primary_category.value,
                expected_reach=candidate.predicted_reach,
                expected_engagement=candidate.predicted_engagement,
                expected_conversions=candidate.predicted_conversions,
                efficiency_score=candidate.efficiency_score,
                priority_score=candidate.efficiency_score,  # Use same for now
                risk_score=candidate.risk_score,
            )
            self.db_session.add(allocation)
        
        await self.db_session.commit()
        
        logger.info(
            "Budget plan created from optimization",
            plan_id=budget_plan.id,
            campaign_id=campaign_id,
            allocations=len(optimization_result.selected_kols)
        )
        
        return budget_plan
    
    async def generate_alternative_scenarios(
        self,
        campaign_id: str,
        base_budget: Decimal,
        budget_scenarios: List[Decimal],
        optimization_objective: OptimizationObjective
    ) -> List[OptimizationResult]:
        """
        Generate multiple optimization scenarios with different budget amounts.
        
        Args:
            campaign_id: Target campaign ID
            base_budget: Base budget amount
            budget_scenarios: List of budget amounts to test
            optimization_objective: Optimization goal
            
        Returns:
            List of optimization results for each scenario
        """
        
        scenarios = []
        
        for scenario_budget in budget_scenarios:
            try:
                result = await self.optimize_campaign_budget(
                    campaign_id=campaign_id,
                    total_budget=scenario_budget,
                    optimization_objective=optimization_objective,
                    allocation_strategy=AllocationStrategy.PERFORMANCE_WEIGHTED
                )
                scenarios.append(result)
            except Exception as e:
                logger.warning(
                    "Failed to generate scenario",
                    budget=float(scenario_budget),
                    error=str(e)
                )
                continue
        
        return scenarios
    
    async def _get_campaign_requirements(self, campaign_id: str) -> Optional[Campaign]:
        """Get campaign with budget requirements."""
        result = await self.db_session.execute(
            select(Campaign).where(Campaign.id == campaign_id)
        )
        return result.scalar_one_or_none()
    
    async def _get_kol_candidates(self, campaign: Campaign) -> List[KOLCandidate]:
        """
        Get eligible KOL candidates with performance predictions.
        
        Args:
            campaign: Target campaign
            
        Returns:
            List of KOL candidates with cost and performance estimates
        """
        
        # AIDEV-NOTE: Query KOLs matching campaign criteria
        query = select(KOL, KOLMetrics).join(KOLMetrics).where(
            and_(
                KOL.is_active == True,
                KOL.is_brand_safe == True,
                KOL.tier.in_(campaign.target_kol_tiers)
            )
        )
        
        # AIDEV-NOTE: Apply follower count filters
        if campaign.min_follower_count:
            query = query.where(KOLMetrics.follower_count >= campaign.min_follower_count)
        if campaign.max_follower_count:
            query = query.where(KOLMetrics.follower_count <= campaign.max_follower_count)
        
        # AIDEV-NOTE: Apply engagement rate filter
        if campaign.min_engagement_rate:
            query = query.where(KOLMetrics.engagement_rate >= campaign.min_engagement_rate)
        
        result = await self.db_session.execute(query)
        kol_metrics_pairs = result.fetchall()
        
        candidates = []
        
        for kol, metrics in kol_metrics_pairs:
            # AIDEV-NOTE: Estimate costs and performance for each candidate
            estimated_cost = self._estimate_kol_cost(kol, metrics)
            predicted_reach = self._predict_reach(kol, metrics)
            predicted_engagement = self._predict_engagement(kol, metrics)
            predicted_conversions = self._predict_conversions(kol, metrics, campaign)
            efficiency_score = self._calculate_efficiency_score(
                estimated_cost, predicted_reach, predicted_engagement
            )
            risk_score = self._calculate_risk_score(kol, metrics)
            
            candidate = KOLCandidate(
                kol=kol,
                metrics=metrics,
                estimated_cost=estimated_cost,
                predicted_reach=predicted_reach,
                predicted_engagement=predicted_engagement,
                predicted_conversions=predicted_conversions,
                efficiency_score=efficiency_score,
                risk_score=risk_score
            )
            
            candidates.append(candidate)
        
        return candidates
    
    async def _run_optimization_algorithm(
        self,
        candidates: List[KOLCandidate],
        total_budget: Decimal,
        objective: OptimizationObjective,
        strategy: AllocationStrategy,
        constraints: Dict[str, Any],
        campaign: Campaign
    ) -> OptimizationResult:
        """
        Run the optimization algorithm to select optimal KOL mix.
        
        This is a simplified version - in production, would use more sophisticated
        algorithms like linear programming, genetic algorithms, or ML-based optimization.
        """
        
        # AIDEV-NOTE: Sort candidates by efficiency based on objective
        if objective == OptimizationObjective.MAXIMIZE_REACH:
            candidates.sort(key=lambda x: x.predicted_reach / float(x.estimated_cost), reverse=True)
        elif objective == OptimizationObjective.MAXIMIZE_ENGAGEMENT:
            candidates.sort(key=lambda x: x.predicted_engagement / float(x.estimated_cost), reverse=True)
        elif objective == OptimizationObjective.MAXIMIZE_CONVERSIONS:
            candidates.sort(key=lambda x: x.predicted_conversions / float(x.estimated_cost), reverse=True)
        else:  # MINIMIZE_COST or BALANCED
            candidates.sort(key=lambda x: float(x.efficiency_score), reverse=True)
        
        # AIDEV-NOTE: Greedy selection algorithm with constraints
        selected_kols = []
        remaining_budget = total_budget
        
        # AIDEV-NOTE: Apply tier constraints if specified
        tier_requirements = constraints.get("tier_requirements", {})
        tier_counts = {tier: 0 for tier in ["nano", "micro", "mid", "macro", "mega"]}
        
        for candidate in candidates:
            # AIDEV-NOTE: Check if we can afford this KOL
            if candidate.estimated_cost > remaining_budget:
                continue
            
            # AIDEV-NOTE: Check tier constraints
            kol_tier = candidate.kol.tier.value
            if tier_requirements.get(kol_tier):
                required = tier_requirements[kol_tier]
                if tier_counts[kol_tier] >= required:
                    continue
            
            # AIDEV-NOTE: Check risk tolerance
            max_risk = constraints.get("max_risk_per_kol", Decimal("0.8"))
            if candidate.risk_score > max_risk:
                continue
            
            # AIDEV-NOTE: Select this KOL
            selected_kols.append(candidate)
            remaining_budget -= candidate.estimated_cost
            tier_counts[kol_tier] += 1
            
            # AIDEV-NOTE: Stop if we've reached KOL count limits
            max_kols = constraints.get("max_kols", 20)
            if len(selected_kols) >= max_kols:
                break
        
        # AIDEV-NOTE: Calculate aggregate performance predictions
        total_cost = sum(kol.estimated_cost for kol in selected_kols)
        total_reach = sum(kol.predicted_reach for kol in selected_kols)
        total_engagement = sum(kol.predicted_engagement for kol in selected_kols)
        total_conversions = sum(kol.predicted_conversions for kol in selected_kols)
        
        # AIDEV-NOTE: Calculate optimization score based on objective
        optimization_score = self._calculate_optimization_score(
            selected_kols, objective, total_budget
        )
        
        # AIDEV-NOTE: Check if all constraints were met
        constraints_met = self._validate_constraints(selected_kols, constraints, tier_requirements)
        
        return OptimizationResult(
            selected_kols=selected_kols,
            total_cost=total_cost,
            predicted_performance={
                "reach": total_reach,
                "engagement": total_engagement,
                "conversions": total_conversions,
            },
            optimization_score=optimization_score,
            alternative_allocations=[],  # Would generate alternatives in full implementation
            constraints_met=constraints_met,
            optimization_metadata={
                "objective": objective.value,
                "strategy": strategy.value,
                "algorithm": "greedy_selection",
                "candidates_evaluated": len(candidates),
                "budget_utilization": float(total_cost / total_budget),
                "tier_distribution": tier_counts,
            }
        )
    
    def _estimate_kol_cost(self, kol: KOL, metrics: KOLMetrics) -> Decimal:
        """Estimate KOL cost based on tier and performance."""
        
        # AIDEV-NOTE: Use tier-based baseline costs
        tier_costs = self.budget_tiers.get(kol.tier.value, {})
        base_cost = Decimal(str(tier_costs.get("avg_cost_per_post", 1000)))
        
        # AIDEV-NOTE: Adjust based on performance metrics
        if metrics.engagement_rate:
            # Higher engagement commands premium
            engagement_multiplier = min(
                Decimal("2.0"),
                Decimal("1.0") + (metrics.engagement_rate * Decimal("10"))
            )
            base_cost *= engagement_multiplier
        
        # AIDEV-NOTE: Verified accounts command premium
        if kol.is_verified:
            base_cost *= Decimal("1.3")
        
        # AIDEV-NOTE: Location-based adjustments (simplified)
        if kol.location and "bangkok" in kol.location.lower():
            base_cost *= Decimal("1.2")  # Urban premium
        
        return base_cost
    
    def _predict_reach(self, kol: KOL, metrics: KOLMetrics) -> int:
        """Predict reach for a KOL post."""
        
        # AIDEV-NOTE: Base reach is a percentage of followers
        base_reach_rate = 0.15  # 15% of followers see typical post
        
        # AIDEV-NOTE: Adjust for engagement rate
        if metrics.engagement_rate:
            # Higher engagement suggests better algorithmic reach
            engagement_boost = min(2.0, 1.0 + (float(metrics.engagement_rate) * 5))
            base_reach_rate *= engagement_boost
        
        # AIDEV-NOTE: Tier-based adjustments
        tier_multipliers = {
            "nano": 0.8,    # Lower algorithmic reach
            "micro": 1.0,   # Baseline
            "mid": 1.1,     # Better platform support
            "macro": 1.3,   # Strong algorithmic support
            "mega": 1.5,    # Maximum reach
        }
        
        multiplier = tier_multipliers.get(kol.tier.value, 1.0)
        predicted_reach = int(metrics.follower_count * base_reach_rate * multiplier)
        
        return min(predicted_reach, metrics.follower_count)  # Can't exceed follower count
    
    def _predict_engagement(self, kol: KOL, metrics: KOLMetrics) -> int:
        """Predict engagement for a KOL post."""
        
        predicted_reach = self._predict_reach(kol, metrics)
        
        # AIDEV-NOTE: Use historical engagement rate
        if metrics.engagement_rate:
            engagement_rate = float(metrics.engagement_rate)
        else:
            # AIDEV-NOTE: Default rates by tier
            default_rates = {
                "nano": 0.05,
                "micro": 0.04,
                "mid": 0.03,
                "macro": 0.02,
                "mega": 0.015,
            }
            engagement_rate = default_rates.get(kol.tier.value, 0.03)
        
        return int(predicted_reach * engagement_rate)
    
    def _predict_conversions(
        self,
        kol: KOL,
        metrics: KOLMetrics,
        campaign: Campaign
    ) -> int:
        """Predict conversions based on engagement and campaign type."""
        
        predicted_engagement = self._predict_engagement(kol, metrics)
        
        # AIDEV-NOTE: Conversion rates by campaign objective
        conversion_rates = {
            "brand_awareness": 0.001,      # Very low conversion
            "engagement": 0.0,             # No direct conversions
            "lead_generation": 0.02,       # 2% of engaged users
            "sales": 0.015,                # 1.5% of engaged users
            "app_installs": 0.025,         # 2.5% of engaged users
            "website_traffic": 0.05,       # 5% click through
        }
        
        conversion_rate = conversion_rates.get(
            campaign.objective.value,
            0.01
        )
        
        return int(predicted_engagement * conversion_rate)
    
    def _calculate_efficiency_score(
        self,
        cost: Decimal,
        reach: int,
        engagement: int
    ) -> Decimal:
        """Calculate efficiency score for cost-benefit analysis."""
        
        if cost == 0:
            return Decimal("0.0")
        
        # AIDEV-NOTE: Cost per reach and cost per engagement
        cost_per_reach = float(cost) / max(reach, 1)
        cost_per_engagement = float(cost) / max(engagement, 1)
        
        # AIDEV-NOTE: Normalize to 0-1 scale (lower cost = higher score)
        # Assuming $0.01-$1 per reach is reasonable range
        reach_efficiency = max(0.0, min(1.0, 1.0 - (cost_per_reach - 0.01) / 0.99))
        
        # AIDEV-NOTE: Assuming $0.1-$10 per engagement is reasonable range
        engagement_efficiency = max(0.0, min(1.0, 1.0 - (cost_per_engagement - 0.1) / 9.9))
        
        # AIDEV-NOTE: Weighted average
        overall_efficiency = (reach_efficiency * 0.4) + (engagement_efficiency * 0.6)
        
        return Decimal(str(overall_efficiency))
    
    def _calculate_risk_score(self, kol: KOL, metrics: KOLMetrics) -> Decimal:
        """Calculate risk score for KOL selection."""
        
        risk_factors = []
        
        # AIDEV-NOTE: Account verification reduces risk
        if not kol.is_verified:
            risk_factors.append(0.2)
        
        # AIDEV-NOTE: Low engagement rate increases risk
        if metrics.engagement_rate and metrics.engagement_rate < Decimal("0.01"):
            risk_factors.append(0.3)
        
        # AIDEV-NOTE: Inconsistent posting increases risk
        if metrics.posts_last_30_days < 3:
            risk_factors.append(0.2)
        
        # AIDEV-NOTE: High fake follower percentage increases risk
        if metrics.fake_follower_percentage and metrics.fake_follower_percentage > Decimal("0.2"):
            risk_factors.append(0.4)
        
        # AIDEV-NOTE: No recent data increases risk
        if not metrics.metrics_date or (
            (metrics.metrics_date - metrics.created_at).days > 30
        ):
            risk_factors.append(0.1)
        
        # AIDEV-NOTE: Calculate overall risk (0 = no risk, 1 = high risk)
        total_risk = min(1.0, sum(risk_factors))
        
        return Decimal(str(total_risk))
    
    def _calculate_optimization_score(
        self,
        selected_kols: List[KOLCandidate],
        objective: OptimizationObjective,
        total_budget: Decimal
    ) -> Decimal:
        """Calculate overall optimization score."""
        
        if not selected_kols:
            return Decimal("0.0")
        
        total_cost = sum(kol.estimated_cost for kol in selected_kols)
        
        if objective == OptimizationObjective.MAXIMIZE_REACH:
            total_reach = sum(kol.predicted_reach for kol in selected_kols)
            # AIDEV-NOTE: Score based on reach per dollar spent
            score = (total_reach / float(total_cost)) / 1000  # Normalize
        elif objective == OptimizationObjective.MAXIMIZE_ENGAGEMENT:
            total_engagement = sum(kol.predicted_engagement for kol in selected_kols)
            score = (total_engagement / float(total_cost)) / 100  # Normalize
        elif objective == OptimizationObjective.MINIMIZE_COST:
            # AIDEV-NOTE: Score based on budget efficiency
            score = 1.0 - (float(total_cost) / float(total_budget))
        else:  # BALANCED or other
            # AIDEV-NOTE: Average efficiency score
            avg_efficiency = sum(kol.efficiency_score for kol in selected_kols) / len(selected_kols)
            score = float(avg_efficiency)
        
        return Decimal(str(min(max(score, 0.0), 1.0)))
    
    def _validate_constraints(
        self,
        selected_kols: List[KOLCandidate],
        constraints: Dict[str, Any],
        tier_requirements: Dict[str, int]
    ) -> bool:
        """Validate that all constraints were met."""
        
        # AIDEV-NOTE: Check tier requirements
        tier_counts = {}
        for kol in selected_kols:
            tier = kol.kol.tier.value
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
        
        for tier, required in tier_requirements.items():
            if tier_counts.get(tier, 0) < required:
                return False
        
        # AIDEV-NOTE: Check minimum KOL count
        min_kols = constraints.get("min_kols", 1)
        if len(selected_kols) < min_kols:
            return False
        
        # AIDEV-NOTE: Check maximum KOL count
        max_kols = constraints.get("max_kols", float('inf'))
        if len(selected_kols) > max_kols:
            return False
        
        return True