"""
ISE Simulator v0.1 - Interactive Simulation Workbench
Ambiguity-Bearing Outputs (ABO) in Interconnected Systems Environments

v0.1 Changes (from v33):
- Renamed to ISE Simulator v0.1
- Added feedback decomposition panel: tracks cumulative v3 and v4 contributions
  to calibration offset separately, enabling visualization of the persistence mechanism
- Feedback decomposition included in PDF and Excel exports
- New Streamlit tab: Feedback Dynamics

v33 Changes:
- Default parameters pre-populated from validated simulation run
  (1200 steps, 90 maturation, δ=[500-800] @ [-0.15,-0.10], v4=0.076, v3=0.000110,
   v4_asym=0.03, financial: 3.0/8.0/0.5, ISCIL baseline=200, guardrail=0.05, damping=0.50)

v25 Changes (from v24):
CRITICAL BUG FIX:
- Delta generation now uses a SEPARATE RNG (delta_rng) from the noise/decision RNG.
  Previously, calling rng.uniform() for semantic_modifier during the delta window
  advanced the shared RNG, causing ALL subsequent noise draws to differ between
  Baseline and ABO scenarios. This made post-delta-window results incomparable.

CHART FIX:
- Financial breakdown chart B redesigned: side-by-side bars (green=revenue, coral=costs)

ISCIL ALERT TIME FIX:
- Now tracks two timestamps: First Alert (first CRS >= threshold) and
  Intervention Start (first sustained breach triggering containment).

EXPORT IMPROVEMENTS:
- Full parameters sheet added as first page of PDF export
- Full parameters sheet added to Excel export

LAYOUT FIXES:
- Switched multi-panel plots to constrained_layout
- Risk Distribution and Jump Effects suptitles no longer overlap subplot titles
- Increased figure sizes across all plot types
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple
import pandas as pd

# =============================================================================
# MindXO Color Palette
# =============================================================================
COLORS = {
    # Core Greens (Authority / Trust)
    'green_900': '#1F3935',
    'green_800': '#254642',
    'green_700': '#2A4D47',
    'green_600': '#335E58',
    'green_500': '#3A5550',
    # Tech Mint (AI / Progress / Future)
    'mint_600': '#2FD9A8',
    'mint_500': '#3EEEB9',
    'mint_400': '#62F0C8',
    'mint_300': '#8AF5D9',
    'mint_200': '#C9FAEC',
    # Living Coral (Risk / Current State / Attention)
    'coral_600': '#E86666',
    'coral_500': '#FF7D7D',
    'coral_400': '#FF9A9A',
    'coral_300': '#FFB5B5',
    # Neutrals
    'silver_300': '#D1D5D8',
    'silver_200': '#E7E8E9',
    'silver_100': '#F4F5F6',
    'white': '#FFFFFF',
    # Text
    'text_primary': '#1F3935',
    'text_muted': '#6B7280',
}

# Scenario colors (up to 5)
SCENARIO_COLORS = [
    COLORS['green_900'],   # Scenario 1 - Dark Green
    COLORS['mint_500'],    # Scenario 2 - Mint
    COLORS['coral_500'],   # Scenario 3 - Coral
    '#5B8DEF',             # Scenario 4 - Blue
    '#9B59B6',             # Scenario 5 - Purple
]

# =============================================================================
# Data Classes
# =============================================================================
class RiskCategory(Enum):
    LOW = 'LOW'
    MEDIUM = 'MEDIUM'
    HIGH = 'HIGH'

class Decision(Enum):
    APPROVE = 'APPROVE'
    APPROVE_ADJUSTED = 'APPROVE_ADJUSTED'
    APPROVE_ESCALATED = 'APPROVE_ESCALATED'  # Approved by human reviewer
    ESCALATE = 'ESCALATE'  # Intermediate state (not final)
    DENY = 'DENY'
    DENY_ESCALATED = 'DENY_ESCALATED'  # Denied by human reviewer

@dataclass
class Applicant:
    id: int
    true_risk: float
    will_default: bool

@dataclass
class AIOutput:
    risk_score: float
    semantic_modifier: float
    base_score: float  # Before delta
    
    @property
    def effective_signal(self):
        # Delta applies as MULTIPLIER
        return self.risk_score * (1 + self.semantic_modifier)

@dataclass
class FinancialConfig:
    """Financial impact parameters for P&L calculation"""
    revenue_per_good_loan: float = 3.0      # Revenue earned from a non-defaulting approved loan
    cost_per_default: float = 8.0           # Loss incurred per defaulting approved loan
    opportunity_cost_per_denial: float = 0.5 # Opportunity cost of denying a non-defaulting applicant

@dataclass
class ScenarioConfig:
    name: str
    n_steps: int = 1200
    n_applicants_per_step: int = 15
    # Delta (ABO)
    delta_enabled: bool = False
    delta_start: int = 500
    delta_end: int = 800
    delta_min: float = -0.15
    delta_max: float = -0.10
    # Feedback loops
    v4_strength: float = 0.076
    v4_asymmetry: float = 0.03  # v24: permissive correction at this fraction of conservative
    v3_strength: float = 0.000110
    v3_proportional: bool = True  # v24: v3 scales with recent approval rate
    v3_lookback: int = 20        # v24: window for computing recent approval rate for v3 scaling
    # Thresholds
    low_thresh: float = 0.38
    high_thresh: float = 0.62
    # Fixed
    maturation_delay: int = 90
    noise_std: float = 0.08
    calibration_window: int = 50
    target_default_rate: float = 0.15
    # Human reviewer noise (v24: replaces omniscient reviewer)
    human_noise_std: float = 0.05
    # Financial impact (v24)
    financial: FinancialConfig = field(default_factory=FinancialConfig)
    # ISCIL Configuration
    iscil_enabled: bool = False
    iscil_containment_enabled: bool = False
    iscil_baseline_window: int = 200
    iscil_threshold: float = 1.0
    iscil_sustained_window: int = 5  # CRS must be above threshold for this many consecutive timesteps
    iscil_roc_window: int = 15  # v23: Window for rate-of-change computation (k)
    iscil_max_guardrail_offset: float = 0.05
    iscil_max_feedback_damping: float = 0.50
    # ISCIL weights for CRS
    iscil_weight_approval: float = 0.4
    iscil_weight_category: float = 0.3
    iscil_weight_escalation: float = 0.2
    iscil_weight_feedback: float = 0.1

# =============================================================================
# Simulation Components
# =============================================================================
def generate_applicant_pool(n_applicants: int, seed: int = 42) -> List[Applicant]:
    """Pre-generate fixed pool of applicants with determined outcomes."""
    rng = np.random.default_rng(seed)
    applicants = []
    for i in range(n_applicants):
        true_risk = rng.beta(2.5, 3.5)
        default_probability = true_risk ** 1.8
        will_default = rng.random() < default_probability
        applicants.append(Applicant(id=i, true_risk=true_risk, will_default=will_default))
    return applicants

class AIRiskAssessment:
    def __init__(self, config: ScenarioConfig, rng: np.random.Generator):
        self.config = config
        self.rng = rng
        self.delta_rng = np.random.default_rng(seed=9999)  # v25: Separate RNG for delta to avoid desyncing noise
        self.calibration_offset = 0.0
    
    def assess(self, applicant: Applicant, current_timestep: int) -> AIOutput:
        noise = self.rng.normal(0, self.config.noise_std)
        base_score = applicant.true_risk + noise
        calibrated = np.clip(base_score + self.calibration_offset, 0, 1)
        
        # Apply delta only in specified interval
        apply_delta = (
            self.config.delta_enabled and 
            self.config.delta_start <= current_timestep < self.config.delta_end
        )
        
        if apply_delta:
            semantic_modifier = self.delta_rng.uniform(self.config.delta_min, self.config.delta_max)
        else:
            semantic_modifier = 0.0
        
        return AIOutput(calibrated, semantic_modifier, base_score)
    
    def receive_outcome_feedback(self, signal: float):
        """
        v4 → v1: Outcome-based feedback with asymmetric correction (v24).
        When signal > 0 (too many defaults → go conservative): full strength.
        When signal < 0 (too few defaults → could be more permissive): reduced strength.
        """
        if self.config.v4_strength > 0:
            if signal < 0:
                adjusted_signal = signal * self.config.v4_asymmetry
            else:
                adjusted_signal = signal
            self.calibration_offset += self.config.v4_strength * adjusted_signal
            self.calibration_offset = np.clip(self.calibration_offset, -0.5, 0.5)
    
    def receive_operational_feedback(self, was_approved: bool, v3_effective_strength: float = None):
        """v3 → v1: Operational feedback. v24: accepts effective strength to avoid config mutation."""
        strength = v3_effective_strength if v3_effective_strength is not None else self.config.v3_strength
        if strength > 0 and was_approved:
            self.calibration_offset -= strength
            self.calibration_offset = np.clip(self.calibration_offset, -0.5, 0.5)

class CategorizationService:
    def __init__(self, config: ScenarioConfig):
        self.config = config
    
    def categorize(self, output: AIOutput) -> RiskCategory:
        signal = output.effective_signal
        if signal < self.config.low_thresh:
            return RiskCategory.LOW
        elif signal < self.config.high_thresh:
            return RiskCategory.MEDIUM
        return RiskCategory.HIGH
    
    def categorize_with_signal(self, signal: float) -> RiskCategory:
        """Categorize using a direct signal value (for ISCIL guardrail)"""
        if signal < self.config.low_thresh:
            return RiskCategory.LOW
        elif signal < self.config.high_thresh:
            return RiskCategory.MEDIUM
        return RiskCategory.HIGH
    
    def categorize_base(self, base_score: float) -> RiskCategory:
        """Categorize without delta (for jump analysis)"""
        if base_score < self.config.low_thresh:
            return RiskCategory.LOW
        elif base_score < self.config.high_thresh:
            return RiskCategory.MEDIUM
        return RiskCategory.HIGH

class DecisionEngine:
    def __init__(self, config: ScenarioConfig, rng: np.random.Generator):
        """
        v24: Human reviewer is now noisy (configurable) instead of omniscient.
        """
        self.human_review_threshold = 0.50
        self.human_noise_std = config.human_noise_std
        self.rng = rng
    
    def decide(self, category: RiskCategory, applicant: Applicant, effective_signal: float) -> Decision:
        if category == RiskCategory.LOW:
            return Decision.APPROVE
        elif category == RiskCategory.MEDIUM:
            if effective_signal < 0.45:
                return Decision.APPROVE_ADJUSTED
            else:
                return self._human_review(applicant)
        else:  # HIGH
            if effective_signal > 0.75:
                return Decision.DENY
            else:
                return self._human_review(applicant)
    
    def _human_review(self, applicant: Applicant) -> Decision:
        """
        v24: Human reviewer is noisy — better than AI but not omniscient.
        Uses true_risk + noise for decision, making escalation imperfect.
        """
        perceived_risk = applicant.true_risk + self.rng.normal(0, self.human_noise_std)
        if perceived_risk < self.human_review_threshold:
            return Decision.APPROVE_ESCALATED
        else:
            return Decision.DENY_ESCALATED

class CalibrationModule:
    def __init__(self, config: ScenarioConfig):
        self.config = config
        self.outcomes = deque(maxlen=config.calibration_window)
    
    def record(self, defaulted: bool):
        self.outcomes.append(defaulted)
    
    def compute_feedback(self) -> float:
        if len(self.outcomes) < 20:
            return 0.0
        current_rate = sum(self.outcomes) / len(self.outcomes)
        return current_rate - self.config.target_default_rate

# =============================================================================
# ISCIL Monitor v23 - Rate-of-Change Based Detection
# =============================================================================
@dataclass
class ISCILResponse:
    """Response from ISCIL containing current state and actions"""
    crs: float = 0.0  # Coherence-Risk Score (now based on rate-of-change)
    intervention_strength: float = 0.0
    guardrail_offset: float = 0.0  # Discretization guardrail (e12)
    v3_damping_factor: float = 1.0  # Feedback damping (e31)
    v4_damping_factor: float = 1.0  # Feedback damping (e41)
    drift_direction: str = 'none'  # 'permissive', 'conservative', or 'none'
    alert_active: bool = False

class ISCILMonitor:
    """
    Inter-System Coherence & Integrity Layer (v24)
    
    Monitors cluster-level telemetry and applies containment actions
    when Coherence-Risk Score exceeds threshold.
    
    v24 fix: Smooths raw telemetry before computing rate-of-change to avoid
    false positives from per-timestep sampling noise (n=15 applicants creates
    ±7% approval rate noise per step).
    
    Telemetry signals (boundary observables only):
    - approval_rate: from e34 (Decision output)
    - category_proportions: from e23 (Categorization output)
    - escalation_rate: from e34 (Decision output)
    - feedback_signal: from e41 (Calibration output)
    
    Containment actions:
    - Discretization guardrails at e12 (blind offset)
    - Feedback damping at e31 and e41 (asymmetric)
    """
    
    SMOOTHING_WINDOW = 10  # Rolling average window for noise reduction
    
    def __init__(self, config: ScenarioConfig):
        self.config = config
        self.baseline_window = config.iscil_baseline_window
        self.threshold = config.iscil_threshold
        self.sustained_window = config.iscil_sustained_window
        self.roc_window = config.iscil_roc_window
        self.max_guardrail = config.iscil_max_guardrail_offset
        self.max_damping = config.iscil_max_feedback_damping
        
        # Weights for CRS
        self.w_approval = config.iscil_weight_approval
        self.w_category = config.iscil_weight_category
        self.w_escalation = config.iscil_weight_escalation
        self.w_feedback = config.iscil_weight_feedback
        
        # State
        self.baseline_established = False
        self.baseline_roc_stats = {}
        self.history = []          # Raw telemetry per timestep
        self.smoothed_history = []  # Smoothed telemetry (rolling avg)
        self.crs_history = []
        self.first_alert_time = None      # v25: First CRS >= threshold (any single breach)
        self.sustained_alert_time = None  # v25: First sustained breach (intervention starts)
        self.alert_time = None            # Legacy: points to first_alert_time for backward compat
        self.intervention_active = False
    
    def _smooth_current(self, signal_name: str, current_value: float) -> float:
        """Return rolling average of signal over SMOOTHING_WINDOW steps."""
        window = self.SMOOTHING_WINDOW
        if len(self.history) < window:
            # Not enough history, average what we have
            vals = [self._extract_signal(h, signal_name) for h in self.history]
            vals.append(current_value)
            return np.mean(vals)
        
        vals = [self._extract_signal(h, signal_name) for h in self.history[-window+1:]]
        vals.append(current_value)
        return np.mean(vals)
    
    def _extract_signal(self, entry: Dict, signal_name: str) -> float:
        """Extract a signal value from a history entry."""
        if signal_name == 'approval_rate':
            return entry.get('approval_rate', 0.5)
        elif signal_name.startswith('category_'):
            cat = signal_name.replace('category_', '')
            return entry.get('category_proportions', {}).get(cat, 0.33)
        elif signal_name == 'escalation_rate':
            return entry.get('escalation_rate', 0.0)
        elif signal_name == 'feedback_magnitude':
            return abs(entry.get('feedback_signal', 0))
        return 0.0
        
    def _compute_rate_of_change(self, signal_name: str, current_smoothed: float) -> float:
        """Compute rate of change over roc_window steps using SMOOTHED values."""
        if len(self.smoothed_history) < self.roc_window:
            return 0.0
        
        older_smoothed = self.smoothed_history[-self.roc_window].get(signal_name, current_smoothed)
        roc = abs(current_smoothed - older_smoothed) / self.roc_window
        return roc
    
    def _compute_roc_z_score(self, roc: float, signal_name: str) -> float:
        """Compute z-score for rate-of-change relative to baseline ROC."""
        if signal_name not in self.baseline_roc_stats:
            return 0.0
        
        mean_roc = self.baseline_roc_stats[signal_name]['mean']
        std_roc = self.baseline_roc_stats[signal_name]['std']
        
        if std_roc <= 0:
            return 0.0
        
        return max(0, (roc - mean_roc)) / std_roc  # Only flag INCREASES in ROC
    
    def _check_sustained_threshold(self, crs: float) -> bool:
        """Check if CRS has been above threshold for sustained_window out of last 2*sustained_window timesteps"""
        self.crs_history.append(crs)
        
        if len(self.crs_history) < self.sustained_window:
            return False
        
        lookback = min(2 * self.sustained_window, len(self.crs_history))
        recent_crs = self.crs_history[-lookback:]
        breaches = sum(1 for c in recent_crs if c >= self.threshold)
        return breaches >= self.sustained_window
    
    def update(self, timestep: int, telemetry: Dict, raw_v4_feedback: float = 0.0) -> ISCILResponse:
        """Update ISCIL with current telemetry and return response."""
        self.history.append({'timestep': timestep, **telemetry})
        
        response = ISCILResponse()
        
        # Phase 1: Baseline establishment
        if timestep <= self.baseline_window:
            # Still compute and store smoothed values during baseline
            smoothed = {}
            for sig in ['approval_rate', 'escalation_rate', 'feedback_magnitude',
                        'category_LOW', 'category_MEDIUM', 'category_HIGH']:
                if sig.startswith('category_'):
                    cat = sig.replace('category_', '')
                    raw = telemetry['category_proportions'].get(cat, 0.33)
                else:
                    raw = abs(raw_v4_feedback) if sig == 'feedback_magnitude' else telemetry.get(sig, 0)
                smoothed[sig] = self._smooth_current(sig, raw)
            self.smoothed_history.append(smoothed)
            
            if timestep == self.baseline_window:
                self._establish_baseline()
            return response
        
        if not self.baseline_established:
            return response
        
        # Phase 2: Compute smoothed values and ROC-based CRS
        smoothed = {}
        for sig in ['approval_rate', 'escalation_rate', 'feedback_magnitude',
                    'category_LOW', 'category_MEDIUM', 'category_HIGH']:
            if sig.startswith('category_'):
                cat = sig.replace('category_', '')
                raw = telemetry['category_proportions'].get(cat, 0.33)
            else:
                raw = abs(raw_v4_feedback) if sig == 'feedback_magnitude' else telemetry.get(sig, 0)
            smoothed[sig] = self._smooth_current(sig, raw)
        self.smoothed_history.append(smoothed)
        
        # Approval rate ROC on smoothed signal
        approval_roc = self._compute_rate_of_change('approval_rate', smoothed['approval_rate'])
        z_approval = self._compute_roc_z_score(approval_roc, 'approval_rate')
        
        # Category ROC: max deviation in any category
        z_category = 0.0
        for cat in ['LOW', 'MEDIUM', 'HIGH']:
            cat_roc = self._compute_rate_of_change(f'category_{cat}', smoothed[f'category_{cat}'])
            z_cat = self._compute_roc_z_score(cat_roc, f'category_{cat}')
            z_category = max(z_category, z_cat)
        
        # Escalation rate ROC
        escalation_roc = self._compute_rate_of_change('escalation_rate', smoothed['escalation_rate'])
        z_escalation = self._compute_roc_z_score(escalation_roc, 'escalation_rate')
        
        # Feedback magnitude ROC
        feedback_roc = self._compute_rate_of_change('feedback_magnitude', smoothed['feedback_magnitude'])
        z_feedback = self._compute_roc_z_score(feedback_roc, 'feedback_magnitude')
        
        # Aggregated CRS
        crs = (self.w_approval * z_approval + 
               self.w_category * z_category + 
               self.w_escalation * z_escalation + 
               self.w_feedback * z_feedback)
        
        response.crs = crs
        
        # Determine drift direction (using smoothed values)
        if len(self.smoothed_history) >= self.roc_window:
            older_approval = self.smoothed_history[-self.roc_window].get('approval_rate', smoothed['approval_rate'])
            current_approval = smoothed['approval_rate']
            
            if current_approval > older_approval + 0.01:
                response.drift_direction = 'permissive'
            elif current_approval < older_approval - 0.01:
                response.drift_direction = 'conservative'
            else:
                response.drift_direction = 'none'
        
        if crs >= self.threshold:
            response.alert_active = True
            # v25: Track first single-timestep threshold breach
            if self.first_alert_time is None:
                self.first_alert_time = timestep
                self.alert_time = timestep  # Legacy compat
        
        # Check sustained threshold for intervention
        sustained_breach = self._check_sustained_threshold(crs)
        
        if sustained_breach and not self.intervention_active:
            self.intervention_active = True
            # v25: Track FIRST time intervention starts (don't overwrite on re-activation)
            if self.sustained_alert_time is None:
                self.sustained_alert_time = timestep
        
        # End intervention when CRS drops below threshold for sustained period
        if self.intervention_active and crs < self.threshold:
            recent_crs = self.crs_history[-self.sustained_window:] if len(self.crs_history) >= self.sustained_window else self.crs_history
            if all(c < self.threshold for c in recent_crs):
                self.intervention_active = False
        
        # Phase 3: Containment (if enabled AND intervention active)
        if not self.config.iscil_containment_enabled:
            return response
        
        if not self.intervention_active:
            return response
        
        # Compute intervention strength based on current CRS
        if crs > self.threshold:
            strength = min(np.sqrt((crs - self.threshold) / self.threshold), 1.0)
            response.intervention_strength = strength
            
            # Action 1: Discretization Guardrails (e12)
            if response.drift_direction == 'permissive':
                response.guardrail_offset = strength * self.max_guardrail
            elif response.drift_direction == 'conservative':
                response.guardrail_offset = -strength * self.max_guardrail
            
            # Action 2: Feedback Damping (asymmetric)
            if response.drift_direction == 'permissive':
                response.v3_damping_factor = 1.0 - (strength * self.max_damping)
                response.v4_damping_factor = 1.0 - (strength * self.max_damping)
            elif response.drift_direction == 'conservative':
                response.v3_damping_factor = 1.0 + (strength * 0.3)
                response.v4_damping_factor = 1.0 + (strength * 0.3)
        
        return response
    
    def _establish_baseline(self):
        """
        v24: Compute baseline rate-of-change statistics from SMOOTHED history.
        Uses a generous std floor to prevent oversensitivity.
        """
        if len(self.smoothed_history) < self.roc_window + 1:
            return
        
        recent = self.smoothed_history
        
        def compute_baseline_rocs(signal_name):
            rocs = []
            for i in range(self.roc_window, len(recent)):
                current = recent[i].get(signal_name, 0)
                older = recent[i - self.roc_window].get(signal_name, 0)
                roc = abs(current - older) / self.roc_window
                rocs.append(roc)
            return rocs
        
        for sig in ['approval_rate', 'escalation_rate', 'feedback_magnitude',
                    'category_LOW', 'category_MEDIUM', 'category_HIGH']:
            rocs = compute_baseline_rocs(sig)
            if rocs:
                mean_roc = np.mean(rocs)
                std_roc = np.std(rocs)
                # Floor: std must be at least 30% of mean to prevent oversensitivity
                # when baseline is unusually smooth
                std_floor = max(0.3 * mean_roc, 0.002)
                self.baseline_roc_stats[sig] = {
                    'mean': mean_roc,
                    'std': max(std_roc, std_floor)
                }
            else:
                self.baseline_roc_stats[sig] = {'mean': 0.0, 'std': 0.01}
        
        self.baseline_established = True

# =============================================================================
# Simulation Results
# =============================================================================
@dataclass
class SimulationResults:
    # Per-timestep metrics
    timestep_approval_rate: List[float] = field(default_factory=list)
    timestep_default_rate: List[Optional[float]] = field(default_factory=list)
    calibration_offset_history: List[float] = field(default_factory=list)
    delta_history: List[float] = field(default_factory=list)  # Track delta values
    
    # Cumulative metrics
    cumulative_approval_rate: List[float] = field(default_factory=list)
    cumulative_default_rate: List[float] = field(default_factory=list)
    
    # Cumulative counts
    cumulative_approval_count: List[int] = field(default_factory=list)
    cumulative_default_count: List[int] = field(default_factory=list)
    
    # Per-timestep counts
    timestep_approval_count: List[int] = field(default_factory=list)
    timestep_default_count: List[int] = field(default_factory=list)
    
    # Jump analysis data
    jump_events: List[Dict] = field(default_factory=list)
    effective_signals: List[float] = field(default_factory=list)
    base_scores: List[float] = field(default_factory=list)
    
    # ISCIL metrics
    iscil_crs_history: List[float] = field(default_factory=list)
    iscil_intervention_strength_history: List[float] = field(default_factory=list)
    iscil_guardrail_offset_history: List[float] = field(default_factory=list)
    iscil_drift_direction_history: List[str] = field(default_factory=list)
    iscil_alert_time: Optional[int] = None
    iscil_sustained_alert_time: Optional[int] = None  # v25: When sustained intervention starts
    iscil_total_intervention_timesteps: int = 0
    iscil_peak_crs: float = 0.0
    
    # Summary
    total_approvals: int = 0
    total_matured_approvals: int = 0  # Only approvals that have matured
    total_denials: int = 0
    total_defaults: int = 0
    all_outcomes: List[Dict] = field(default_factory=list)
    
    # Closed-window metrics (only loans where full maturation cycle completed)
    closed_window_approvals: int = 0      # Approvals at t <= (n_steps - maturation_delay)
    closed_window_applicants: int = 0     # Applicants at t <= (n_steps - maturation_delay)
    closed_window_defaults: int = 0       # Defaults from closed-window approvals
    
    # Decision type counts
    count_approve: int = 0
    count_approve_adjusted: int = 0
    count_approve_escalated: int = 0
    count_deny: int = 0
    count_deny_escalated: int = 0
    
    # v24: Financial impact
    timestep_pnl: List[float] = field(default_factory=list)
    cumulative_pnl: List[float] = field(default_factory=list)
    total_revenue: float = 0.0
    total_default_losses: float = 0.0
    total_opportunity_cost: float = 0.0
    
    # v24: Confusion matrix (accumulated at end from closed-window matured loans)
    true_positives: int = 0   # Correctly denied defaulters
    true_negatives: int = 0   # Correctly approved non-defaulters
    false_positives: int = 0  # Denied non-defaulters (opportunity cost)
    false_negatives: int = 0  # Approved defaulters (losses)
    
    # v24: v3 effective strength history
    v3_effective_history: List[float] = field(default_factory=list)
    
    # v0.1: Feedback decomposition — cumulative v3 and v4 contributions
    v3_cumulative_contribution: List[float] = field(default_factory=list)
    v4_cumulative_contribution: List[float] = field(default_factory=list)
    # v0.1: Per-timestep v3 and v4 deltas (raw signal before clipping)
    v3_per_step_delta: List[float] = field(default_factory=list)
    v4_per_step_delta: List[float] = field(default_factory=list)

# =============================================================================
# Main Simulation
# =============================================================================
class UnderwritingISE:
    def __init__(self, config: ScenarioConfig, applicant_pool: List[Applicant], seed: int = 123):
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.applicant_pool = applicant_pool
        
        self.ai = AIRiskAssessment(config, self.rng)
        self.categorization = CategorizationService(config)
        self.decision_engine = DecisionEngine(config, self.rng)
        self.calibration = CalibrationModule(config)
        
        self.iscil = ISCILMonitor(config) if config.iscil_enabled else None
        self.current_iscil_response = ISCILResponse()
        
        self.timestep = 0
        self.applicant_index = 0
        self.outcomes = []
        self.results = SimulationResults()
        
        # v24: Maturation queue - O(1) per timestep
        self.maturation_queue: Dict[int, List[Dict]] = {}
        # v24: Recent approval rates for proportional v3
        self.recent_approval_rates = deque(maxlen=config.v3_lookback)
        # v24: Running counters
        self._running_total_approvals = 0
        self._running_total_decisions = 0
        self._running_matured_approvals = 0
        self._running_total_defaults = 0
        self._running_pnl = 0.0
        # v0.1: Feedback decomposition running counters
        self._v3_cumulative = 0.0
        self._v4_cumulative = 0.0
    
    def _collect_telemetry(self, timestep_outcomes):
        if not timestep_outcomes:
            return {'approval_rate': 0.0, 'category_proportions': {'LOW': 0.33, 'MEDIUM': 0.34, 'HIGH': 0.33},
                    'escalation_rate': 0.0, 'feedback_signal': 0.0}
        n = len(timestep_outcomes)
        approvals = sum(1 for o in timestep_outcomes if o['decision'] in [Decision.APPROVE, Decision.APPROVE_ADJUSTED, Decision.APPROVE_ESCALATED])
        cat_counts = {'LOW': 0, 'MEDIUM': 0, 'HIGH': 0}
        for o in timestep_outcomes:
            cat = o.get('category', 'MEDIUM')
            if cat in cat_counts: cat_counts[cat] += 1
        escalations = sum(1 for o in timestep_outcomes if o['decision'] in [Decision.APPROVE_ESCALATED, Decision.DENY_ESCALATED])
        return {'approval_rate': approvals / n, 'category_proportions': {k: v/n for k, v in cat_counts.items()},
                'escalation_rate': escalations / n, 'feedback_signal': self.calibration.compute_feedback()}
    
    def _compute_v3_effective_strength(self):
        base = self.config.v3_strength
        if not self.config.v3_proportional or not self.recent_approval_rates:
            return base
        rate = np.mean(self.recent_approval_rates)
        scale = max(0.5, min(2.0, 1.0 + (rate - 0.5)))
        return base * scale
    
    def step(self):
        self.timestep += 1
        timestep_outcomes = []
        timestep_deltas = []
        guardrail_offset = self.current_iscil_response.guardrail_offset if self.iscil else 0.0
        # v0.1: Per-timestep feedback delta accumulators
        v3_step_delta = 0.0
        
        # v24: v3 effective strength with optional ISCIL damping
        v3_eff = self._compute_v3_effective_strength()
        if self.iscil and self.current_iscil_response.v3_damping_factor != 1.0:
            v3_eff *= self.current_iscil_response.v3_damping_factor
        self.results.v3_effective_history.append(v3_eff)
        
        for _ in range(self.config.n_applicants_per_step):
            if self.applicant_index >= len(self.applicant_pool):
                break
            applicant = self.applicant_pool[self.applicant_index]
            self.applicant_index += 1
            ai_output = self.ai.assess(applicant, self.timestep)
            
            if guardrail_offset != 0:
                modulated_signal = np.clip(ai_output.effective_signal + guardrail_offset, 0, 1)
                category_with_delta = self.categorization.categorize_with_signal(modulated_signal)
            else:
                modulated_signal = ai_output.effective_signal
                category_with_delta = self.categorization.categorize(ai_output)
            
            category_without_delta = self.categorization.categorize_base(ai_output.risk_score)
            decision = self.decision_engine.decide(category_with_delta, applicant, modulated_signal)
            timestep_deltas.append(ai_output.semantic_modifier)
            
            # v24: v3 feedback without config mutation
            was_approved = decision in [Decision.APPROVE, Decision.APPROVE_ADJUSTED, Decision.APPROVE_ESCALATED]
            # v0.1: Track v3 contribution before applying (so we know exact delta)
            if v3_eff > 0 and was_approved:
                self._v3_cumulative -= v3_eff
                v3_step_delta -= v3_eff
            self.ai.receive_operational_feedback(was_approved, v3_effective_strength=v3_eff)
            
            if category_with_delta != category_without_delta:
                self.results.jump_events.append({
                    'timestep': self.timestep, 'base_score': ai_output.risk_score,
                    'effective_signal': ai_output.effective_signal, 'delta': ai_output.semantic_modifier,
                    'category_before': category_without_delta, 'category_after': category_with_delta,
                    'true_risk': applicant.true_risk, 'will_default': applicant.will_default, 'decision': decision,
                })
            
            self.results.effective_signals.append(ai_output.effective_signal)
            self.results.base_scores.append(ai_output.risk_score)
            
            outcome = {
                'timestep': self.timestep, 'applicant_id': applicant.id,
                'true_risk': applicant.true_risk, 'will_default': applicant.will_default,
                'decision': decision, 'effective_signal': ai_output.effective_signal,
                'base_score': ai_output.risk_score, 'delta': ai_output.semantic_modifier,
                'category': category_with_delta.value,
            }
            self.outcomes.append(outcome)
            timestep_outcomes.append(outcome)
            
            # v24: Schedule maturation via queue
            if was_approved:
                mat_t = self.timestep + self.config.maturation_delay
                if mat_t not in self.maturation_queue:
                    self.maturation_queue[mat_t] = []
                self.maturation_queue[mat_t].append(outcome)
        
        self.results.delta_history.append(np.mean(timestep_deltas) if timestep_deltas else 0.0)
        
        approvals_this_step = sum(1 for o in timestep_outcomes if o['decision'] in [Decision.APPROVE, Decision.APPROVE_ADJUSTED, Decision.APPROVE_ESCALATED])
        approval_rate = approvals_this_step / len(timestep_outcomes) if timestep_outcomes else 0
        self.results.timestep_approval_rate.append(approval_rate)
        self.results.timestep_approval_count.append(approvals_this_step)
        self.recent_approval_rates.append(approval_rate)
        self._running_total_approvals += approvals_this_step
        self._running_total_decisions += len(timestep_outcomes)
        
        # v24: Process matured loans via queue (O(1))
        defaults_this_step = 0
        matured_this_step = 0
        pnl_this_step = 0.0
        
        if self.timestep in self.maturation_queue:
            for outcome in self.maturation_queue[self.timestep]:
                outcome['defaulted'] = outcome['will_default']
                self.calibration.record(outcome['defaulted'])
                matured_this_step += 1
                self._running_matured_approvals += 1
                if outcome['defaulted']:
                    defaults_this_step += 1
                    self._running_total_defaults += 1
                    pnl_this_step -= self.config.financial.cost_per_default
                else:
                    pnl_this_step += self.config.financial.revenue_per_good_loan
            del self.maturation_queue[self.timestep]
        
        for o in timestep_outcomes:
            if o['decision'] in [Decision.DENY, Decision.DENY_ESCALATED] and not o['will_default']:
                pnl_this_step -= self.config.financial.opportunity_cost_per_denial
        
        self.results.timestep_default_rate.append(defaults_this_step / matured_this_step if matured_this_step > 0 else None)
        self.results.timestep_default_count.append(defaults_this_step)
        self._running_pnl += pnl_this_step
        self.results.timestep_pnl.append(pnl_this_step)
        self.results.cumulative_pnl.append(self._running_pnl)
        
        # v4 feedback with ISCIL damping
        feedback_signal = self.calibration.compute_feedback()
        if self.iscil and self.current_iscil_response.v4_damping_factor != 1.0:
            if self.current_iscil_response.drift_direction == 'permissive':
                if feedback_signal < 0: feedback_signal *= self.current_iscil_response.v4_damping_factor
                else: feedback_signal *= (2.0 - self.current_iscil_response.v4_damping_factor)
            elif self.current_iscil_response.drift_direction == 'conservative':
                if feedback_signal > 0: feedback_signal *= self.current_iscil_response.v4_damping_factor
                else: feedback_signal *= (2.0 - self.current_iscil_response.v4_damping_factor)
        
        self.ai.receive_outcome_feedback(feedback_signal)
        # v0.1: Track v4 contribution (same logic as receive_outcome_feedback)
        v4_step_delta = 0.0
        if self.config.v4_strength > 0:
            if feedback_signal < 0:
                v4_step_delta = self.config.v4_strength * feedback_signal * self.config.v4_asymmetry
            else:
                v4_step_delta = self.config.v4_strength * feedback_signal
            self._v4_cumulative += v4_step_delta
        self.results.v3_cumulative_contribution.append(self._v3_cumulative)
        self.results.v4_cumulative_contribution.append(self._v4_cumulative)
        self.results.v3_per_step_delta.append(v3_step_delta)
        self.results.v4_per_step_delta.append(v4_step_delta)
        self.results.calibration_offset_history.append(self.ai.calibration_offset)
        
        # ISCIL update
        if self.iscil:
            telemetry = self._collect_telemetry(timestep_outcomes)
            self.current_iscil_response = self.iscil.update(self.timestep, telemetry, self.calibration.compute_feedback())
            self.results.iscil_crs_history.append(self.current_iscil_response.crs)
            self.results.iscil_intervention_strength_history.append(self.current_iscil_response.intervention_strength)
            self.results.iscil_guardrail_offset_history.append(self.current_iscil_response.guardrail_offset)
            self.results.iscil_drift_direction_history.append(self.current_iscil_response.drift_direction)
            if self.current_iscil_response.crs > self.results.iscil_peak_crs:
                self.results.iscil_peak_crs = self.current_iscil_response.crs
            if self.current_iscil_response.intervention_strength > 0:
                self.results.iscil_total_intervention_timesteps += 1
        else:
            self.results.iscil_crs_history.append(0.0)
            self.results.iscil_intervention_strength_history.append(0.0)
            self.results.iscil_guardrail_offset_history.append(0.0)
            self.results.iscil_drift_direction_history.append('none')
        
        # v24: Cumulative metrics (incremental)
        self.results.cumulative_approval_rate.append(self._running_total_approvals / self._running_total_decisions if self._running_total_decisions > 0 else 0)
        self.results.cumulative_default_rate.append(self._running_total_defaults / self._running_matured_approvals if self._running_matured_approvals > 0 else 0)
        self.results.cumulative_approval_count.append(self._running_total_approvals)
        self.results.cumulative_default_count.append(self._running_total_defaults)
    
    def run(self) -> SimulationResults:
        for _ in range(self.config.n_steps):
            self.step()
        
        all_approvals = [o for o in self.outcomes if o['decision'] in [Decision.APPROVE, Decision.APPROVE_ADJUSTED, Decision.APPROVE_ESCALATED]]
        all_denials = [o for o in self.outcomes if o['decision'] in [Decision.DENY, Decision.DENY_ESCALATED]]
        matured_approvals = [o for o in all_approvals if 'defaulted' in o]
        
        self.results.total_approvals = len(all_approvals)
        self.results.total_matured_approvals = len(matured_approvals)
        self.results.total_denials = len(all_denials)
        self.results.total_defaults = sum(1 for o in matured_approvals if o['defaulted'])
        self.results.all_outcomes = self.outcomes
        
        cutoff = self.config.n_steps - self.config.maturation_delay
        cw_outcomes = [o for o in self.outcomes if o['timestep'] <= cutoff]
        cw_approvals = [o for o in cw_outcomes if o['decision'] in [Decision.APPROVE, Decision.APPROVE_ADJUSTED, Decision.APPROVE_ESCALATED]]
        cw_denials = [o for o in cw_outcomes if o['decision'] in [Decision.DENY, Decision.DENY_ESCALATED]]
        
        self.results.closed_window_applicants = len(cw_outcomes)
        self.results.closed_window_approvals = len(cw_approvals)
        self.results.closed_window_defaults = sum(1 for o in cw_approvals if o.get('defaulted', False))
        
        self.results.count_approve = sum(1 for o in self.outcomes if o['decision'] == Decision.APPROVE)
        self.results.count_approve_adjusted = sum(1 for o in self.outcomes if o['decision'] == Decision.APPROVE_ADJUSTED)
        self.results.count_approve_escalated = sum(1 for o in self.outcomes if o['decision'] == Decision.APPROVE_ESCALATED)
        self.results.count_deny = sum(1 for o in self.outcomes if o['decision'] == Decision.DENY)
        self.results.count_deny_escalated = sum(1 for o in self.outcomes if o['decision'] == Decision.DENY_ESCALATED)
        
        if self.iscil and self.iscil.alert_time:
            self.results.iscil_alert_time = self.iscil.alert_time
        if self.iscil and self.iscil.sustained_alert_time:
            self.results.iscil_sustained_alert_time = self.iscil.sustained_alert_time
        
        # v24: Confusion matrix
        for o in cw_approvals:
            if o.get('defaulted', False): self.results.false_negatives += 1
            else: self.results.true_negatives += 1
        for o in cw_denials:
            if o['will_default']: self.results.true_positives += 1
            else: self.results.false_positives += 1
        
        # v24: Financial totals
        self.results.total_revenue = self.results.true_negatives * self.config.financial.revenue_per_good_loan
        self.results.total_default_losses = self.results.false_negatives * self.config.financial.cost_per_default
        self.results.total_opportunity_cost = self.results.false_positives * self.config.financial.opportunity_cost_per_denial
        
        return self.results

# =============================================================================
# Plotting Functions
# =============================================================================
def rolling_average(data: List, window: int = 20) -> List:
    """Compute rolling average, handling None values."""
    result = []
    for i in range(len(data)):
        start = max(0, i - window + 1)
        window_data = [d for d in data[start:i+1] if d is not None]
        if window_data:
            result.append(sum(window_data) / len(window_data))
        else:
            result.append(None)
    return result

def plot_scenario_settings(scenarios: Dict[str, Tuple[ScenarioConfig, SimulationResults]]) -> plt.Figure:
    """Figure 1b: Scenario Settings Table"""
    fig, ax = plt.subplots(figsize=(16, 4 + len(scenarios) * 0.6))
    ax.axis('off')
    
    # Prepare data
    columns = ['Scenario', 'δ Enabled', 'δ Start', 'δ End', 'δ Min', 'δ Max', 
               'v3 Strength', 'v4 Strength', 'LOW/MED\nThresh', 'MED/HIGH\nThresh',
               'Steps', 'Applicants\n/Step', 'Maturation\nDelay']
    
    rows = []
    
    for name, (config, results) in scenarios.items():
        rows.append([
            name,
            '✓' if config.delta_enabled else '✗',
            str(config.delta_start) if config.delta_enabled else '-',
            str(config.delta_end) if config.delta_enabled else '-',
            f'{config.delta_min:.2f}' if config.delta_enabled else '-',
            f'{config.delta_max:.2f}' if config.delta_enabled else '-',
            f'{config.v3_strength:.6f}',
            f'{config.v4_strength:.2f}',
            f'{config.low_thresh:.2f}',
            f'{config.high_thresh:.2f}',
            str(config.n_steps),
            str(config.n_applicants_per_step),
            str(config.maturation_delay),
        ])
    
    # Create table
    table = ax.table(
        cellText=rows,
        colLabels=columns,
        loc='upper center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 2.0)
    
    # Style header
    for i in range(len(columns)):
        table[(0, i)].set_facecolor(COLORS['green_900'])
        table[(0, i)].set_text_props(color='white', fontweight='bold', fontsize=8)
        table[(0, i)].set_height(0.18)
    
    # Style data rows
    for row_idx in range(len(rows)):
        for col_idx in range(len(columns)):
            table[(row_idx + 1, col_idx)].set_facecolor(COLORS['silver_100'])
            table[(row_idx + 1, col_idx)].set_height(0.10)
            # Highlight delta columns if enabled
            if col_idx in [1, 2, 3, 4, 5]:
                config = list(scenarios.values())[row_idx][0]
                if config.delta_enabled:
                    table[(row_idx + 1, col_idx)].set_facecolor(COLORS['coral_300'])
    
    ax.set_title('Scenario Settings', fontsize=14, fontweight='bold', color=COLORS['text_primary'], pad=20, loc='left')
    
    # Add legend
    legend_text = """
    ─────────────────────────────────────────────────────────────────────────────────────────────────────────
    SETTINGS LEGEND:
    
    • δ (Delta/ABO): Semantic modifier applied during δ window. Formula: effective_signal = risk_score × (1 + δ)
      → Negative δ = permissive (lowers perceived risk) | Positive δ = conservative (raises perceived risk)
    • v3 Strength: Operational feedback - per approval, offset decreases by this amount (permissive pressure)
    • v4 Strength: Outcome feedback - offset adjusts by v4 × (default_rate - 0.15) to maintain 15% target
    • Thresholds: LOW/MED at {low}, MED/HIGH at {high} - determines risk categorization boundaries
    • Maturation Delay: Timesteps before approved loan outcome (default/no-default) is observed
    """.format(low=list(scenarios.values())[0][0].low_thresh, 
               high=list(scenarios.values())[0][0].high_thresh)
    
    fig.text(0.02, 0.02, legend_text, fontsize=9, color=COLORS['text_muted'], 
             verticalalignment='bottom', family='monospace')
    
    plt.tight_layout(rect=[0, 0.25, 1, 1])
    return fig

def plot_summary_table(scenarios: Dict[str, Tuple[ScenarioConfig, SimulationResults]], 
                       initial_window: int = 50) -> plt.Figure:
    """Figure 1: Summary Metrics - 3 Tables"""
    n_scenarios = len(scenarios)
    # Increase figure height based on number of scenarios
    fig_height = max(10, 3 + n_scenarios * 3.5)
    fig, axes = plt.subplots(3, 1, figsize=(16, fig_height))
    
    for ax in axes:
        ax.axis('off')
    
    # =========================================================================
    # TABLE 1: Closed-Window Results (All loans have completed maturation)
    # Only includes applicants from t=1 to t=(n_steps - maturation_delay)
    # In this window: Approvals = Matured (all approved loans have had time to mature)
    # =========================================================================
    columns1 = ['Scenario', 'CW\nApplicants', 'CW\nApprovals', 'CW\nMatured', 
                'Approval\nRate', 'Default Rate\n(of Matured)', 'Default Rate\n(of Approved)', 'Default Rate\n(of Total)',
                'CW\nDefaults']
    
    rows1 = []
    for name, (config, results) in scenarios.items():
        # In closed window: applicants = those in first (n_steps - maturation_delay) timesteps
        cw_applicants = results.closed_window_applicants
        cw_approvals = results.closed_window_approvals
        cw_matured = results.closed_window_approvals  # In CW, all approvals have matured
        cw_defaults = results.closed_window_defaults
        
        cw_approval_rate = (cw_approvals / cw_applicants * 100) if cw_applicants > 0 else 0
        cw_default_of_matured = (cw_defaults / cw_matured * 100) if cw_matured > 0 else 0
        cw_default_of_approved = (cw_defaults / cw_approvals * 100) if cw_approvals > 0 else 0
        cw_default_of_total = (cw_defaults / cw_applicants * 100) if cw_applicants > 0 else 0
        
        rows1.append([
            name,
            f'{cw_applicants:,}',
            f'{cw_approvals:,}',
            f'{cw_matured:,}',
            f'{cw_approval_rate:.1f}%',
            f'{cw_default_of_matured:.1f}%',
            f'{cw_default_of_approved:.1f}%',
            f'{cw_default_of_total:.1f}%',
            f'{cw_defaults:,}'
        ])
    
    table1 = axes[0].table(cellText=rows1, colLabels=columns1, loc='center', cellLoc='center')
    table1.auto_set_font_size(False)
    table1.set_fontsize(9)
    table1.scale(1.1, 1.5 + 0.1 * n_scenarios)
    
    for i in range(len(columns1)):
        table1[(0, i)].set_facecolor(COLORS['green_900'])
        table1[(0, i)].set_text_props(color='white', fontweight='bold', fontsize=8)
        table1[(0, i)].set_height(0.15)
    for row_idx in range(len(rows1)):
        for col_idx in range(len(columns1)):
            table1[(row_idx + 1, col_idx)].set_facecolor(COLORS['silver_100'])
            table1[(row_idx + 1, col_idx)].set_height(0.12)
    
    cutoff = list(scenarios.values())[0][0].n_steps - list(scenarios.values())[0][0].maturation_delay
    axes[0].set_title(f'Table 1: Closed-Window (t ≤ {cutoff}) — All approvals have matured', 
                      fontsize=11, fontweight='bold', color=COLORS['text_primary'], pad=5, loc='left')
    
    # =========================================================================
    # TABLE 2: Full Simulation Results (Includes unmatured recent approvals)
    # All applicants from t=1 to t=n_steps
    # Approvals > Matured because recent approvals haven't had time to mature
    # =========================================================================
    columns2 = ['Scenario', 'Total\nApplicants', 'Total\nApprovals', 'Total\nMatured', 
                'Approval\nRate', 'Default Rate\n(of Matured)', 'Default Rate\n(of Approved)', 'Default Rate\n(of Total)',
                'Total\nDefaults']
    
    rows2 = []
    for name, (config, results) in scenarios.items():
        total_applicants = config.n_steps * config.n_applicants_per_step
        approval_rate = (results.total_approvals / total_applicants * 100) if total_applicants > 0 else 0
        default_of_matured = (results.total_defaults / results.total_matured_approvals * 100) if results.total_matured_approvals > 0 else 0
        default_of_approved = (results.total_defaults / results.total_approvals * 100) if results.total_approvals > 0 else 0
        default_of_total = (results.total_defaults / total_applicants * 100) if total_applicants > 0 else 0
        
        rows2.append([
            name,
            f'{total_applicants:,}',
            f'{results.total_approvals:,}',
            f'{results.total_matured_approvals:,}',
            f'{approval_rate:.1f}%',
            f'{default_of_matured:.1f}%',
            f'{default_of_approved:.1f}%',
            f'{default_of_total:.1f}%',
            f'{results.total_defaults:,}'
        ])
    
    table2 = axes[1].table(cellText=rows2, colLabels=columns2, loc='center', cellLoc='center')
    table2.auto_set_font_size(False)
    table2.set_fontsize(9)
    table2.scale(1.1, 1.5 + 0.1 * n_scenarios)
    
    for i in range(len(columns2)):
        table2[(0, i)].set_facecolor(COLORS['mint_600'])
        table2[(0, i)].set_text_props(color='white', fontweight='bold', fontsize=8)
        table2[(0, i)].set_height(0.15)
    for row_idx in range(len(rows2)):
        for col_idx in range(len(columns2)):
            table2[(row_idx + 1, col_idx)].set_facecolor(COLORS['silver_100'])
            table2[(row_idx + 1, col_idx)].set_height(0.12)
    
    axes[1].set_title('Table 2: Full Simulation (all timesteps) — Includes unmatured recent approvals', 
                      fontsize=11, fontweight='bold', color=COLORS['text_primary'], pad=5, loc='left')
    
    # =========================================================================
    # TABLE 3: Time Dynamics
    # =========================================================================
    columns3 = ['Scenario', 'Initial\nApproval', 'Final\nApproval', 'Drift',
                'Initial\nDefault', 'Final\nDefault', 'Final\nOffset']
    
    rows3 = []
    for name, (config, results) in scenarios.items():
        n = len(results.timestep_approval_rate)
        
        initial_approval = np.mean(results.timestep_approval_rate[:initial_window]) * 100 if n >= initial_window else 0
        final_approval = np.mean(results.timestep_approval_rate[-initial_window:]) * 100 if n >= initial_window else 0
        drift = final_approval - initial_approval
        
        valid_defaults = [d for d in results.timestep_default_rate if d is not None]
        if len(valid_defaults) >= initial_window:
            initial_default = np.mean(valid_defaults[:initial_window]) * 100
            final_default = np.mean(valid_defaults[-initial_window:]) * 100
        else:
            initial_default = 0
            final_default = 0
        
        final_offset = results.calibration_offset_history[-1] if results.calibration_offset_history else 0
        
        rows3.append([
            name,
            f'{initial_approval:.1f}%',
            f'{final_approval:.1f}%',
            f'{drift:+.1f}%',
            f'{initial_default:.1f}%',
            f'{final_default:.1f}%',
            f'{final_offset:.3f}'
        ])
    
    table3 = axes[2].table(cellText=rows3, colLabels=columns3, loc='center', cellLoc='center')
    table3.auto_set_font_size(False)
    table3.set_fontsize(9)
    table3.scale(1.1, 1.5 + 0.1 * n_scenarios)
    
    for i in range(len(columns3)):
        table3[(0, i)].set_facecolor(COLORS['coral_500'])
        table3[(0, i)].set_text_props(color='white', fontweight='bold', fontsize=8)
        table3[(0, i)].set_height(0.15)
    for row_idx in range(len(rows3)):
        for col_idx in range(len(columns3)):
            table3[(row_idx + 1, col_idx)].set_facecolor(COLORS['silver_100'])
            table3[(row_idx + 1, col_idx)].set_height(0.12)
    
    axes[2].set_title(f'Table 3: Time Dynamics (window={initial_window} timesteps)', 
                      fontsize=11, fontweight='bold', color=COLORS['text_primary'], pad=5, loc='left')
    
    plt.tight_layout()
    return fig

def plot_decision_breakdown(scenarios: Dict[str, Tuple[ScenarioConfig, SimulationResults]]) -> plt.Figure:
    """Figure 1b: Decision Type Breakdown Histogram"""
    n_scenarios = len(scenarios)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    scenario_names = list(scenarios.keys())
    
    # Approval types
    approve_counts = [scenarios[name][1].count_approve for name in scenario_names]
    approve_adj_counts = [scenarios[name][1].count_approve_adjusted for name in scenario_names]
    approve_esc_counts = [scenarios[name][1].count_approve_escalated for name in scenario_names]
    total_approve_counts = [scenarios[name][1].total_approvals for name in scenario_names]
    
    # Denial types
    deny_counts = [scenarios[name][1].count_deny for name in scenario_names]
    deny_esc_counts = [scenarios[name][1].count_deny_escalated for name in scenario_names]
    total_deny_counts = [scenarios[name][1].total_denials for name in scenario_names]
    
    x = np.arange(n_scenarios)
    width = 0.2
    
    # Left plot: Approvals
    ax1 = axes[0]
    bars1 = ax1.bar(x - width*1.5, approve_counts, width, label='APPROVE', color=COLORS['mint_500'], edgecolor='white')
    bars2 = ax1.bar(x - width*0.5, approve_adj_counts, width, label='APPROVE_ADJUSTED', color=COLORS['mint_400'], edgecolor='white')
    bars3 = ax1.bar(x + width*0.5, approve_esc_counts, width, label='APPROVE_ESCALATED', color=COLORS['mint_300'], edgecolor='white')
    bars4 = ax1.bar(x + width*1.5, total_approve_counts, width, label='TOTAL APPROVALS', color=COLORS['green_700'], edgecolor='white')
    
    ax1.set_xlabel('Scenario', fontsize=11, color=COLORS['text_primary'])
    ax1.set_ylabel('Count', fontsize=11, color=COLORS['text_primary'])
    ax1.set_title('A. Approval Decisions by Type', fontsize=12, fontweight='bold', color=COLORS['text_primary'])
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenario_names, fontsize=10)
    ax1.legend(fontsize=9, loc='upper right')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{int(height):,}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, rotation=90)
    
    # Right plot: Denials
    ax2 = axes[1]
    width_deny = 0.25
    bars5 = ax2.bar(x - width_deny, deny_counts, width_deny, label='DENY', color=COLORS['coral_500'], edgecolor='white')
    bars6 = ax2.bar(x, deny_esc_counts, width_deny, label='DENY_ESCALATED', color=COLORS['coral_400'], edgecolor='white')
    bars7 = ax2.bar(x + width_deny, total_deny_counts, width_deny, label='TOTAL DENIALS', color=COLORS['coral_600'], edgecolor='white')
    
    ax2.set_xlabel('Scenario', fontsize=11, color=COLORS['text_primary'])
    ax2.set_ylabel('Count', fontsize=11, color=COLORS['text_primary'])
    ax2.set_title('B. Denial Decisions by Type', fontsize=12, fontweight='bold', color=COLORS['text_primary'])
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenario_names, fontsize=10)
    ax2.legend(fontsize=9, loc='upper right')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars5, bars6, bars7]:
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f'{int(height):,}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, rotation=90)
    
    # Add description
    description = """
    DECISION TYPES:
    • APPROVE: Auto-approved (LOW risk, effective_signal < 0.38)
    • APPROVE_ADJUSTED: Approved within MEDIUM (effective_signal < 0.45)
    • APPROVE_ESCALATED: Human-approved after review (true_risk < 0.50)
    • DENY: Auto-denied (HIGH risk, effective_signal > 0.75)
    • DENY_ESCALATED: Human-denied after review (true_risk ≥ 0.50)
    """
    fig.text(0.02, 0.02, description, fontsize=9, color=COLORS['text_muted'],
             verticalalignment='bottom', family='monospace')
    
    plt.tight_layout(rect=[0, 0.12, 1, 1])
    return fig

def plot_applicant_pool(applicant_pool: List[Applicant]) -> plt.Figure:
    """Figure 2: Applicant Pool Distribution with descriptions"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    true_risks = [a.true_risk for a in applicant_pool]
    will_defaults = [a.will_default for a in applicant_pool]
    
    # Summary stats
    total_applicants = len(applicant_pool)
    total_defaulters = sum(will_defaults)
    avg_default_rate = total_defaulters / total_applicants * 100
    
    # A. True Risk Histogram
    ax1 = axes[0, 0]
    ax1.hist(true_risks, bins=50, color=COLORS['green_900'], edgecolor='white', alpha=0.8)
    mean_val = np.mean(true_risks)
    median_val = np.median(true_risks)
    ax1.axvline(x=mean_val, color=COLORS['coral_500'], linestyle='--', linewidth=2)
    ax1.axvline(x=median_val, color=COLORS['mint_500'], linestyle='--', linewidth=2)
    ax1.set_xlabel('True Risk', fontsize=11, color=COLORS['text_primary'])
    ax1.set_ylabel('Count', fontsize=11, color=COLORS['text_primary'])
    ax1.set_title('A. Distribution of True Risk (Beta(2.5, 3.5))', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Legend below x-axis
    ax1.legend([f'Mean: {mean_val:.3f}', f'Median: {median_val:.3f}'], 
               loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    
    # B. Default Probability Curve
    ax2 = axes[0, 1]
    x_range = np.linspace(0, 1, 100)
    y_default_prob = x_range ** 1.8
    ax2.plot(x_range, y_default_prob * 100, color=COLORS['coral_500'], linewidth=2)
    ax2.fill_between(x_range, 0, y_default_prob * 100, color=COLORS['coral_500'], alpha=0.2)
    ax2.set_xlabel('True Risk', fontsize=11, color=COLORS['text_primary'])
    ax2.set_ylabel('Default Probability (%)', fontsize=11, color=COLORS['text_primary'])
    ax2.set_title('B. Default Probability = True Risk^1.8', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Legend below x-axis
    ax2.legend(['P(default) = true_risk^1.8'], 
               loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=1, frameon=False)
    
    # C. Applicants and Defaults by Risk Bucket
    ax3 = axes[1, 0]
    buckets = [(0.0, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), 
               (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.0)]
    
    bucket_data = []
    for low, high in buckets:
        in_bucket = [a for a in applicant_pool if low <= a.true_risk < high]
        count = len(in_bucket)
        defaults = sum(1 for a in in_bucket if a.will_default)
        bucket_data.append({
            'label': f'{low:.1f}-{high:.1f}',
            'count': count,
            'defaults': defaults,
        })
    
    labels = [b['label'] for b in bucket_data]
    counts = [b['count'] for b in bucket_data]
    defaults = [b['defaults'] for b in bucket_data]
    
    x = np.arange(len(labels))
    width = 0.35
    
    ax3.bar(x - width/2, counts, width, color=COLORS['green_900'], alpha=0.8)
    ax3.bar(x + width/2, defaults, width, color=COLORS['coral_500'], alpha=0.8)
    ax3.set_xlabel('True Risk Bucket', fontsize=11, color=COLORS['text_primary'])
    ax3.set_ylabel('Count', fontsize=11, color=COLORS['text_primary'])
    ax3.set_title('C. Applicants and Defaults by Risk Bucket', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Legend below x-axis
    ax3.legend(['Total Applicants', 'Will Default'], 
               loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    
    # D. Default Rate by Risk Bucket
    ax4 = axes[1, 1]
    default_rates = [b['defaults'] / b['count'] * 100 if b['count'] > 0 else 0 for b in bucket_data]
    bars = ax4.bar(labels, default_rates, color=COLORS['coral_500'], alpha=0.8, edgecolor='white')
    ax4.set_xlabel('True Risk Bucket', fontsize=11, color=COLORS['text_primary'])
    ax4.set_ylabel('Default Rate (%)', fontsize=11, color=COLORS['text_primary'])
    ax4.set_title('D. Default Rate by Risk Bucket', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    for bar, rate in zip(bars, default_rates):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{rate:.0f}%',
                ha='center', va='bottom', fontsize=9)
    
    # Legend below x-axis
    ax4.legend(['Default Rate (%)'], 
               loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=1, frameon=False)
    
    plt.tight_layout(rect=[0, 0.15, 1, 1])
    
    # Add descriptions at the bottom
    fig.text(0.02, 0.11, 
             f"SUMMARY: Total Applicants: {total_applicants:,} | Total Default Profiles: {total_defaulters:,} | Average Real Default Rate: {avg_default_rate:.1f}%",
             fontsize=11, fontweight='bold', color=COLORS['text_primary'])
    
    descriptions = """
    HOW TO READ THESE GRAPHS:
    A. True Risk Histogram: Shows the distribution of inherent risk levels in the applicant pool. Most applicants cluster around 0.3-0.5 risk (Beta distribution).
       The dashed lines show mean and median - when similar, the distribution is relatively symmetric.
    B. Default Probability Curve: Shows the non-linear relationship between true risk and default probability. The ^1.8 exponent means high-risk applicants 
       are disproportionately more likely to default (e.g., 0.8 risk → 68% default probability, not 80%).
    C. Applicants and Defaults by Bucket: Compare the two bars - the gap between them represents applicants who will NOT default in each risk bucket.
       Lower-risk buckets have large gaps (few defaults); higher-risk buckets have smaller gaps (more defaults).
    D. Default Rate by Bucket: Direct view of default probability by risk level. Validates the ^1.8 curve - rates should increase non-linearly from left to right.
    """
    fig.text(0.02, 0.01, descriptions, fontsize=9, color=COLORS['text_muted'], verticalalignment='bottom')
    
    return fig

def plot_cumulative_rates(scenarios: Dict[str, Tuple[ScenarioConfig, SimulationResults]]) -> plt.Figure:
    """Figure 3: Cumulative Rates and Counts"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=True)
    ax1, ax2 = axes[0]  # Top row: rates
    ax3, ax4 = axes[1]  # Bottom row: counts
    
    # Find common delta window for shading
    delta_windows = []
    for name, (config, results) in scenarios.items():
        if config.delta_enabled:
            delta_windows.append((config.delta_start, config.delta_end))
    
    for ax in [ax1, ax2, ax3, ax4]:
        for start, end in delta_windows:
            ax.axvspan(start, end, alpha=0.15, color=COLORS['coral_500'])
    
    # Collect legend handles
    lines = []
    labels = []
    
    # Plot each scenario
    for idx, (name, (config, results)) in enumerate(scenarios.items()):
        color = SCENARIO_COLORS[idx % len(SCENARIO_COLORS)]
        timesteps = range(1, len(results.cumulative_approval_rate) + 1)
        
        # A. Cumulative Approval Rate
        ax1.plot(timesteps, [r * 100 for r in results.cumulative_approval_rate],
                 color=color, linewidth=2, label=name)
        
        # B. Cumulative Default Rate
        ax2.plot(timesteps, [r * 100 for r in results.cumulative_default_rate],
                 color=color, linewidth=2)
        
        # C. Cumulative Approval Count (use stored data)
        ax3.plot(timesteps, results.cumulative_approval_count, color=color, linewidth=2)
        
        # D. Cumulative Default Count (use stored data)
        ax4.plot(timesteps, results.cumulative_default_count, color=color, linewidth=2)
        
        lines.append(plt.Line2D([0], [0], color=color, linewidth=2))
        labels.append(name)
    
    # Titles and labels
    ax1.set_xlabel('Timestep', fontsize=11, color=COLORS['text_primary'])
    ax1.set_ylabel('Cumulative Approval Rate (%)', fontsize=11, color=COLORS['text_primary'])
    ax1.set_title('A. Cumulative Approval Rate', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', frameon=True)
    
    ax2.set_xlabel('Timestep', fontsize=11, color=COLORS['text_primary'])
    ax2.set_ylabel('Cumulative Default Rate (%)', fontsize=11, color=COLORS['text_primary'])
    ax2.set_title('B. Cumulative Default Rate', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    ax3.set_xlabel('Timestep', fontsize=11, color=COLORS['text_primary'])
    ax3.set_ylabel('Cumulative Approvals (count)', fontsize=11, color=COLORS['text_primary'])
    ax3.set_title('C. Cumulative Approval Count', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    ax4.set_xlabel('Timestep', fontsize=11, color=COLORS['text_primary'])
    ax4.set_ylabel('Cumulative Defaults (count)', fontsize=11, color=COLORS['text_primary'])
    ax4.set_title('D. Cumulative Default Count', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    return fig

def plot_time_dynamics(scenarios: Dict[str, Tuple[ScenarioConfig, SimulationResults]], 
                       rolling_window: int = 20) -> plt.Figure:
    """Figure 4: Time-Based Dynamics (3x2 layout) with per-timestep counts"""
    fig, axes = plt.subplots(3, 2, figsize=(14, 14), constrained_layout=True)
    ax1, ax2 = axes[0]  # Top row: rates
    ax3, ax4 = axes[1]  # Middle row: counts per timestep
    ax5, ax6 = axes[2]  # Bottom row: offset and delta
    
    # Find common delta window for shading
    delta_windows = []
    for name, (config, results) in scenarios.items():
        if config.delta_enabled:
            delta_windows.append((config.delta_start, config.delta_end, config.delta_min, config.delta_max))
    
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        for start, end, _, _ in delta_windows:
            ax.axvspan(start, end, alpha=0.15, color=COLORS['coral_500'])
    
    for idx, (name, (config, results)) in enumerate(scenarios.items()):
        color = SCENARIO_COLORS[idx % len(SCENARIO_COLORS)]
        timesteps = range(1, len(results.timestep_approval_rate) + 1)
        
        # A. Approval Rate (smoothed)
        smoothed_approval = rolling_average(results.timestep_approval_rate, rolling_window)
        ax1.plot(timesteps, [r * 100 if r else 0 for r in smoothed_approval],
                 color=color, linewidth=2, label=name)
        
        # B. Default Rate (smoothed)
        smoothed_default = rolling_average(results.timestep_default_rate, rolling_window)
        valid_t = [t for t, r in zip(timesteps, smoothed_default) if r is not None]
        valid_r = [r * 100 for r in smoothed_default if r is not None]
        ax2.plot(valid_t, valid_r, color=color, linewidth=2)
        
        # C. Approvals per Timestep (count)
        ax3.plot(timesteps, results.timestep_approval_count, color=color, linewidth=1.5, alpha=0.8)
        
        # D. Defaults per Timestep (count)
        ax4.plot(timesteps, results.timestep_default_count, color=color, linewidth=1.5, alpha=0.8)
        
        # E. Calibration Offset
        ax5.plot(timesteps, results.calibration_offset_history, color=color, linewidth=2)
        
        # F. Delta values
        ax6.plot(timesteps, results.delta_history, color=color, linewidth=2)
    
    ax1.set_xlabel('Timestep', fontsize=11, color=COLORS['text_primary'])
    ax1.set_ylabel('Approval Rate (%)', fontsize=11, color=COLORS['text_primary'])
    ax1.set_title(f'A. Approval Rate ({rolling_window}-step rolling avg)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', frameon=True)
    
    ax2.set_xlabel('Timestep', fontsize=11, color=COLORS['text_primary'])
    ax2.set_ylabel('Default Rate (%)', fontsize=11, color=COLORS['text_primary'])
    ax2.set_title(f'B. Default Rate ({rolling_window}-step rolling avg)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    ax3.set_xlabel('Timestep', fontsize=11, color=COLORS['text_primary'])
    ax3.set_ylabel('Approvals (count)', fontsize=11, color=COLORS['text_primary'])
    ax3.set_title('C. Approvals per Timestep', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    ax4.set_xlabel('Timestep', fontsize=11, color=COLORS['text_primary'])
    ax4.set_ylabel('Defaults (count)', fontsize=11, color=COLORS['text_primary'])
    ax4.set_title('D. Defaults per Timestep', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    ax5.set_xlabel('Timestep', fontsize=11, color=COLORS['text_primary'])
    ax5.set_ylabel('Calibration Offset', fontsize=11, color=COLORS['text_primary'])
    ax5.set_title('E. AI Calibration Offset', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    ax6.set_xlabel('Timestep', fontsize=11, color=COLORS['text_primary'])
    ax6.set_ylabel('Delta (δ)', fontsize=11, color=COLORS['text_primary'])
    ax6.set_title('F. Delta (ABO) Applied', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    ax6.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    return fig


def plot_risk_distribution(scenarios: Dict[str, Tuple[ScenarioConfig, SimulationResults]],
                           applicant_pool: List[Applicant],
                           timestep_start: int, timestep_end: int) -> plt.Figure:
    """Figure 4E: Risk Distribution - True Risk vs Effective Signal for selected timestep range"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get true risks for applicants in the selected timestep range
    # Each timestep processes n_applicants_per_step applicants
    first_config = list(scenarios.values())[0][0]
    n_per_step = first_config.n_applicants_per_step
    
    start_idx = (timestep_start - 1) * n_per_step
    end_idx = timestep_end * n_per_step
    
    # True risk from pool
    selected_applicants = applicant_pool[start_idx:end_idx]
    true_risks = [a.true_risk for a in selected_applicants]
    
    # Plot true risk as step line - GREY, SOLID (neutral reference)
    bins = np.linspace(0, 1, 51)
    counts_true, bin_edges = np.histogram(true_risks, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    ax.step(bin_centers, counts_true, where='mid', color=COLORS['text_muted'], 
            linewidth=2.5, linestyle='-', label='True Risk (ground truth)')
    
    # Plot effective signals for each scenario using SCENARIO_COLORS
    linestyles = ['--', ':', '-.', '--', ':']
    
    for idx, (name, (config, results)) in enumerate(scenarios.items()):
        # Filter outcomes for selected timestep range
        selected_outcomes = [o for o in results.all_outcomes 
                           if timestep_start <= o['timestep'] <= timestep_end]
        
        if selected_outcomes:
            effective_signals = [o['effective_signal'] for o in selected_outcomes]
            counts_eff, _ = np.histogram(effective_signals, bins=bins)
            color = SCENARIO_COLORS[idx % len(SCENARIO_COLORS)]
            ax.step(bin_centers, counts_eff, where='mid', color=color, 
                    linewidth=2.5, linestyle=linestyles[idx % len(linestyles)],
                    label=f'{name} - Effective Signal')
    
    # Add threshold lines
    ax.axvline(x=first_config.low_thresh, color=COLORS['coral_600'], linestyle='--', 
               linewidth=2, label=f'LOW threshold ({first_config.low_thresh})')
    ax.axvline(x=first_config.high_thresh, color=COLORS['coral_600'], linestyle=':', 
               linewidth=2, label=f'HIGH threshold ({first_config.high_thresh})')
    
    ax.set_xlabel('Risk Score', fontsize=11, color=COLORS['text_primary'])
    ax.set_ylabel('Count', fontsize=11, color=COLORS['text_primary'])
    ax.set_title(f'E. True Risk vs Effective Signal (Timesteps {timestep_start}-{timestep_end})', 
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    
    plt.tight_layout(rect=[0, 0.15, 1, 1])
    
    # Add description
    description = """
    HOW TO READ THIS GRAPH:
    This compares the TRUE RISK (what applicants actually are) vs EFFECTIVE SIGNAL (what the AI sees after calibration + δ).
    • If lines overlap perfectly → AI perception matches reality
    • If Effective Signal shifts LEFT → AI is more permissive (underestimating risk)
    • If Effective Signal shifts RIGHT → AI is more conservative (overestimating risk)
    • Spike at 0 or 1 → Clipping due to extreme calibration offset
    """
    fig.text(0.02, 0.01, description, fontsize=9, color=COLORS['text_muted'], verticalalignment='bottom')
    
    return fig

def plot_jump_effects(scenarios: Dict[str, Tuple[ScenarioConfig, SimulationResults]],
                      config: ScenarioConfig) -> plt.Figure:
    """Figure 5: Jump Effects Analysis with descriptions"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 13))
    
    # Combine data from all scenarios for analysis
    all_effective_signals = []
    all_base_scores = []
    all_jump_events = []
    
    for name, (cfg, results) in scenarios.items():
        all_effective_signals.extend(results.effective_signals)
        all_base_scores.extend(results.base_scores)
        all_jump_events.extend(results.jump_events)
    
    low_thresh = config.low_thresh
    high_thresh = config.high_thresh
    
    # A. Signal Distribution with Jump Zones
    ax1 = axes[0, 0]
    ax1.hist(all_base_scores, bins=50, color=COLORS['green_700'], alpha=0.6, edgecolor='white')
    ax1.hist(all_effective_signals, bins=50, color=COLORS['mint_500'], alpha=0.6, edgecolor='white')
    ax1.axvline(x=low_thresh, color=COLORS['coral_600'], linestyle='--', linewidth=2)
    ax1.axvline(x=high_thresh, color=COLORS['coral_600'], linestyle='--', linewidth=2)
    
    # Shade jump zones
    ax1.axvspan(low_thresh - 0.1, low_thresh + 0.1, alpha=0.2, color=COLORS['coral_300'])
    ax1.axvspan(high_thresh - 0.1, high_thresh + 0.1, alpha=0.2, color=COLORS['coral_300'])
    
    ax1.set_xlabel('Signal Value', fontsize=11, color=COLORS['text_primary'])
    ax1.set_ylabel('Count', fontsize=11, color=COLORS['text_primary'])
    ax1.set_title('A. Signal Distribution with Jump Zones', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    ax1.legend(['Base Score', 'Effective Signal', f'LOW thresh ({low_thresh})', f'HIGH thresh ({high_thresh})', 'Jump zones'], 
               loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False, fontsize=8)
    
    # B. Category Jump Rate by Score Region
    ax2 = axes[0, 1]
    bins = np.arange(0, 1.05, 0.05)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    jump_rates = []
    for i in range(len(bins) - 1):
        in_bin = [e for e in all_jump_events if bins[i] <= e['base_score'] < bins[i+1]]
        total_in_bin = len([s for s in all_base_scores if bins[i] <= s < bins[i+1]])
        jump_rates.append(len(in_bin) / total_in_bin if total_in_bin > 0 else 0)
    
    ax2.bar(bin_centers, jump_rates, width=0.04, color=COLORS['coral_400'], alpha=0.8, edgecolor='white')
    ax2.axvline(x=low_thresh, color=COLORS['coral_600'], linestyle='--', linewidth=2)
    ax2.axvline(x=high_thresh, color=COLORS['coral_600'], linestyle='--', linewidth=2)
    ax2.set_xlabel('Base Score (before δ)', fontsize=11, color=COLORS['text_primary'])
    ax2.set_ylabel('Jump Rate', fontsize=11, color=COLORS['text_primary'])
    ax2.set_title('B. Category Jump Rate by Score Region', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add annotation for peak
    if jump_rates and max(jump_rates) > 0:
        max_idx = np.argmax(jump_rates)
        ax2.annotate('Peak jump rate\nnear thresholds', 
                    xy=(bin_centers[max_idx], jump_rates[max_idx]),
                    xytext=(bin_centers[max_idx] + 0.15, jump_rates[max_idx]),
                    fontsize=9, color=COLORS['text_muted'],
                    arrowprops=dict(arrowstyle='->', color=COLORS['text_muted']))
    
    ax2.legend(['Jump Rate', f'Thresholds ({low_thresh}, {high_thresh})'], 
               loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    
    # C. Jump Direction Analysis
    ax3 = axes[1, 0]
    jump_directions = {}
    for event in all_jump_events:
        before = event['category_before'].name
        after = event['category_after'].name
        key = f'{before} → {after}'
        jump_directions[key] = jump_directions.get(key, 0) + 1
    
    if jump_directions:
        directions = list(jump_directions.keys())
        counts = list(jump_directions.values())
        
        # Color based on direction (permissive = coral, conservative = mint)
        bar_colors = []
        for d in directions:
            if 'HIGH → MEDIUM' in d or 'MEDIUM → LOW' in d or 'HIGH → LOW' in d:
                bar_colors.append(COLORS['coral_500'])  # Permissive
            else:
                bar_colors.append(COLORS['mint_500'])  # Conservative
        
        y_pos = np.arange(len(directions))
        bars = ax3.barh(y_pos, counts, color=bar_colors, alpha=0.8, edgecolor='white')
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(directions)
        ax3.set_xlabel('Number of Jumps', fontsize=11, color=COLORS['text_primary'])
        ax3.set_title('C. Jump Direction Analysis', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='x')
        
        for i, count in enumerate(counts):
            ax3.text(count + max(counts)*0.02, i, str(count), va='center', fontsize=9)
        
        from matplotlib.patches import Patch
        ax3.legend([Patch(facecolor=COLORS['coral_500'], alpha=0.8), 
                   Patch(facecolor=COLORS['mint_500'], alpha=0.8)],
                  ['Permissive (↓ risk category)', 'Conservative (↑ risk category)'],
                  loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    else:
        ax3.text(0.5, 0.5, 'No jump events recorded', ha='center', va='center', 
                transform=ax3.transAxes, fontsize=12, color=COLORS['text_muted'])
        ax3.set_title('C. Jump Direction Analysis', fontsize=12, fontweight='bold')
    
    # D. Impact of Category Jumps on Decision Quality
    ax4 = axes[1, 1]
    beneficial = 0  # Jump led to correct decision
    harmful = 0     # Jump led to wrong decision
    neutral = 0     # Same outcome either way
    
    for event in all_jump_events:
        before_cat = event['category_before']
        after_cat = event['category_after']
        will_default = event['will_default']
        
        # Determine if jump was in "correct" direction
        # Permissive jump on non-defaulter = beneficial
        # Permissive jump on defaulter = harmful
        # Conservative jump on defaulter = beneficial
        # Conservative jump on non-defaulter = harmful
        
        before_val = {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2}[before_cat.name]
        after_val = {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2}[after_cat.name]
        
        if after_val < before_val:  # More permissive
            if will_default:
                harmful += 1
            else:
                beneficial += 1
        else:  # More conservative
            if will_default:
                beneficial += 1
            else:
                harmful += 1
    
    neutral = len(all_jump_events) - beneficial - harmful
    
    if all_jump_events:
        categories = ['Beneficial\n(correct direction)', 'Harmful\n(wrong direction)', 'Neutral\n(same outcome)']
        values = [beneficial, harmful, neutral]
        colors_bar = [COLORS['mint_500'], COLORS['coral_500'], COLORS['silver_300']]
        
        bars = ax4.bar(categories, values, color=colors_bar, alpha=0.8, edgecolor='white')
        ax4.set_ylabel('Number of Jump Events', fontsize=11, color=COLORS['text_primary'])
        ax4.set_title('D. Impact of Category Jumps on Decision Quality', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.02, str(val),
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax4.legend(['Beneficial: δ moved category in direction matching true outcome',
                   'Harmful: δ moved category opposite to true outcome'],
                  loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=1, frameon=False, fontsize=8)
    else:
        ax4.text(0.5, 0.5, 'No jump events recorded', ha='center', va='center',
                transform=ax4.transAxes, fontsize=12, color=COLORS['text_muted'])
        ax4.set_title('D. Impact of Category Jumps on Decision Quality', fontsize=12, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0.18, 1, 1])
    
    # Add descriptions - moved lower to avoid overlap
    descriptions = """
    HOW TO READ THESE GRAPHS:
    A. Signal Distribution: Compares base scores (before δ) vs effective signals (after δ). The shaded "jump zones" near thresholds show where small δ changes can flip categories.
    B. Jump Rate by Score: Shows probability of a category jump at each score level. Peaks near thresholds indicate where δ has maximum impact.
    C. Jump Direction: Counts of category transitions. Coral = permissive jumps (HIGH→MED, MED→LOW). Mint = conservative jumps (LOW→MED, MED→HIGH).
    D. Decision Quality Impact: Beneficial = jump matched true outcome. Harmful = jump opposed true outcome. High harmful count = δ is degrading decisions.
    """
    fig.text(0.02, 0.01, descriptions, fontsize=9, color=COLORS['text_muted'], verticalalignment='bottom')
    
    return fig

# =============================================================================
# v24: Financial Impact and Decision Quality Plots
# =============================================================================
def plot_financial_impact(scenarios: Dict[str, Tuple[ScenarioConfig, SimulationResults]]) -> plt.Figure:
    """Cumulative P&L and financial breakdown per scenario"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    ax_pnl, ax_breakdown = axes[0]
    ax_pnl_per_step, ax_summary = axes[1]
    
    # A. Cumulative P&L over time
    for ax in [ax_pnl, ax_pnl_per_step]:
        for name, (config, results) in scenarios.items():
            if config.delta_enabled:
                ax.axvspan(config.delta_start, config.delta_end, 
                          alpha=0.08, color=COLORS['coral_500'], zorder=0)
                break  # Only shade once
    
    for idx, (name, (config, results)) in enumerate(scenarios.items()):
        color = SCENARIO_COLORS[idx % len(SCENARIO_COLORS)]
        timesteps = range(1, len(results.cumulative_pnl) + 1)
        ax_pnl.plot(timesteps, results.cumulative_pnl, color=color, linewidth=2, label=name)
        
        smoothed_pnl = rolling_average(results.timestep_pnl, 20)
        ax_pnl_per_step.plot(timesteps, smoothed_pnl, color=color, linewidth=1.5, alpha=0.8)
    
    ax_pnl.set_xlabel('Timestep'); ax_pnl.set_ylabel('Cumulative P&L (units)')
    ax_pnl.set_title('A. Cumulative Profit & Loss', fontsize=12, fontweight='bold')
    ax_pnl.grid(True, alpha=0.3); ax_pnl.legend(loc='best', frameon=True)
    ax_pnl.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    ax_pnl_per_step.set_xlabel('Timestep'); ax_pnl_per_step.set_ylabel('P&L per Step (20-step avg)')
    ax_pnl_per_step.set_title('C. P&L per Timestep (smoothed)', fontsize=12, fontweight='bold')
    ax_pnl_per_step.grid(True, alpha=0.3)
    ax_pnl_per_step.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # B. Financial component breakdown (side-by-side bars: Revenue vs Costs)
    scenario_names = list(scenarios.keys())
    n_scenarios = len(scenario_names)
    x = np.arange(n_scenarios)
    bar_width = 0.35
    
    revenues = [scenarios[n][1].total_revenue for n in scenario_names]
    default_losses = [scenarios[n][1].total_default_losses for n in scenario_names]
    opp_costs = [scenarios[n][1].total_opportunity_cost for n in scenario_names]
    total_costs = [dl + oc for dl, oc in zip(default_losses, opp_costs)]
    
    # Revenue bars (green, left)
    bars_rev = ax_breakdown.bar(x - bar_width/2, revenues, bar_width, 
                                label='Revenue (good loans)', color=COLORS['mint_500'], 
                                edgecolor='white', linewidth=0.5)
    
    # Cost bars: default losses (coral) + opportunity cost (lighter coral) stacked
    bars_loss = ax_breakdown.bar(x + bar_width/2, default_losses, bar_width, 
                                 label='Default losses', color=COLORS['coral_500'], 
                                 edgecolor='white', linewidth=0.5)
    bars_opp = ax_breakdown.bar(x + bar_width/2, opp_costs, bar_width, 
                                bottom=default_losses,
                                label='Opportunity cost (denied)', color=COLORS['coral_300'], 
                                edgecolor='white', linewidth=0.5)
    
    # Net P&L annotation between bars
    for i in range(n_scenarios):
        net = revenues[i] - total_costs[i]
        mid_x = x[i]
        bar_top = max(revenues[i], total_costs[i])
        ax_breakdown.annotate(f'P&L: {net:+,.0f}', xy=(mid_x, bar_top), 
                              xytext=(0, 8), textcoords='offset points',
                              ha='center', va='bottom', fontsize=8, fontweight='bold',
                              color=COLORS['green_900'] if net >= 0 else COLORS['coral_600'])
    
    ax_breakdown.set_xlabel('Scenario'); ax_breakdown.set_ylabel('Financial Impact (units)')
    ax_breakdown.set_title('B. Financial Components Breakdown', fontsize=12, fontweight='bold')
    ax_breakdown.set_xticks(x); ax_breakdown.set_xticklabels(scenario_names, fontsize=9)
    ax_breakdown.legend(fontsize=8, loc='upper right'); ax_breakdown.grid(True, alpha=0.3, axis='y')
    ax_breakdown.set_ylim(bottom=0)
    
    # D. Summary table
    ax_summary.axis('off')
    columns = ['Scenario', 'Revenue', 'Default\nLosses', 'Opportunity\nCost', 'Net P&L', 'P&L per\nApplicant']
    rows = []
    for name, (config, results) in scenarios.items():
        total_app = config.n_steps * config.n_applicants_per_step
        net = results.total_revenue - results.total_default_losses - results.total_opportunity_cost
        per_app = net / total_app if total_app > 0 else 0
        rows.append([
            name, f'{results.total_revenue:,.0f}', f'{results.total_default_losses:,.0f}',
            f'{results.total_opportunity_cost:,.0f}', f'{net:+,.0f}', f'{per_app:+.4f}'
        ])
    
    table = ax_summary.table(cellText=rows, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False); table.set_fontsize(9)
    table.scale(1.1, 1.8)
    for i in range(len(columns)):
        table[(0, i)].set_facecolor(COLORS['green_900'])
        table[(0, i)].set_text_props(color='white', fontweight='bold', fontsize=8)
    for row_idx in range(len(rows)):
        for col_idx in range(len(columns)):
            table[(row_idx + 1, col_idx)].set_facecolor(COLORS['silver_100'])
    
    ax_summary.set_title('D. Financial Summary', fontsize=12, fontweight='bold')
    
    fig.text(0.02, 0.01, 
        f"Financial model: Revenue per good loan = {list(scenarios.values())[0][0].financial.revenue_per_good_loan}, "
        f"Cost per default = {list(scenarios.values())[0][0].financial.cost_per_default}, "
        f"Opportunity cost per denied non-defaulter = {list(scenarios.values())[0][0].financial.opportunity_cost_per_denial}",
        fontsize=8, color=COLORS['text_muted'], verticalalignment='bottom')
    
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    return fig

def plot_confusion_matrix(scenarios: Dict[str, Tuple[ScenarioConfig, SimulationResults]]) -> plt.Figure:
    """Confusion matrix and decision accuracy metrics per scenario"""
    n_scenarios = len(scenarios)
    fig, axes = plt.subplots(1, n_scenarios, figsize=(5 * n_scenarios, 5), squeeze=False)
    axes = axes[0]
    
    for idx, (name, (config, results)) in enumerate(scenarios.items()):
        ax = axes[idx]
        
        tp = results.true_positives
        tn = results.true_negatives
        fp = results.false_positives
        fn = results.false_negatives
        total = tp + tn + fp + fn
        
        matrix = np.array([[tn, fp], [fn, tp]])
        labels = np.array([
            [f'TN\n{tn:,}\n({tn/total*100:.1f}%)' if total > 0 else 'TN\n0',
             f'FP\n{fp:,}\n({fp/total*100:.1f}%)' if total > 0 else 'FP\n0'],
            [f'FN\n{fn:,}\n({fn/total*100:.1f}%)' if total > 0 else 'FN\n0',
             f'TP\n{tp:,}\n({tp/total*100:.1f}%)' if total > 0 else 'TP\n0']
        ])
        
        # Color map: greens for correct, corals for errors
        colors = np.array([
            [COLORS['mint_200'], COLORS['coral_300']],
            [COLORS['coral_300'], COLORS['mint_200']]
        ])
        
        for i in range(2):
            for j in range(2):
                ax.add_patch(plt.Rectangle((j, 1-i), 1, 1, fill=True, 
                            facecolor=colors[i][j], edgecolor='white', linewidth=2))
                ax.text(j + 0.5, 1.5 - i, labels[i][j], ha='center', va='center', 
                       fontsize=10, fontweight='bold', color=COLORS['text_primary'])
        
        ax.set_xlim(0, 2); ax.set_ylim(0, 2)
        ax.set_xticks([0.5, 1.5]); ax.set_xticklabels(['Approved', 'Denied'])
        ax.set_yticks([0.5, 1.5]); ax.set_yticklabels(['Defaulter', 'Non-Defaulter'])
        ax.set_xlabel('Decision', fontsize=10, fontweight='bold')
        ax.set_ylabel('Actual Outcome', fontsize=10, fontweight='bold')
        
        # Compute derived metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        accuracy = (tp + tn) / total if total > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        title = f'{name}\nAcc={accuracy:.1%} | Prec={precision:.1%} | Rec={recall:.1%} | F1={f1:.1%}'
        ax.set_title(title, fontsize=10, fontweight='bold', color=COLORS['text_primary'])
    
    fig.text(0.02, 0.01,
        "TP = Correctly denied defaulter | TN = Correctly approved non-defaulter | "
        "FP = Denied non-defaulter (lost revenue) | FN = Approved defaulter (loss). "
        "Closed-window only (all loans matured).",
        fontsize=8, color=COLORS['text_muted'], verticalalignment='bottom')
    
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    return fig

# =============================================================================
# ISCIL Plotting Functions
# =============================================================================
def plot_iscil_detection(scenarios: Dict[str, Tuple[ScenarioConfig, SimulationResults]]) -> plt.Figure:
    """Plot ISCIL Coherence-Risk Score over time for all scenarios"""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Find max threshold across scenarios for reference line
    max_threshold = 0
    for name, (config, results) in scenarios.items():
        if config.iscil_enabled:
            max_threshold = max(max_threshold, config.iscil_threshold)
    if max_threshold == 0:
        max_threshold = 1.0  # fallback if no ISCIL scenario
    
    for idx, (name, (config, results)) in enumerate(scenarios.items()):
        color = SCENARIO_COLORS[idx % len(SCENARIO_COLORS)]
        timesteps = list(range(1, len(results.iscil_crs_history) + 1))
        
        if any(crs > 0 for crs in results.iscil_crs_history):
            ax.plot(timesteps, results.iscil_crs_history, 
                   color=color, linewidth=1.5, label=name, alpha=0.9)
    
    # Add threshold line
    ax.axhline(y=max_threshold, color=COLORS['coral_500'], linestyle='--', 
               linewidth=2, label=f'Threshold (θ={max_threshold})', alpha=0.8)
    
    # Add delta windows as shaded regions
    for idx, (name, (config, results)) in enumerate(scenarios.items()):
        if config.delta_enabled:
            ax.axvspan(config.delta_start, config.delta_end, 
                      alpha=0.15, color=COLORS['coral_300'], label='δ active window' if idx == 0 else None)
    
    ax.set_xlabel('Timestep', fontsize=12, color=COLORS['text_primary'])
    ax.set_ylabel('Coherence-Risk Score (CRS)', fontsize=12, color=COLORS['text_primary'])
    ax.set_title('A. ISCIL Coherence-Risk Score Detection', fontsize=14, fontweight='bold', color=COLORS['text_primary'])
    ax.legend(loc='upper left', frameon=True, fancybox=True)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(len(results.iscil_crs_history) for _, (_, results) in scenarios.items()))
    
    plt.tight_layout()
    return fig

def plot_iscil_intervention(scenarios: Dict[str, Tuple[ScenarioConfig, SimulationResults]]) -> plt.Figure:
    """Plot ISCIL Intervention Strength over time"""
    fig, ax = plt.subplots(figsize=(14, 5))
    
    for idx, (name, (config, results)) in enumerate(scenarios.items()):
        color = SCENARIO_COLORS[idx % len(SCENARIO_COLORS)]
        timesteps = list(range(1, len(results.iscil_intervention_strength_history) + 1))
        
        if config.iscil_containment_enabled and any(s > 0 for s in results.iscil_intervention_strength_history):
            ax.fill_between(timesteps, results.iscil_intervention_strength_history, 
                           alpha=0.4, color=color, label=name)
            ax.plot(timesteps, results.iscil_intervention_strength_history, 
                   color=color, linewidth=1.5, alpha=0.9)
    
    # Add delta windows
    for idx, (name, (config, results)) in enumerate(scenarios.items()):
        if config.delta_enabled:
            ax.axvspan(config.delta_start, config.delta_end, 
                      alpha=0.15, color=COLORS['coral_300'], label='δ active window' if idx == 0 else None)
    
    ax.set_xlabel('Timestep', fontsize=12, color=COLORS['text_primary'])
    ax.set_ylabel('Intervention Strength (0-1)', fontsize=12, color=COLORS['text_primary'])
    ax.set_title('B. ISCIL Proportional Intervention', fontsize=14, fontweight='bold', color=COLORS['text_primary'])
    ax.legend(loc='upper left', frameon=True, fancybox=True)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    ax.set_xlim(0, max(len(results.iscil_intervention_strength_history) for _, (_, results) in scenarios.items()))
    
    plt.tight_layout()
    return fig

def plot_iscil_combined(scenarios: Dict[str, Tuple[ScenarioConfig, SimulationResults]], 
                        rolling_window: int = 1) -> plt.Figure:
    """Combined ISCIL plot with CRS and Intervention Strength"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 11), sharex=True, constrained_layout=True)
    
    # Find max threshold
    max_threshold = 0
    for name, (config, results) in scenarios.items():
        if config.iscil_enabled:
            max_threshold = max(max_threshold, config.iscil_threshold)
    if max_threshold == 0:
        max_threshold = 1.0
    
    # Plot A: Coherence-Risk Score
    for idx, (name, (config, results)) in enumerate(scenarios.items()):
        color = SCENARIO_COLORS[idx % len(SCENARIO_COLORS)]
        timesteps = list(range(1, len(results.iscil_crs_history) + 1))
        
        if any(crs > 0 for crs in results.iscil_crs_history):
            # Apply rolling average if window > 1
            if rolling_window > 1:
                smoothed_crs = rolling_average(results.iscil_crs_history, rolling_window)
                ax1.plot(timesteps, smoothed_crs, 
                        color=color, linewidth=1.5, label=name, alpha=0.9)
            else:
                ax1.plot(timesteps, results.iscil_crs_history, 
                        color=color, linewidth=1.5, label=name, alpha=0.9)
    
    ax1.axhline(y=max_threshold, color=COLORS['coral_500'], linestyle='--', 
                linewidth=2, label=f'Threshold (θ={max_threshold})', alpha=0.8)
    
    # Add delta windows
    for idx, (name, (config, results)) in enumerate(scenarios.items()):
        if config.delta_enabled:
            ax1.axvspan(config.delta_start, config.delta_end, 
                       alpha=0.15, color=COLORS['coral_300'], label='δ active window' if idx == 0 else None)
    
    ax1.set_ylabel('Coherence-Risk Score (CRS)', fontsize=12, color=COLORS['text_primary'])
    title_suffix = f" ({rolling_window}-step rolling avg)" if rolling_window > 1 else ""
    ax1.set_title(f'A. ISCIL Coherence-Risk Score Detection{title_suffix}', fontsize=14, fontweight='bold', color=COLORS['text_primary'])
    ax1.legend(loc='upper left', frameon=True, fancybox=True)
    ax1.grid(True, alpha=0.3)
    
    # Plot B: Intervention Strength
    for idx, (name, (config, results)) in enumerate(scenarios.items()):
        color = SCENARIO_COLORS[idx % len(SCENARIO_COLORS)]
        timesteps = list(range(1, len(results.iscil_intervention_strength_history) + 1))
        
        if config.iscil_containment_enabled and any(s > 0 for s in results.iscil_intervention_strength_history):
            # Apply rolling average if window > 1
            if rolling_window > 1:
                smoothed_strength = rolling_average(results.iscil_intervention_strength_history, rolling_window)
                ax2.fill_between(timesteps, smoothed_strength, 
                                alpha=0.4, color=color, label=name)
                ax2.plot(timesteps, smoothed_strength, 
                        color=color, linewidth=1.5, alpha=0.9)
            else:
                ax2.fill_between(timesteps, results.iscil_intervention_strength_history, 
                                alpha=0.4, color=color, label=name)
                ax2.plot(timesteps, results.iscil_intervention_strength_history, 
                        color=color, linewidth=1.5, alpha=0.9)
    
    for idx, (name, (config, results)) in enumerate(scenarios.items()):
        if config.delta_enabled:
            ax2.axvspan(config.delta_start, config.delta_end, 
                       alpha=0.15, color=COLORS['coral_300'])
    
    ax2.set_xlabel('Timestep', fontsize=12, color=COLORS['text_primary'])
    ax2.set_ylabel('Intervention Strength (0-1)', fontsize=12, color=COLORS['text_primary'])
    ax2.set_title(f'B. ISCIL Proportional Intervention{title_suffix}', fontsize=14, fontweight='bold', color=COLORS['text_primary'])
    ax2.legend(loc='upper left', frameon=True, fancybox=True)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.1)
    
    return fig

def plot_iscil_summary_table(scenarios: Dict[str, Tuple[ScenarioConfig, SimulationResults]]) -> plt.Figure:
    """ISCIL Summary metrics table - v23 with ROC Window"""
    fig, ax = plt.subplots(figsize=(16, 4 + len(scenarios) * 0.5))
    ax.axis('off')
    
    columns = ['Scenario', 'ISCIL\nEnabled', 'Containment', 'Threshold', 
               'ROC\nWindow (k)', 'First\nAlert', 'Intervention\nStart', 'Peak CRS', 'Intervention\nTimesteps', 
               'Total\nDefaults', 'Default Rate\n(of Matured)']
    
    rows = []
    for name, (config, results) in scenarios.items():
        total_applicants = config.n_steps * config.n_applicants_per_step
        default_rate = (results.total_defaults / results.total_matured_approvals * 100) if results.total_matured_approvals > 0 else 0
        
        rows.append([
            name,
            '✓' if config.iscil_enabled else '✗',
            '✓' if config.iscil_containment_enabled else '✗',
            f'{config.iscil_threshold:.1f}' if config.iscil_enabled else '-',
            f'{config.iscil_roc_window}' if config.iscil_enabled else '-',
            f't={results.iscil_alert_time}' if results.iscil_alert_time else '-',
            f't={results.iscil_sustained_alert_time}' if results.iscil_sustained_alert_time else '-',
            f'{results.iscil_peak_crs:.2f}' if results.iscil_peak_crs > 0 else '-',
            f'{results.iscil_total_intervention_timesteps}' if config.iscil_containment_enabled else '-',
            f'{results.total_defaults:,}',
            f'{default_rate:.1f}%'
        ])
    
    table = ax.table(
        cellText=rows,
        colLabels=columns,
        loc='center',
        cellLoc='center'
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # Style header
    for col_idx in range(len(columns)):
        table[(0, col_idx)].set_facecolor(COLORS['green_900'])
        table[(0, col_idx)].set_text_props(color='white', fontweight='bold')
        table[(0, col_idx)].set_height(0.12)
    
    # Style data rows
    for row_idx in range(len(rows)):
        for col_idx in range(len(columns)):
            table[(row_idx + 1, col_idx)].set_facecolor(COLORS['silver_100'])
            table[(row_idx + 1, col_idx)].set_height(0.08)
    
    ax.set_title('ISCIL Summary Metrics (v23: Rate-of-Change Detection)', fontsize=14, fontweight='bold', 
                 color=COLORS['text_primary'], pad=20, loc='left')
    
    plt.tight_layout()
    return fig

# =============================================================================
# Feedback Decomposition (v0.1)
# =============================================================================
def plot_feedback_decomposition(scenarios: Dict[str, Tuple[ScenarioConfig, SimulationResults]],
                                smoothing_window: int = 1) -> plt.Figure:
    """Plot cumulative v3 and v4 contributions to calibration offset for all scenarios,
    including per-timestep raw signals to show when feedback forces fire.
    smoothing_window: 1 = raw signals, >1 = rolling average over that many steps."""
    n_scenarios = len(scenarios)
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.patch.set_facecolor(COLORS['white'])
    
    scenario_list = list(scenarios.items())
    
    # Color map for scenarios
    scenario_colors = [COLORS['green_900'], COLORS['mint_500'], COLORS['coral_500']]
    scenario_styles = [('--', 0.5), ('-', 1.0), ('-', 1.0)]
    
    # Find delta window for shading
    delta_start, delta_end = None, None
    for name, (config, results) in scenarios.items():
        if config.delta_enabled:
            delta_start = config.delta_start
            delta_end = config.delta_end
            break
    
    def shade_delta(ax):
        if delta_start is not None and delta_end is not None:
            ax.axvspan(delta_start, delta_end, alpha=0.1, color=COLORS['coral_500'])
    
    def smooth(series, window):
        """Apply rolling average. Window=1 returns raw data."""
        if window <= 1 or len(series) < window:
            return series
        arr = np.array(series)
        kernel = np.ones(window) / window
        smoothed = np.convolve(arr, kernel, mode='valid')
        # Pad front with NaN so x-axis stays aligned
        pad = [np.nan] * (window - 1)
        return pad + list(smoothed)
    
    smooth_label = f' ({smoothing_window}-step avg)' if smoothing_window > 1 else ' (raw)'
    
    # ===== ROW 1: CUMULATIVE =====
    ax = axes[0, 0]
    ax.set_title('A. v\u2083 Cumulative (permissive \u2193)', fontsize=10, fontweight='bold', color=COLORS['text_primary'])
    for i, (name, (config, results)) in enumerate(scenario_list):
        if results.v3_cumulative_contribution:
            ls, alpha = scenario_styles[min(i, len(scenario_styles)-1)]
            ax.plot(results.v3_cumulative_contribution, color=scenario_colors[min(i, len(scenario_colors)-1)],
                    linewidth=1.5, linestyle=ls, alpha=alpha, label=name)
    shade_delta(ax)
    ax.axhline(y=0, color=COLORS['text_muted'], linewidth=0.5, linestyle=':')
    ax.set_ylabel('Cumulative \u0394\u03c9 from v\u2083', fontsize=9, color=COLORS['text_primary'])
    ax.legend(fontsize=8)
    
    ax = axes[0, 1]
    ax.set_title('B. v\u2084 Cumulative (corrective \u2191)', fontsize=10, fontweight='bold', color=COLORS['text_primary'])
    for i, (name, (config, results)) in enumerate(scenario_list):
        if results.v4_cumulative_contribution:
            ls, alpha = scenario_styles[min(i, len(scenario_styles)-1)]
            ax.plot(results.v4_cumulative_contribution, color=scenario_colors[min(i, len(scenario_colors)-1)],
                    linewidth=1.5, linestyle=ls, alpha=alpha, label=name)
    shade_delta(ax)
    ax.axhline(y=0, color=COLORS['text_muted'], linewidth=0.5, linestyle=':')
    ax.set_ylabel('Cumulative \u0394\u03c9 from v\u2084', fontsize=9, color=COLORS['text_primary'])
    ax.legend(fontsize=8)
    
    ax = axes[0, 2]
    ax.set_title('C. Net Calibration Offset \u03c9 = v\u2083 + v\u2084', fontsize=10, fontweight='bold', color=COLORS['text_primary'])
    for i, (name, (config, results)) in enumerate(scenario_list):
        if results.calibration_offset_history:
            ls, alpha = scenario_styles[min(i, len(scenario_styles)-1)]
            ax.plot(results.calibration_offset_history, color=scenario_colors[min(i, len(scenario_colors)-1)],
                    linewidth=1.5, linestyle=ls, alpha=alpha, label=name)
    shade_delta(ax)
    ax.axhline(y=0, color=COLORS['text_muted'], linewidth=0.5, linestyle=':')
    ax.set_ylabel('Calibration Offset (\u03c9)', fontsize=9, color=COLORS['text_primary'])
    ax.legend(fontsize=8)
    
    # ===== ROW 2: PER-TIMESTEP SIGNALS (with optional smoothing) =====
    ax = axes[1, 0]
    ax.set_title(f'D. v\u2083 Per-Timestep Push{smooth_label}', fontsize=10, fontweight='bold', color=COLORS['text_primary'])
    for i, (name, (config, results)) in enumerate(scenario_list):
        if results.v3_per_step_delta:
            data = smooth(results.v3_per_step_delta, smoothing_window)
            ls, alpha = scenario_styles[min(i, len(scenario_styles)-1)]
            ax.plot(data, color=scenario_colors[min(i, len(scenario_colors)-1)],
                    linewidth=0.8 if smoothing_window <= 1 else 1.2, linestyle=ls, alpha=alpha, label=name)
    shade_delta(ax)
    ax.axhline(y=0, color=COLORS['text_muted'], linewidth=0.5, linestyle=':')
    ax.set_ylabel('v\u2083 \u0394\u03c9 per step', fontsize=9, color=COLORS['text_primary'])
    ax.legend(fontsize=8)
    
    ax = axes[1, 1]
    ax.set_title(f'E. v\u2084 Per-Timestep Push{smooth_label}', fontsize=10, fontweight='bold', color=COLORS['text_primary'])
    for i, (name, (config, results)) in enumerate(scenario_list):
        if results.v4_per_step_delta:
            data = smooth(results.v4_per_step_delta, smoothing_window)
            ls, alpha = scenario_styles[min(i, len(scenario_styles)-1)]
            ax.plot(data, color=scenario_colors[min(i, len(scenario_colors)-1)],
                    linewidth=0.8 if smoothing_window <= 1 else 1.2, linestyle=ls, alpha=alpha, label=name)
    shade_delta(ax)
    ax.axhline(y=0, color=COLORS['text_muted'], linewidth=0.5, linestyle=':')
    ax.set_ylabel('v\u2084 \u0394\u03c9 per step', fontsize=9, color=COLORS['text_primary'])
    ax.legend(fontsize=8)
    
    ax = axes[1, 2]
    ax.set_title(f'F. Net \u03c9 Change Per Timestep{smooth_label}', fontsize=10, fontweight='bold', color=COLORS['text_primary'])
    for i, (name, (config, results)) in enumerate(scenario_list):
        if results.v3_per_step_delta and results.v4_per_step_delta:
            net_delta = [v3 + v4 for v3, v4 in zip(results.v3_per_step_delta, results.v4_per_step_delta)]
            data = smooth(net_delta, smoothing_window)
            ls, alpha = scenario_styles[min(i, len(scenario_styles)-1)]
            ax.plot(data, color=scenario_colors[min(i, len(scenario_colors)-1)],
                    linewidth=0.8 if smoothing_window <= 1 else 1.2, linestyle=ls, alpha=alpha, label=name)
    shade_delta(ax)
    ax.axhline(y=0, color=COLORS['text_muted'], linewidth=0.5, linestyle=':')
    ax.set_ylabel('Net \u0394\u03c9 per step', fontsize=9, color=COLORS['text_primary'])
    ax.legend(fontsize=8)
    
    # ===== ROW 3: EXCESS (ABO - Baseline) =====
    baseline_name = None
    baseline_results = None
    for name, (config, results) in scenarios.items():
        if not config.delta_enabled:
            baseline_name = name
            baseline_results = results
            break
    
    if baseline_results and len(baseline_results.v3_cumulative_contribution) > 0:
        for name, (config, results) in scenarios.items():
            if config.delta_enabled and not config.iscil_enabled:
                min_len = min(len(results.v3_cumulative_contribution), len(baseline_results.v3_cumulative_contribution))
                v3_diff = [results.v3_cumulative_contribution[i] - baseline_results.v3_cumulative_contribution[i] for i in range(min_len)]
                v4_diff = [results.v4_cumulative_contribution[i] - baseline_results.v4_cumulative_contribution[i] for i in range(min_len)]
                
                min_len_off = min(len(results.calibration_offset_history), len(baseline_results.calibration_offset_history))
                offset_diff = [results.calibration_offset_history[i] - baseline_results.calibration_offset_history[i] for i in range(min_len_off)]
                
                ax = axes[2, 0]
                ax.set_title(f'G. v\u2083 Excess ({name} \u2212 {baseline_name})', fontsize=10, fontweight='bold', color=COLORS['text_primary'])
                ax.fill_between(range(len(v3_diff)), v3_diff, 0, where=[v < 0 for v in v3_diff], color=COLORS['mint_500'], alpha=0.3)
                ax.fill_between(range(len(v3_diff)), v3_diff, 0, where=[v >= 0 for v in v3_diff], color=COLORS['coral_500'], alpha=0.3)
                ax.plot(range(len(v3_diff)), v3_diff, color=COLORS['mint_600'], linewidth=1.5)
                shade_delta(ax)
                ax.axhline(y=0, color=COLORS['green_900'], linewidth=0.8)
                ax.set_ylabel(f'\u0394v\u2083 ({name} \u2212 {baseline_name})', fontsize=9, color=COLORS['text_primary'])
                ax.set_xlabel('Timestep', fontsize=9)
                
                ax = axes[2, 1]
                ax.set_title(f'H. v\u2084 Excess ({name} \u2212 {baseline_name})', fontsize=10, fontweight='bold', color=COLORS['text_primary'])
                ax.fill_between(range(len(v4_diff)), v4_diff, 0, where=[v > 0 for v in v4_diff], color=COLORS['coral_500'], alpha=0.3)
                ax.fill_between(range(len(v4_diff)), v4_diff, 0, where=[v <= 0 for v in v4_diff], color=COLORS['mint_500'], alpha=0.3)
                ax.plot(range(len(v4_diff)), v4_diff, color=COLORS['coral_600'], linewidth=1.5)
                shade_delta(ax)
                ax.axhline(y=0, color=COLORS['green_900'], linewidth=0.8)
                ax.set_ylabel(f'\u0394v\u2084 ({name} \u2212 {baseline_name})', fontsize=9, color=COLORS['text_primary'])
                ax.set_xlabel('Timestep', fontsize=9)
                
                ax = axes[2, 2]
                ax.set_title(f'I. Net Offset Gap (\u03c9_{name} \u2212 \u03c9_{baseline_name})', fontsize=10, fontweight='bold', color=COLORS['text_primary'])
                ax.fill_between(range(len(offset_diff)), offset_diff, 0, where=[v > 0 for v in offset_diff], color=COLORS['coral_500'], alpha=0.3)
                ax.fill_between(range(len(offset_diff)), offset_diff, 0, where=[v <= 0 for v in offset_diff], color=COLORS['mint_500'], alpha=0.3)
                ax.plot(range(len(offset_diff)), offset_diff, color=COLORS['green_900'], linewidth=1.5)
                shade_delta(ax)
                ax.axhline(y=0, color=COLORS['green_900'], linewidth=0.8)
                ax.set_ylabel('Offset Gap', fontsize=9, color=COLORS['text_primary'])
                ax.set_xlabel('Timestep', fontsize=9)
                break
    else:
        for j in range(3):
            axes[2, j].set_visible(False)
    
    for row in axes:
        for ax in row:
            if ax.get_visible():
                ax.tick_params(labelsize=8)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
    
    fig.text(0.5, 0.005,
        'Shaded region: \u03b4-active window. Row 1: cumulative trajectories. Row 2: per-timestep raw signals. '
        'Row 3: ABO excess over baseline. '
        'v\u2083 pushes \u03c9 down (permissive); v\u2084 pushes \u03c9 up (conservative).',
        ha='center', fontsize=9, color=COLORS['text_muted'], style='italic')
    
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    return fig

# =============================================================================
# Full Parameters Sheet (v25)
# =============================================================================
def plot_full_parameters(scenarios: Dict[str, Tuple[ScenarioConfig, SimulationResults]]) -> plt.Figure:
    """Full parameter dump for reproducibility - included in PDF export"""
    fig, axes = plt.subplots(3, 1, figsize=(16, 18), gridspec_kw={'height_ratios': [1, 1, 0.8]})
    for ax in axes:
        ax.axis('off')
    
    # --- Table 1: Core Simulation Parameters ---
    core_cols = ['Parameter'] + list(scenarios.keys())
    core_rows = []
    param_list = [
        ('Timesteps', lambda c: str(c.n_steps)),
        ('Applicants/Step', lambda c: str(c.n_applicants_per_step)),
        ('Maturation Delay', lambda c: str(c.maturation_delay)),
        ('Noise Std', lambda c: f'{c.noise_std:.3f}'),
        ('Target Default Rate', lambda c: f'{c.target_default_rate:.2f}'),
        ('Human Reviewer Noise', lambda c: f'{c.human_noise_std:.3f}'),
        ('LOW/MED Threshold', lambda c: f'{c.low_thresh:.2f}'),
        ('MED/HIGH Threshold', lambda c: f'{c.high_thresh:.2f}'),
        ('Calibration Window', lambda c: str(c.calibration_window)),
        ('δ Enabled', lambda c: '✓' if c.delta_enabled else '✗'),
        ('δ Start', lambda c: str(c.delta_start) if c.delta_enabled else '-'),
        ('δ End', lambda c: str(c.delta_end) if c.delta_enabled else '-'),
        ('δ Min', lambda c: f'{c.delta_min:.3f}' if c.delta_enabled else '-'),
        ('δ Max', lambda c: f'{c.delta_max:.3f}' if c.delta_enabled else '-'),
    ]
    for label, fn in param_list:
        row = [label] + [fn(scenarios[n][0]) for n in scenarios]
        core_rows.append(row)
    
    t1 = axes[0].table(cellText=core_rows, colLabels=core_cols, loc='upper center', cellLoc='center')
    t1.auto_set_font_size(False); t1.set_fontsize(9); t1.scale(1.0, 1.6)
    for i in range(len(core_cols)):
        t1[(0, i)].set_facecolor(COLORS['green_900'])
        t1[(0, i)].set_text_props(color='white', fontweight='bold', fontsize=9)
    for r in range(len(core_rows)):
        for c in range(len(core_cols)):
            t1[(r+1, c)].set_facecolor(COLORS['silver_100'] if r % 2 == 0 else COLORS['white'])
    axes[0].set_title('Simulation Parameters', fontsize=14, fontweight='bold', 
                       color=COLORS['text_primary'], pad=15, loc='left')
    
    # --- Table 2: Feedback Parameters ---
    fb_cols = ['Parameter'] + list(scenarios.keys())
    fb_rows = []
    fb_params = [
        ('v4 Strength (Outcome FB)', lambda c: f'{c.v4_strength:.6f}'),
        ('v4 Asymmetry', lambda c: f'{c.v4_asymmetry:.3f}'),
        ('v3 Strength (Operational FB)', lambda c: f'{c.v3_strength:.6f}'),
        ('v3 Proportional', lambda c: '✓' if c.v3_proportional else '✗'),
        ('v3 Lookback Window', lambda c: str(c.v3_lookback)),
    ]
    for label, fn in fb_params:
        row = [label] + [fn(scenarios[n][0]) for n in scenarios]
        fb_rows.append(row)
    
    t2 = axes[1].table(cellText=fb_rows, colLabels=fb_cols, loc='upper center', cellLoc='center')
    t2.auto_set_font_size(False); t2.set_fontsize(9); t2.scale(1.0, 1.6)
    for i in range(len(fb_cols)):
        t2[(0, i)].set_facecolor(COLORS['green_900'])
        t2[(0, i)].set_text_props(color='white', fontweight='bold', fontsize=9)
    for r in range(len(fb_rows)):
        for c in range(len(fb_cols)):
            t2[(r+1, c)].set_facecolor(COLORS['silver_100'] if r % 2 == 0 else COLORS['white'])
    axes[1].set_title('Feedback & ISCIL Parameters', fontsize=14, fontweight='bold', 
                       color=COLORS['text_primary'], pad=15, loc='left')
    
    # ISCIL params in same table
    iscil_params = [
        ('ISCIL Enabled', lambda c: '✓' if c.iscil_enabled else '✗'),
        ('ISCIL Containment', lambda c: '✓' if c.iscil_containment_enabled else '✗'),
        ('ISCIL Baseline Window', lambda c: str(c.iscil_baseline_window) if c.iscil_enabled else '-'),
        ('ISCIL Threshold (θ)', lambda c: f'{c.iscil_threshold:.2f}' if c.iscil_enabled else '-'),
        ('ISCIL Sustained Window', lambda c: str(c.iscil_sustained_window) if c.iscil_enabled else '-'),
        ('ISCIL ROC Window (k)', lambda c: str(c.iscil_roc_window) if c.iscil_enabled else '-'),
        ('ISCIL Max Guardrail', lambda c: f'{c.iscil_max_guardrail_offset:.3f}' if c.iscil_containment_enabled else '-'),
        ('ISCIL Max Damping', lambda c: f'{c.iscil_max_feedback_damping:.2f}' if c.iscil_containment_enabled else '-'),
        ('ISCIL Weight: Approval', lambda c: f'{c.iscil_weight_approval:.2f}' if c.iscil_enabled else '-'),
        ('ISCIL Weight: Category', lambda c: f'{c.iscil_weight_category:.2f}' if c.iscil_enabled else '-'),
        ('ISCIL Weight: Escalation', lambda c: f'{c.iscil_weight_escalation:.2f}' if c.iscil_enabled else '-'),
        ('ISCIL Weight: Feedback', lambda c: f'{c.iscil_weight_feedback:.2f}' if c.iscil_enabled else '-'),
    ]
    iscil_rows = []
    for label, fn in iscil_params:
        row = [label] + [fn(scenarios[n][0]) for n in scenarios]
        iscil_rows.append(row)
    
    t3 = axes[1].table(cellText=iscil_rows, colLabels=fb_cols, loc='lower center', cellLoc='center')
    t3.auto_set_font_size(False); t3.set_fontsize(9); t3.scale(1.0, 1.6)
    for i in range(len(fb_cols)):
        t3[(0, i)].set_facecolor(COLORS['green_800'])
        t3[(0, i)].set_text_props(color='white', fontweight='bold', fontsize=9)
    for r in range(len(iscil_rows)):
        for c in range(len(fb_cols)):
            t3[(r+1, c)].set_facecolor(COLORS['silver_100'] if r % 2 == 0 else COLORS['white'])
    
    # --- Table 3: Financial Parameters ---
    fin_cols = ['Parameter'] + list(scenarios.keys())
    fin_params = [
        ('Revenue / Good Loan', lambda c: f'{c.financial.revenue_per_good_loan:.2f}'),
        ('Cost / Default', lambda c: f'{c.financial.cost_per_default:.2f}'),
        ('Opp Cost / Denied Non-Defaulter', lambda c: f'{c.financial.opportunity_cost_per_denial:.2f}'),
    ]
    fin_rows = []
    for label, fn in fin_params:
        row = [label] + [fn(scenarios[n][0]) for n in scenarios]
        fin_rows.append(row)
    
    t4 = axes[2].table(cellText=fin_rows, colLabels=fin_cols, loc='upper center', cellLoc='center')
    t4.auto_set_font_size(False); t4.set_fontsize(9); t4.scale(1.0, 1.6)
    for i in range(len(fin_cols)):
        t4[(0, i)].set_facecolor(COLORS['green_900'])
        t4[(0, i)].set_text_props(color='white', fontweight='bold', fontsize=9)
    for r in range(len(fin_rows)):
        for c in range(len(fin_cols)):
            t4[(r+1, c)].set_facecolor(COLORS['silver_100'] if r % 2 == 0 else COLORS['white'])
    axes[2].set_title('Financial Model', fontsize=14, fontweight='bold', 
                       color=COLORS['text_primary'], pad=15, loc='left')
    
    plt.tight_layout()
    return fig

# =============================================================================
# Export Functions
# =============================================================================
from io import BytesIO
from matplotlib.backends.backend_pdf import PdfPages
import xlsxwriter

def export_to_pdf(scenarios: Dict[str, Tuple[ScenarioConfig, SimulationResults]], 
                  applicant_pool: List[Applicant],
                  rolling_window: int = 20) -> BytesIO:
    """Export all figures to a PDF file."""
    buffer = BytesIO()
    
    with PdfPages(buffer) as pdf:
        # Page 0: Full Parameters Sheet (v25)
        fig_params = plot_full_parameters(scenarios)
        pdf.savefig(fig_params, bbox_inches='tight')
        plt.close(fig_params)
        
        # Page 1: Summary Table
        fig = plot_summary_table(scenarios)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Page 2: Decision Breakdown
        fig = plot_decision_breakdown(scenarios)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Page 3: Scenario Settings Table
        fig = plot_scenario_settings(scenarios)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Page 4: Applicant Pool
        fig = plot_applicant_pool(applicant_pool)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Page 5: Cumulative Rates
        fig = plot_cumulative_rates(scenarios)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Page 6: Time Dynamics
        fig = plot_time_dynamics(scenarios, rolling_window)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Page 7: ISCIL Analysis (if any scenario has ISCIL enabled)
        any_iscil = any(config.iscil_enabled for _, (config, _) in scenarios.items())
        if any_iscil:
            # ISCIL Summary Table
            fig = plot_iscil_summary_table(scenarios)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            
            # ISCIL Combined CRS and Intervention
            fig = plot_iscil_combined(scenarios, rolling_window=10)  # Use 10-step smoothing for PDF
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
        
        # Risk Distribution - DYNAMIC time windows based on actual delta config (v24 fix)
        first_config = list(scenarios.values())[0][0]
        max_steps = first_config.n_steps
        
        # Find the earliest delta_start and latest delta_end across all scenarios
        delta_starts = [cfg.delta_start for _, (cfg, _) in scenarios.items() if cfg.delta_enabled]
        delta_ends = [cfg.delta_end for _, (cfg, _) in scenarios.items() if cfg.delta_enabled]
        
        if delta_starts and delta_ends:
            d_start = min(delta_starts)
            d_end = max(delta_ends)
            recovery_end = min(d_end + (d_end - d_start), max_steps)
            
            time_windows = [
                (max(1, d_start - min(50, d_start - 1)), d_start - 1, "Before δ"),
                (d_start, d_end, "During δ"),
                (d_end + 1, recovery_end, "Recovery"),
            ]
            # Add late period if there's enough room
            if recovery_end + 50 < max_steps:
                time_windows.append((recovery_end + 1, min(recovery_end + (d_end - d_start), max_steps), "Late Period"))
        else:
            # No delta scenarios — use equal thirds
            third = max_steps // 3
            time_windows = [
                (1, third, "First Third"),
                (third + 1, 2 * third, "Middle Third"),
                (2 * third + 1, max_steps, "Final Third"),
            ]
        
        for start, end, label in time_windows:
            if 0 < start <= end <= max_steps:
                fig = plot_risk_distribution(scenarios, applicant_pool, start, end)
                # v25: use suptitle with y > 1 to push above subplot title
                fig.suptitle(f'Risk Distribution: {label} (t={start}-{end})', 
                            fontsize=13, fontweight='bold', y=1.03,
                            color=COLORS['text_primary'])
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
        
        # Jump Effects (for each scenario with delta enabled)
        for name, (config, results) in scenarios.items():
            if config.delta_enabled:
                fig = plot_jump_effects({name: (config, results)}, config)
                # v25: use suptitle with y > 1 to push above subplot titles
                fig.suptitle(f'Jump Effects Analysis: {name}', fontsize=13, 
                            fontweight='bold', y=1.02, color=COLORS['text_primary'])
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
        
        # v24: Financial Impact
        fig = plot_financial_impact(scenarios)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # v24: Confusion Matrix / Decision Accuracy
        fig = plot_confusion_matrix(scenarios)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # v0.1: Feedback Decomposition
        fig = plot_feedback_decomposition(scenarios)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    buffer.seek(0)
    return buffer

def export_to_excel(scenarios: Dict[str, Tuple[ScenarioConfig, SimulationResults]], 
                    applicant_pool: List[Applicant],
                    initial_window: int = 50) -> BytesIO:
    """Export all raw data to an Excel file."""
    buffer = BytesIO()
    
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        # Sheet 0: Full Parameters (v25)
        params_data = []
        for name, (config, results) in scenarios.items():
            params_data.append({
                'Scenario': name,
                'Timesteps': config.n_steps,
                'Applicants/Step': config.n_applicants_per_step,
                'Maturation Delay': config.maturation_delay,
                'Noise Std': config.noise_std,
                'Target Default Rate': config.target_default_rate,
                'Human Reviewer Noise': config.human_noise_std,
                'LOW/MED Threshold': config.low_thresh,
                'MED/HIGH Threshold': config.high_thresh,
                'Calibration Window': config.calibration_window,
                'δ Enabled': config.delta_enabled,
                'δ Start': config.delta_start if config.delta_enabled else None,
                'δ End': config.delta_end if config.delta_enabled else None,
                'δ Min': config.delta_min if config.delta_enabled else None,
                'δ Max': config.delta_max if config.delta_enabled else None,
                'v4 Strength': config.v4_strength,
                'v4 Asymmetry': config.v4_asymmetry,
                'v3 Strength': config.v3_strength,
                'v3 Proportional': config.v3_proportional,
                'v3 Lookback': config.v3_lookback,
                'ISCIL Enabled': config.iscil_enabled,
                'ISCIL Containment': config.iscil_containment_enabled,
                'ISCIL Baseline Window': config.iscil_baseline_window,
                'ISCIL Threshold': config.iscil_threshold,
                'ISCIL Sustained Window': config.iscil_sustained_window,
                'ISCIL ROC Window (k)': config.iscil_roc_window,
                'ISCIL Max Guardrail': config.iscil_max_guardrail_offset,
                'ISCIL Max Damping': config.iscil_max_feedback_damping,
                'ISCIL Weight Approval': config.iscil_weight_approval,
                'ISCIL Weight Category': config.iscil_weight_category,
                'ISCIL Weight Escalation': config.iscil_weight_escalation,
                'ISCIL Weight Feedback': config.iscil_weight_feedback,
                'Revenue/Good Loan': config.financial.revenue_per_good_loan,
                'Cost/Default': config.financial.cost_per_default,
                'Opp Cost/Denial': config.financial.opportunity_cost_per_denial,
            })
        df_params = pd.DataFrame(params_data)
        df_params.to_excel(writer, sheet_name='Parameters', index=False)
        
        # Sheet 1: Summary Metrics
        summary_data = []
        for name, (config, results) in scenarios.items():
            n = len(results.timestep_approval_rate)
            valid_defaults = [d for d in results.timestep_default_rate if d is not None]
            total_applicants = config.n_steps * config.n_applicants_per_step
            
            initial_approval = np.mean(results.timestep_approval_rate[:initial_window]) * 100 if n >= initial_window else 0
            final_approval = np.mean(results.timestep_approval_rate[-initial_window:]) * 100 if n >= initial_window else 0
            # Avg Approval = Total Approvals / Total Applicants
            avg_approval = (results.total_approvals / total_applicants * 100) if total_applicants > 0 else 0
            
            if len(valid_defaults) >= initial_window:
                initial_default = np.mean(valid_defaults[:initial_window]) * 100
                final_default = np.mean(valid_defaults[-initial_window:]) * 100
            else:
                initial_default = 0
                final_default = 0
            
            # Three CLOSED-WINDOW default rates
            cw_default_rate_of_matured = (results.closed_window_defaults / results.closed_window_approvals * 100) if results.closed_window_approvals > 0 else 0
            cw_default_rate_of_approved = (results.closed_window_defaults / results.closed_window_approvals * 100) if results.closed_window_approvals > 0 else 0
            cw_default_rate_of_total = (results.closed_window_defaults / results.closed_window_applicants * 100) if results.closed_window_applicants > 0 else 0
            
            drift = final_approval - initial_approval
            final_offset = results.calibration_offset_history[-1] if results.calibration_offset_history else 0
            
            summary_data.append({
                'Scenario': name,
                'Initial Approval (%)': initial_approval,
                'Final Approval (%)': final_approval,
                'Avg Approval (%)': avg_approval,
                'Initial Default (%)': initial_default,
                'Final Default (%)': final_default,
                'CW Default Rate (of Matured) (%)': cw_default_rate_of_matured,
                'CW Default Rate (of Approved) (%)': cw_default_rate_of_approved,
                'CW Default Rate (of Total) (%)': cw_default_rate_of_total,
                'Closed Window Applicants': results.closed_window_applicants,
                'Closed Window Approvals': results.closed_window_approvals,
                'Closed Window Defaults': results.closed_window_defaults,
                'Total Approvals (all)': results.total_approvals,
                'Total Matured Approvals': results.total_matured_approvals,
                'Total Defaults (all)': results.total_defaults,
                'Drift (%)': drift,
                'Final Offset': final_offset,
                'Delta Enabled': config.delta_enabled,
                'Delta Start': config.delta_start if config.delta_enabled else None,
                'Delta End': config.delta_end if config.delta_enabled else None,
                'Delta Min': config.delta_min if config.delta_enabled else None,
                'Delta Max': config.delta_max if config.delta_enabled else None,
                'v3 Strength': config.v3_strength,
                'v4 Strength': config.v4_strength,
                'Low Threshold': config.low_thresh,
                'High Threshold': config.high_thresh,
            })
        
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_excel(writer, sheet_name='Summary', index=False)
        
        # Sheet 2: Applicant Pool
        pool_data = [{
            'ID': a.id,
            'True Risk': a.true_risk,
            'Will Default': a.will_default,
        } for a in applicant_pool]
        df_pool = pd.DataFrame(pool_data)
        df_pool.to_excel(writer, sheet_name='Applicant Pool', index=False)
        
        # Sheet 3: Time Series Data (one sheet per scenario)
        for name, (config, results) in scenarios.items():
            n = len(results.timestep_approval_rate)
            ts_data = {
                'Timestep': list(range(1, n + 1)),
                'Approval Rate': results.timestep_approval_rate,
                'Default Rate': results.timestep_default_rate,
                'Cumulative Approval Rate': results.cumulative_approval_rate,
                'Cumulative Default Rate': results.cumulative_default_rate,
                'Calibration Offset': results.calibration_offset_history,
                'Delta Applied': results.delta_history,
                'v3 Cumulative': results.v3_cumulative_contribution if results.v3_cumulative_contribution else [0.0] * n,
                'v4 Cumulative': results.v4_cumulative_contribution if results.v4_cumulative_contribution else [0.0] * n,
                'v3 Per-Step Delta': results.v3_per_step_delta if results.v3_per_step_delta else [0.0] * n,
                'v4 Per-Step Delta': results.v4_per_step_delta if results.v4_per_step_delta else [0.0] * n,
            }
            df_ts = pd.DataFrame(ts_data)
            # Sanitize sheet name (max 31 chars, no special chars)
            sheet_name = f'TS_{name[:25]}'.replace('/', '_').replace('\\', '_')
            df_ts.to_excel(writer, sheet_name=sheet_name, index=False)
        
        # Sheet 4: Jump Events (all scenarios combined)
        all_jumps = []
        for name, (config, results) in scenarios.items():
            for event in results.jump_events:
                all_jumps.append({
                    'Scenario': name,
                    'Timestep': event['timestep'],
                    'Base Score': event['base_score'],
                    'Effective Signal': event['effective_signal'],
                    'Delta': event['delta'],
                    'Category Before': event['category_before'].name,
                    'Category After': event['category_after'].name,
                    'True Risk': event['true_risk'],
                    'Will Default': event['will_default'],
                    'Decision': event['decision'].name,
                })
        
        if all_jumps:
            df_jumps = pd.DataFrame(all_jumps)
            df_jumps.to_excel(writer, sheet_name='Jump Events', index=False)
        
        # Sheet 5: All Outcomes (one sheet per scenario, limited to first 10k rows)
        for name, (config, results) in scenarios.items():
            outcomes_data = []
            for o in results.all_outcomes[:10000]:  # Limit to avoid huge files
                outcomes_data.append({
                    'Timestep': o['timestep'],
                    'Applicant ID': o['applicant_id'],
                    'True Risk': o['true_risk'],
                    'Will Default': o['will_default'],
                    'Decision': o['decision'].name,
                    'Effective Signal': o['effective_signal'],
                    'Base Score': o['base_score'],
                    'Delta': o['delta'],
                    'Defaulted': o.get('defaulted', None),
                })
            if outcomes_data:
                df_outcomes = pd.DataFrame(outcomes_data)
                sheet_name = f'Out_{name[:24]}'.replace('/', '_').replace('\\', '_')
                df_outcomes.to_excel(writer, sheet_name=sheet_name, index=False)
        
        # Sheet 6: ISCIL Data (if any scenario has ISCIL enabled)
        any_iscil = any(config.iscil_enabled for _, (config, _) in scenarios.items())
        if any_iscil:
            # ISCIL Summary
            iscil_summary = []
            for name, (config, results) in scenarios.items():
                if config.iscil_enabled:
                    iscil_summary.append({
                        'Scenario': name,
                        'ISCIL Enabled': config.iscil_enabled,
                        'Containment Enabled': config.iscil_containment_enabled,
                        'Threshold': config.iscil_threshold,
                        'Baseline Window': config.iscil_baseline_window,
                        'Sustained Window': config.iscil_sustained_window,
                        'Max Guardrail': config.iscil_max_guardrail_offset,
                        'Max Damping': config.iscil_max_feedback_damping,
                        'First Alert Time': results.iscil_alert_time,
                        'Intervention Start Time': results.iscil_sustained_alert_time,
                        'Peak CRS': results.iscil_peak_crs,
                        'Intervention Timesteps': results.iscil_total_intervention_timesteps,
                    })
            
            if iscil_summary:
                df_iscil = pd.DataFrame(iscil_summary)
                df_iscil.to_excel(writer, sheet_name='ISCIL Summary', index=False)
            
            # ISCIL Time Series (one sheet per ISCIL-enabled scenario)
            for name, (config, results) in scenarios.items():
                if config.iscil_enabled and len(results.iscil_crs_history) > 0:
                    n = len(results.iscil_crs_history)
                    iscil_ts = {
                        'Timestep': list(range(1, n + 1)),
                        'CRS': results.iscil_crs_history,
                        'Intervention Strength': results.iscil_intervention_strength_history,
                        'Guardrail Offset': results.iscil_guardrail_offset_history,
                        'Drift Direction': results.iscil_drift_direction_history,
                    }
                    df_iscil_ts = pd.DataFrame(iscil_ts)
                    sheet_name = f'ISCIL_{name[:20]}'.replace('/', '_').replace('\\', '_')
                    df_iscil_ts.to_excel(writer, sheet_name=sheet_name, index=False)
    
    buffer.seek(0)
    return buffer

# =============================================================================
# Streamlit App
# =============================================================================
def main():
    st.set_page_config(
        page_title="ISE Simulator v0.1",
        page_icon="🔬",
        layout="wide",
    )
    
    st.title("🔬 ISE Simulator v0.1")
    st.markdown("**Ambiguity-Bearing Outputs (ABO) in Interconnected Systems Environments**")
    
    # Initialize session state
    if 'scenarios' not in st.session_state:
        st.session_state.scenarios = {}
    if 'applicant_pool' not in st.session_state:
        st.session_state.applicant_pool = None
    if 'results' not in st.session_state:
        st.session_state.results = {}
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Global settings
        st.subheader("Global Settings")
        n_steps = st.number_input("Simulation Steps", min_value=100, max_value=5000, value=1200, step=100)
        n_applicants = st.number_input("Applicants per Step", min_value=1, max_value=100, value=15)
        maturation_delay = st.number_input("Maturation Delay (timesteps)", min_value=10, max_value=200, value=90, step=10,
                                           help="Timesteps before loan outcome (default/no-default) is observed. Realistic: 60-90 (2-3 months)")
        pool_seed = st.number_input("Applicant Pool Seed", min_value=1, value=42)
        
        # Generate/regenerate pool
        if st.button("🔄 Generate Applicant Pool"):
            total_applicants = n_steps * n_applicants
            st.session_state.applicant_pool = generate_applicant_pool(total_applicants, seed=pool_seed)
            st.success(f"Generated {total_applicants:,} applicants")
        
        st.divider()
        
        # Scenario configuration
        st.subheader("📋 Scenario Configuration")
        
        num_scenarios = st.slider("Number of Scenarios", min_value=1, max_value=5, value=3)
        
        # Pre-populated defaults for 3 recommended scenarios
        PRESETS = {
            0: {"name": "Baseline", "delta_enabled": False, "delta_start": 500, "delta_end": 800, "delta_min": 0.0, "delta_max": 0.0,
                "v4_strength": 0.076, "v3_strength": 0.000110, "v4_asymmetry": 0.03, "v3_proportional": True,
                "human_noise": 0.05, "low_thresh": 0.38, "high_thresh": 0.62,
                "iscil_enabled": False, "iscil_containment": False,
                "iscil_threshold": 1.0, "iscil_baseline": 200, "iscil_sustained": 5, "iscil_roc_window": 15,
                "iscil_max_guardrail": 0.05, "iscil_max_damping": 0.50,
                "fin_revenue": 3.0, "fin_cost": 8.0, "fin_opp": 0.5},
            1: {"name": "ABO (no ISCIL)", "delta_enabled": True, "delta_start": 500, "delta_end": 800, "delta_min": -0.15, "delta_max": -0.10,
                "v4_strength": 0.076, "v3_strength": 0.000110, "v4_asymmetry": 0.03, "v3_proportional": True,
                "human_noise": 0.05, "low_thresh": 0.38, "high_thresh": 0.62,
                "iscil_enabled": False, "iscil_containment": False,
                "iscil_threshold": 1.0, "iscil_baseline": 200, "iscil_sustained": 5, "iscil_roc_window": 15,
                "iscil_max_guardrail": 0.05, "iscil_max_damping": 0.50,
                "fin_revenue": 3.0, "fin_cost": 8.0, "fin_opp": 0.5},
            2: {"name": "ABO + ISCIL", "delta_enabled": True, "delta_start": 500, "delta_end": 800, "delta_min": -0.15, "delta_max": -0.10,
                "v4_strength": 0.076, "v3_strength": 0.000110, "v4_asymmetry": 0.03, "v3_proportional": True,
                "human_noise": 0.05, "low_thresh": 0.38, "high_thresh": 0.62,
                "iscil_enabled": True, "iscil_containment": True,
                "iscil_threshold": 1.0, "iscil_baseline": 200, "iscil_sustained": 5, "iscil_roc_window": 15,
                "iscil_max_guardrail": 0.05, "iscil_max_damping": 0.50,
                "fin_revenue": 3.0, "fin_cost": 8.0, "fin_opp": 0.5},
        }
        
        scenarios_config = {}
        
        for i in range(num_scenarios):
            p = PRESETS.get(i, PRESETS[0])  # Fall back to baseline defaults for extra scenarios
            with st.expander(f"Scenario {i+1}", expanded=(i==0)):
                name = st.text_input(f"Name", value=p["name"], key=f"name_{i}")
                
                st.markdown("**Delta (ABO)**")
                delta_enabled = st.checkbox("Enable δ", value=p["delta_enabled"], key=f"delta_enabled_{i}")
                
                if delta_enabled:
                    col1, col2 = st.columns(2)
                    with col1:
                        delta_start = st.number_input("δ Start", min_value=0, max_value=n_steps, value=p["delta_start"], key=f"delta_start_{i}")
                    with col2:
                        delta_end = st.number_input("δ End", min_value=0, max_value=n_steps, value=p["delta_end"], key=f"delta_end_{i}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        delta_min = st.number_input("δ Min", min_value=-5.0, max_value=5.0, value=p["delta_min"], step=0.05, key=f"delta_min_{i}")
                    with col2:
                        delta_max = st.number_input("δ Max", min_value=-5.0, max_value=5.0, value=p["delta_max"], step=0.05, key=f"delta_max_{i}")
                else:
                    delta_start, delta_end, delta_min, delta_max = 50, 100, 0.0, 0.0
                
                st.markdown("**Feedback Loops**")
                col1, col2 = st.columns(2)
                with col1:
                    v4_strength = st.number_input("v4 Strength", min_value=0.0, max_value=5.0, value=p["v4_strength"], step=0.000001, format="%.6f", key=f"v4_{i}",
                                                  help="Outcome feedback (0 = disabled). Lower = slower correction.")
                with col2:
                    v3_strength = st.number_input("v3 Strength", min_value=0.0, max_value=0.01, value=p["v3_strength"], step=0.000001, format="%.6f", key=f"v3_{i}",
                                                  help="Operational feedback (0 = disabled)")
                
                col1, col2 = st.columns(2)
                with col1:
                    v4_asymmetry = st.number_input("v4 Asymmetry", min_value=0.0, max_value=1.0, value=p["v4_asymmetry"], step=0.05, key=f"v4_asym_{i}",
                                                    help="v24: Permissive correction strength (0.1 = permissive correction at 10% of conservative). Lower = more persistent permissive drift.")
                with col2:
                    v3_proportional = st.checkbox("v3 Proportional", value=p["v3_proportional"], key=f"v3_prop_{i}",
                                                  help="v24: v3 scales with recent approval rate (higher approval → stronger permissive pressure)")
                
                human_noise = st.number_input("Human Reviewer Noise", min_value=0.0, max_value=0.3, value=p["human_noise"], step=0.01, key=f"human_noise_{i}",
                                              help="v24: Noise std for human reviewer (0 = omniscient). Realistic: 0.03-0.08")
                
                st.markdown("**Thresholds**")
                col1, col2 = st.columns(2)
                with col1:
                    low_thresh = st.number_input("LOW/MED", min_value=0.0, max_value=1.0, value=p["low_thresh"], step=0.01, key=f"low_{i}")
                with col2:
                    high_thresh = st.number_input("MED/HIGH", min_value=0.0, max_value=1.0, value=p["high_thresh"], step=0.01, key=f"high_{i}")
                
                # ISCIL Configuration
                st.markdown("**ISCIL (Inter-System Coherence & Integrity Layer)**")
                iscil_enabled = st.checkbox("Enable ISCIL Monitoring", value=p["iscil_enabled"], key=f"iscil_enabled_{i}",
                                           help="Monitor cluster-level telemetry for drift detection")
                
                # v24: Financial config (collapsed by default)
                with st.expander("💰 Financial Impact Model"):
                    fin_col1, fin_col2, fin_col3 = st.columns(3)
                    with fin_col1:
                        fin_revenue = st.number_input("Revenue/good loan", min_value=0.0, value=p["fin_revenue"], step=0.1, key=f"fin_rev_{i}")
                    with fin_col2:
                        fin_cost = st.number_input("Cost/default", min_value=0.0, value=p["fin_cost"], step=0.5, key=f"fin_cost_{i}")
                    with fin_col3:
                        fin_opp = st.number_input("Opp cost/denial", min_value=0.0, value=p["fin_opp"], step=0.05, key=f"fin_opp_{i}")
                
                if iscil_enabled:
                    iscil_containment = st.checkbox("Enable Containment", value=p["iscil_containment"], key=f"iscil_containment_{i}",
                                                   help="Apply discretization guardrails and feedback damping")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        iscil_threshold = st.number_input("CRS Threshold", min_value=0.5, max_value=5.0, value=p["iscil_threshold"], step=0.1, key=f"iscil_thresh_{i}",
                                                         help="Coherence-Risk Score threshold for intervention")
                    with col2:
                        iscil_baseline = st.number_input("Baseline Window", min_value=20, max_value=2000, value=p["iscil_baseline"], key=f"iscil_baseline_{i}",
                                                        help="Timesteps for baseline establishment")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        iscil_sustained = st.number_input("Sustained Window", min_value=1, max_value=50, value=p["iscil_sustained"], key=f"iscil_sustained_{i}",
                                                         help="CRS must exceed threshold for this many consecutive timesteps before intervention starts")
                    with col2:
                        iscil_roc_window = st.number_input("ROC Window (k)", min_value=1, max_value=50, value=p["iscil_roc_window"], key=f"iscil_roc_{i}",
                                                          help="Window for rate-of-change computation. Larger k = smoother detection, smaller k = faster response")
                    
                    if iscil_containment:
                        col1, col2 = st.columns(2)
                        with col1:
                            iscil_max_guardrail = st.number_input("Max Guardrail", min_value=0.0, max_value=0.5, value=p["iscil_max_guardrail"], step=0.01, key=f"iscil_guardrail_{i}",
                                                                 help="Maximum discretization guardrail offset (0 = disabled)")
                        with col2:
                            iscil_max_damping = st.number_input("Max Damping", min_value=0.0, max_value=1.0, value=p["iscil_max_damping"], step=0.1, key=f"iscil_damping_{i}",
                                                              help="Maximum feedback damping factor (0 = disabled)")
                    else:
                        iscil_max_guardrail = p["iscil_max_guardrail"]
                        iscil_max_damping = p["iscil_max_damping"]
                        iscil_sustained = p["iscil_sustained"]
                else:
                    iscil_containment = False
                    iscil_threshold = p["iscil_threshold"]
                    iscil_baseline = p["iscil_baseline"]
                    iscil_sustained = p["iscil_sustained"]
                    iscil_roc_window = p["iscil_roc_window"]
                    iscil_max_guardrail = p["iscil_max_guardrail"]
                    iscil_max_damping = p["iscil_max_damping"]
                
                scenarios_config[name] = ScenarioConfig(
                    name=name,
                    n_steps=n_steps,
                    n_applicants_per_step=n_applicants,
                    maturation_delay=maturation_delay,
                    delta_enabled=delta_enabled,
                    delta_start=delta_start,
                    delta_end=delta_end,
                    delta_min=delta_min,
                    delta_max=delta_max,
                    v4_strength=v4_strength,
                    v4_asymmetry=v4_asymmetry,
                    v3_strength=v3_strength,
                    v3_proportional=v3_proportional,
                    human_noise_std=human_noise,
                    low_thresh=low_thresh,
                    high_thresh=high_thresh,
                    financial=FinancialConfig(
                        revenue_per_good_loan=fin_revenue,
                        cost_per_default=fin_cost,
                        opportunity_cost_per_denial=fin_opp,
                    ),
                    iscil_enabled=iscil_enabled,
                    iscil_containment_enabled=iscil_containment,
                    iscil_threshold=iscil_threshold,
                    iscil_baseline_window=iscil_baseline,
                    iscil_sustained_window=iscil_sustained,
                    iscil_roc_window=iscil_roc_window,
                    iscil_max_guardrail_offset=iscil_max_guardrail,
                    iscil_max_feedback_damping=iscil_max_damping,
                )
        
        st.divider()
        
        # Run simulation
        if st.button("▶️ Run Simulation", type="primary"):
            if st.session_state.applicant_pool is None:
                st.error("Please generate applicant pool first!")
            else:
                st.session_state.scenarios = scenarios_config
                st.session_state.results = {}
                
                progress = st.progress(0)
                for idx, (name, config) in enumerate(scenarios_config.items()):
                    with st.spinner(f"Running {name}..."):
                        sim = UnderwritingISE(config, st.session_state.applicant_pool, seed=123)
                        results = sim.run()
                        st.session_state.results[name] = (config, results)
                    progress.progress((idx + 1) / len(scenarios_config))
                
                st.success("Simulation complete!")
    
    # Main content area
    if st.session_state.results:
        # Export buttons at the top
        st.subheader("📥 Export Results")
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            pdf_buffer = export_to_pdf(
                st.session_state.results, 
                st.session_state.applicant_pool,
                rolling_window=20
            )
            st.download_button(
                label="📄 Download PDF (Figures)",
                data=pdf_buffer,
                file_name="ise_simulation_results.pdf",
                mime="application/pdf"
            )
        
        with col2:
            excel_buffer = export_to_excel(
                st.session_state.results,
                st.session_state.applicant_pool
            )
            st.download_button(
                label="📊 Download Excel (Raw Data)",
                data=excel_buffer,
                file_name="ise_simulation_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        st.divider()
        
        # Tabs for different views
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
            "📊 Summary", 
            "👥 Applicant Pool", 
            "📈 Cumulative Rates",
            "⏱️ Time Dynamics",
            "🔀 Jump Effects",
            "💰 Financial Impact",
            "🎯 Decision Quality",
            "🛡️ ISCIL",
            "⚡ Feedback Dynamics"
        ])
        
        with tab1:
            st.subheader("Summary Metrics")
            fig = plot_summary_table(st.session_state.results)
            st.pyplot(fig)
            plt.close(fig)
        
        with tab2:
            st.subheader("Applicant Pool Distribution")
            if st.session_state.applicant_pool:
                fig = plot_applicant_pool(st.session_state.applicant_pool)
                st.pyplot(fig)
                plt.close(fig)
        
        with tab3:
            st.subheader("Cumulative Rates Over Time")
            fig = plot_cumulative_rates(st.session_state.results)
            st.pyplot(fig)
            plt.close(fig)
        
        with tab4:
            st.subheader("Time-Based Dynamics")
            rolling_window = st.slider("Rolling Average Window", min_value=5, max_value=50, value=20)
            fig = plot_time_dynamics(st.session_state.results, rolling_window)
            st.pyplot(fig)
            plt.close(fig)
            
            # Graph E: Risk Distribution Comparison
            st.subheader("E. Risk Distribution: True Risk vs Effective Signal")
            
            # Get max timesteps from first scenario
            first_config = list(st.session_state.results.values())[0][0]
            max_steps = first_config.n_steps
            
            col1, col2 = st.columns(2)
            with col1:
                ts_start = st.number_input("Timestep Start", min_value=1, max_value=max_steps, value=1)
            with col2:
                ts_end = st.number_input("Timestep End", min_value=1, max_value=max_steps, value=min(100, max_steps))
            
            if ts_start <= ts_end:
                fig_e = plot_risk_distribution(
                    st.session_state.results,
                    st.session_state.applicant_pool,
                    ts_start, ts_end
                )
                st.pyplot(fig_e)
                plt.close(fig_e)
            else:
                st.error("Start timestep must be ≤ End timestep")
        
        with tab5:
            st.subheader("Jump Effects Analysis")
            
            # Scenario selector
            scenario_names = list(st.session_state.results.keys())
            selected_scenario = st.selectbox(
                "Select scenario to analyze:",
                scenario_names,
                index=0
            )
            
            selected_config, selected_results = st.session_state.results[selected_scenario]
            
            if not selected_config.delta_enabled:
                st.warning(f"⚠️ '{selected_scenario}' has δ disabled. No jump events expected.")
            
            fig = plot_jump_effects({selected_scenario: (selected_config, selected_results)}, selected_config)
            st.pyplot(fig)
            plt.close(fig)
        
        with tab6:
            st.subheader("💰 Financial Impact Analysis")
            fig = plot_financial_impact(st.session_state.results)
            st.pyplot(fig)
            plt.close(fig)
        
        with tab7:
            st.subheader("🎯 Decision Quality (Confusion Matrix)")
            st.caption("Based on closed-window outcomes only (all loans matured)")
            fig = plot_confusion_matrix(st.session_state.results)
            st.pyplot(fig)
            plt.close(fig)
        
        with tab8:
            st.subheader("🛡️ ISCIL Analysis")
            
            # v23 explanation
            with st.expander("ℹ️ v23: Rate-of-Change Detection (ΔCRS)", expanded=False):
                st.markdown("""
                **v23 Change:** ISCIL now uses **rate-of-change** instead of absolute deviation from baseline.
                
                - **Old behavior:** CRS measured how far current values are from initial baseline → triggered on natural drift
                - **New behavior:** CRS measures how fast values are changing → triggers only on sudden shifts (like ABO attacks)
                
                **Key parameter:** `ROC Window (k)` controls how many timesteps are used to compute the rate of change.
                - Larger k = smoother detection, less sensitive to noise
                - Smaller k = faster response, more sensitive to rapid changes
                
                **Result:** Intervention ends naturally when the system stabilizes, even if at a new equilibrium.
                """)
            
            # Check if any scenario has ISCIL enabled
            any_iscil = any(config.iscil_enabled for _, (config, _) in st.session_state.results.items())
            
            if any_iscil:
                # Rolling average slider for ISCIL graphs
                iscil_rolling_window = st.slider(
                    "Rolling Average Window (for CRS & Intervention graphs)", 
                    min_value=1, max_value=50, value=1, 
                    help="Smooth noisy timestep data with rolling average. 1 = no smoothing."
                )
                
                # ISCIL Summary Table
                st.markdown("### ISCIL Summary")
                fig_summary = plot_iscil_summary_table(st.session_state.results)
                st.pyplot(fig_summary)
                plt.close(fig_summary)
                
                st.markdown("---")
                
                # Combined CRS and Intervention plot with rolling average
                st.markdown("### Coherence-Risk Score & Intervention")
                fig_combined = plot_iscil_combined(st.session_state.results, rolling_window=iscil_rolling_window)
                st.pyplot(fig_combined)
                plt.close(fig_combined)
                
                # Show ISCIL details
                st.markdown("---")
                st.markdown("### ISCIL Configuration Details")
                for name, (config, results) in st.session_state.results.items():
                    if config.iscil_enabled:
                        with st.expander(f"📋 {name} - ISCIL Details"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("**Configuration:**")
                                st.write(f"- Monitoring: ✓ Enabled")
                                st.write(f"- Containment: {'✓ Enabled' if config.iscil_containment_enabled else '✗ Disabled'}")
                                st.write(f"- Threshold (θ): {config.iscil_threshold}")
                                st.write(f"- Baseline Window: {config.iscil_baseline_window} timesteps")
                                st.write(f"- Sustained Window: {config.iscil_sustained_window} timesteps")
                                st.write(f"- ROC Window (k): {config.iscil_roc_window} timesteps")
                                if config.iscil_containment_enabled:
                                    st.write(f"- Max Guardrail: ±{config.iscil_max_guardrail_offset}")
                                    st.write(f"- Max Damping: {config.iscil_max_feedback_damping*100:.0f}%")
                            with col2:
                                st.markdown("**Results:**")
                                st.write(f"- First Alert: {'t=' + str(results.iscil_alert_time) if results.iscil_alert_time else 'None'}")
                                st.write(f"- Intervention Start: {'t=' + str(results.iscil_sustained_alert_time) if results.iscil_sustained_alert_time else 'None'}")
                                st.write(f"- Peak CRS: {results.iscil_peak_crs:.2f}")
                                st.write(f"- Intervention Timesteps: {results.iscil_total_intervention_timesteps}")
                                st.write(f"- Total Defaults: {results.total_defaults:,}")
                                st.write(f"- Matured Approvals: {results.total_matured_approvals:,}")
                                default_rate = (results.total_defaults / results.total_matured_approvals * 100) if results.total_matured_approvals > 0 else 0
                                st.write(f"- Default Rate (of Matured): {default_rate:.1f}%")
            else:
                st.info("No scenarios have ISCIL enabled. Enable ISCIL monitoring in the sidebar to see analysis.")
    
        with tab9:
            st.subheader("⚡ Feedback Dynamics — v₃/v₄ Decomposition")
            st.caption("Cumulative contributions of v₃ (permissive) and v₄ (corrective) feedback to calibration offset")
            fb_smoothing = st.slider("Smoothing window (Row 2: per-timestep signals)", 
                                     min_value=1, max_value=50, value=1, step=1,
                                     help="1 = raw signals, higher = rolling average to smooth noise",
                                     key="fb_smoothing")
            fig = plot_feedback_decomposition(st.session_state.results, smoothing_window=fb_smoothing)
            st.pyplot(fig)
            plt.close(fig)
    
    else:
        st.info("👈 Configure scenarios in the sidebar and click **Run Simulation** to see results.")

if __name__ == "__main__":
    main()
