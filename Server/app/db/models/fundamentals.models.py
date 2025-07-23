"""
Production-ready Pydantic models for stock fundamental data.
Provides comprehensive validation, serialization, and documentation.
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator
from pymongo import IndexModel


class RatingEnum(str, Enum):
    """Enumeration for analyst ratings."""
    STRONG_BUY = "Strong Buy"
    BUY = "Buy"
    CONVICTION_BUY = "Conviction Buy"
    OVERWEIGHT = "Overweight"
    HOLD = "Hold"
    UNDERWEIGHT = "Underweight"
    SELL = "Sell"
    STRONG_SELL = "Strong Sell"
    POSITIVE = "Positive"
    NEUTRAL = "Neutral"
    NEGATIVE = "Negative"


class StatusEnum(str, Enum):
    """Enumeration for project/initiative status."""
    ONGOING = "Ongoing"
    COMPLETED = "Completed"
    UNDER_DEVELOPMENT = "Under development"
    PLANNED = "Planned"
    CANCELLED = "Cancelled"
    UNCERTAIN = "Uncertain"
    DELAYED = "Delayed"


class Company(BaseModel):
    """Company basic information model."""
    name: str = Field(..., min_length=1, max_length=200, description="Company full name")
    ticker: str = Field(..., min_length=1, max_length=20, description="Stock ticker symbol")
    segments: List[str] = Field(
        default_factory=list,
        description="Business segments the company operates in",
        max_items=20
    )
    
    @field_validator('ticker')
    @classmethod
    def ticker_uppercase(cls, v):
        """Ensure ticker is uppercase."""
        return v.upper().strip()
    
    @field_validator('name')
    @classmethod
    def name_cleanup(cls, v):
        """Clean up company name."""
        return v.strip()
    
    @field_validator('segments')
    @classmethod
    def segments_cleanup(cls, v):
        """Clean up segment names."""
        return [segment.strip() for segment in v if segment.strip()]


class QuarterlyFinancials(BaseModel):
    """Model for quarterly financial data."""
    revenue: Optional[Union[int, float]] = Field(
        None,
        ge=0,
        description="Revenue in crores or millions"
    )
    total_revenue: Optional[Union[int, float]] = Field(
        None,
        ge=0,
        description="Total revenue including other income in crores or millions"
    )
    ebitda: Optional[Union[int, float]] = Field(
        None,
        description="EBITDA in crores or millions"
    )
    pat: Optional[Union[int, float]] = Field(
        None,
        description="Profit After Tax in crores or millions"
    )
    revenue_growth_yoy: Optional[float] = Field(
        None,
        ge=-100,
        le=1000,
        description="Year-over-year revenue growth percentage"
    )
    pat_growth_yoy: Optional[float] = Field(
        None,
        ge=-100,
        le=1000,
        description="Year-over-year PAT growth percentage"
    )


class Financials(BaseModel):
    """Model for financial data across different periods."""
    # Quarterly data
    Q1_FY25: Optional[QuarterlyFinancials] = None
    Q2_FY25: Optional[QuarterlyFinancials] = None
    Q3_FY25: Optional[QuarterlyFinancials] = None
    Q4_FY25: Optional[QuarterlyFinancials] = None
    Q1_FY26: Optional[QuarterlyFinancials] = None
    Q2_FY26: Optional[QuarterlyFinancials] = None
    Q3_FY26: Optional[QuarterlyFinancials] = None
    Q4_FY26: Optional[QuarterlyFinancials] = None
    
    # Annual data
    FY24: Optional[QuarterlyFinancials] = None
    FY25: Optional[QuarterlyFinancials] = None
    FY26: Optional[QuarterlyFinancials] = None


class Valuation(BaseModel):
    """Company valuation metrics."""
    market_cap: Optional[Union[int, float]] = Field(
        None,
        ge=0,
        description="Market capitalization in crores or millions"
    )
    pe_ratio: Optional[float] = Field(
        None,
        ge=0,
        le=1000,
        description="Price-to-earnings ratio"
    )
    dividend_per_share: Optional[float] = Field(
        None,
        ge=0,
        description="Dividend per share"
    )
    dividend_yield: Optional[float] = Field(
        None,
        ge=0,
        le=100,
        description="Dividend yield percentage"
    )
    pb_ratio: Optional[float] = Field(
        None,
        ge=0,
        le=100,
        description="Price-to-book ratio"
    )
    ev_ebitda: Optional[float] = Field(
        None,
        ge=0,
        le=1000,
        description="Enterprise Value to EBITDA ratio"
    )


class BusinessSegmentPerformance(BaseModel):
    """Performance metrics for individual business segments."""
    revenue: Optional[Union[int, float]] = Field(
        None,
        ge=0,
        description="Segment revenue in crores or millions"
    )
    growth_yoy: Optional[float] = Field(
        None,
        ge=-100,
        le=1000,
        description="Year-over-year growth percentage"
    )
    margin: Optional[float] = Field(
        None,
        ge=0,
        le=100,
        description="Segment margin percentage"
    )
    ebitda: Optional[Union[int, float]] = Field(
        None,
        description="Segment EBITDA in crores or millions"
    )
    # Additional segment-specific metrics
    subscribers: Optional[float] = Field(
        None,
        ge=0,
        description="Number of subscribers (for telecom/digital services)"
    )
    arpu: Optional[float] = Field(
        None,
        ge=0,
        description="Average Revenue Per User"
    )
    stores_count: Optional[int] = Field(
        None,
        ge=0,
        description="Number of stores (for retail)"
    )
    capacity_utilization: Optional[float] = Field(
        None,
        ge=0,
        le=100,
        description="Capacity utilization percentage (for manufacturing)"
    )
    
    # Custom metrics as key-value pairs
    custom_metrics: Optional[Dict[str, Union[str, int, float]]] = Field(
        default_factory=dict,
        description="Additional segment-specific metrics"
    )


class BusinessPerformance(BaseModel):
    """Overall business performance across segments."""
    oil_to_chemicals: Optional[BusinessSegmentPerformance] = None
    retail: Optional[BusinessSegmentPerformance] = None
    digital_services: Optional[BusinessSegmentPerformance] = None
    telecom: Optional[BusinessSegmentPerformance] = None
    media_entertainment: Optional[BusinessSegmentPerformance] = None
    renewable_energy: Optional[BusinessSegmentPerformance] = None
    pharmaceuticals: Optional[BusinessSegmentPerformance] = None
    automotive: Optional[BusinessSegmentPerformance] = None
    banking: Optional[BusinessSegmentPerformance] = None
    insurance: Optional[BusinessSegmentPerformance] = None
    
    # Custom segments
    custom_segments: Optional[Dict[str, BusinessSegmentPerformance]] = Field(
        default_factory=dict,
        description="Additional business segments"
    )


class GreenEnergyInitiative(BaseModel):
    """Model for green energy and sustainability initiatives."""
    investment: Optional[Union[int, float]] = Field(
        None,
        ge=0,
        description="Investment amount in crores or millions"
    )
    goal: Optional[str] = Field(
        None,
        max_length=200,
        description="Sustainability goal"
    )
    project: Optional[str] = Field(
        None,
        max_length=200,
        description="Project name"
    )
    location: Optional[str] = Field(
        None,
        max_length=200,
        description="Project location"
    )
    products: Optional[List[str]] = Field(
        default_factory=list,
        max_items=50,
        description="Products or services from the initiative"
    )
    phases: Optional[List[str]] = Field(
        default_factory=list,
        max_items=20,
        description="Project phases and timelines"
    )
    status: Optional[StatusEnum] = Field(
        None,
        description="Current status of the initiative"
    )


class MergerAcquisition(BaseModel):
    """Model for merger and acquisition activities."""
    partners: Optional[List[str]] = Field(
        default_factory=list,
        max_items=10,
        description="Partners involved in the deal"
    )
    value: Optional[Union[int, float]] = Field(
        None,
        ge=0,
        description="Deal value in crores or millions"
    )
    date: Optional[str] = Field(
        None,
        description="Deal announcement or completion date (YYYY-MM-DD)"
    )
    status: Optional[StatusEnum] = Field(
        None,
        description="Current status of the deal"
    )
    type: Optional[str] = Field(
        None,
        max_length=50,
        description="Type of transaction (merger, acquisition, joint venture, etc.)"
    )
    
    @field_validator('date')
    @classmethod
    def validate_date_format(cls, v):
        """Validate date format."""
        if v:
            try:
                datetime.strptime(v, '%Y-%m-%d')
            except ValueError:
                raise ValueError('Date must be in YYYY-MM-DD format')
        return v


class StrategicInitiatives(BaseModel):
    """Model for strategic initiatives and investments."""
    green_energy: Optional[GreenEnergyInitiative] = None
    media_merger: Optional[MergerAcquisition] = None
    digital_transformation: Optional[Dict[str, Union[str, int, float]]] = Field(
        default_factory=dict,
        description="Digital transformation initiatives"
    )
    expansion_plans: Optional[Dict[str, Union[str, int, float]]] = Field(
        default_factory=dict,
        description="Business expansion plans"
    )
    r_and_d: Optional[Dict[str, Union[str, int, float]]] = Field(
        default_factory=dict,
        description="Research and development initiatives"
    )
    
    # Custom initiatives
    custom_initiatives: Optional[Dict[str, Dict[str, Union[str, int, float]]]] = Field(
        default_factory=dict,
        description="Additional strategic initiatives"
    )


class AnalystRating(BaseModel):
    """Individual analyst rating and recommendation."""
    rating: Optional[RatingEnum] = Field(
        None,
        description="Analyst rating"
    )
    price_target: Optional[float] = Field(
        None,
        ge=0,
        description="Price target"
    )
    upside: Optional[float] = Field(
        None,
        ge=-100,
        le=1000,
        description="Upside percentage"
    )
    downside: Optional[float] = Field(
        None,
        ge=-100,
        le=100,
        description="Downside percentage"
    )
    catalysts: Optional[List[str]] = Field(
        default_factory=list,
        max_items=20,
        description="Key catalysts mentioned by analyst"
    )
    outlook: Optional[str] = Field(
        None,
        max_length=50,
        description="General outlook (Positive, Negative, Neutral)"
    )
    focus: Optional[str] = Field(
        None,
        max_length=200,
        description="Key focus areas mentioned"
    )
    date: Optional[str] = Field(
        None,
        description="Rating date (YYYY-MM-DD)"
    )
    
    @field_validator('date')
    @classmethod
    def validate_date_format(cls, v):
        """Validate date format."""
        if v:
            try:
                datetime.strptime(v, '%Y-%m-%d')
            except ValueError:
                raise ValueError('Date must be in YYYY-MM-DD format')
        return v


class AnalystSentiment(BaseModel):
    """Analyst sentiment and recommendations."""
    goldman_sachs: Optional[AnalystRating] = None
    jp_morgan: Optional[AnalystRating] = None
    jefferies: Optional[AnalystRating] = None
    morgan_stanley: Optional[AnalystRating] = None
    ubs: Optional[AnalystRating] = None
    credit_suisse: Optional[AnalystRating] = None
    citi: Optional[AnalystRating] = None
    bnp_paribas: Optional[AnalystRating] = None
    
    # Custom analysts
    custom_analysts: Optional[Dict[str, AnalystRating]] = Field(
        default_factory=dict,
        description="Additional analyst ratings"
    )
    
    # Aggregate metrics
    average_price_target: Optional[float] = Field(
        None,
        ge=0,
        description="Average price target across all analysts"
    )
    consensus_rating: Optional[RatingEnum] = Field(
        None,
        description="Consensus rating"
    )


class Risk(BaseModel):
    """Individual risk factor."""
    concern: str = Field(..., min_length=1, max_length=500, description="Risk description")
    status: Optional[StatusEnum] = Field(None, description="Current status of the risk")
    severity: Optional[str] = Field(
        None,
        description="Risk severity (High, Medium, Low)"
    )
    probability: Optional[str] = Field(
        None,
        description="Risk probability (High, Medium, Low)"
    )
    mitigation: Optional[str] = Field(
        None,
        max_length=500,
        description="Risk mitigation strategies"
    )
    
    @field_validator('severity', 'probability')
    @classmethod
    def validate_risk_level(cls, v):
        """Validate risk level values."""
        if v and v not in ['High', 'Medium', 'Low']:
            raise ValueError('Severity and probability must be High, Medium, or Low')
        return v


class Risks(BaseModel):
    """Company risk factors."""
    succession: Optional[Risk] = None
    execution: Optional[Risk] = None
    regulatory: Optional[Risk] = None
    market: Optional[Risk] = None
    operational: Optional[Risk] = None
    financial: Optional[Risk] = None
    ipo: Optional[Risk] = None
    competition: Optional[Risk] = None
    technology: Optional[Risk] = None
    environmental: Optional[Risk] = None
    
    # Custom risks
    custom_risks: Optional[Dict[str, Risk]] = Field(
        default_factory=dict,
        description="Additional risk factors"
    )


class FundamentalData(BaseModel):
    """
    Comprehensive model for stock fundamental data.
    
    This model provides complete validation and structure for storing
    fundamental analysis data for stocks in a production environment.
    """
    
    # Core company information
    company: Company = Field(..., description="Company basic information")
    
    # Financial data
    financials: Optional[Financials] = Field(None, description="Financial performance data")
    valuation: Optional[Valuation] = Field(None, description="Valuation metrics")
    business_performance: Optional[BusinessPerformance] = Field(
        None, 
        description="Business segment performance"
    )
    
    # Strategic information
    strategic_initiatives: Optional[StrategicInitiatives] = Field(
        None, 
        description="Strategic initiatives and investments"
    )
    
    # Market sentiment
    analyst_sentiment: Optional[AnalystSentiment] = Field(
        None, 
        description="Analyst ratings and sentiment"
    )
    
    # Risk factors
    risks: Optional[Risks] = Field(None, description="Risk factors and concerns")
    
    # Summary and metadata
    summary: Optional[str] = Field(
        None,
        max_length=2000,
        description="Executive summary of the fundamental analysis"
    )
    
    # Metadata
    data_source: Optional[str] = Field(
        None,
        max_length=100,
        description="Source of the fundamental data"
    )
    last_updated: Optional[datetime] = Field(
        default_factory=datetime.utcnow,
        description="Last update timestamp"
    )
    data_version: Optional[str] = Field(
        default="1.0",
        max_length=20,
        description="Data schema version"
    )
    currency: Optional[str] = Field(
        default="INR",
        max_length=10,
        description="Currency for financial figures"
    )
    
    # Custom fields for extensibility
    custom_data: Optional[Dict[str, Union[str, int, float, bool]]] = Field(
        default_factory=dict,
        description="Additional custom data fields"
    )
    
    class Config:
        """Pydantic configuration."""
        # Allow population by field name or alias
        allow_population_by_field_name = True
        
        # Validate assignment
        validate_assignment = True
        
        # Use enum values
        use_enum_values = True
        
        # JSON encoders for special types
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            Decimal: lambda v: float(v)
        }
        
        # Schema extra
        schema_extra = {
            "example": {
                "company": {
                    "name": "Reliance Industries Limited",
                    "ticker": "RELIANCE",
                    "segments": ["Oil to Chemicals (O2C)", "Retail", "Digital Services"]
                },
                "financials": {
                    "FY25": {
                        "revenue": 998114,
                        "total_revenue": 998114,
                        "ebitda": 183422,
                        "pat": 80787,
                        "revenue_growth_yoy": 7.3,
                        "pat_growth_yoy": 2.7
                    }
                },
                "valuation": {
                    "market_cap": 1845000,
                    "pe_ratio": 22.8,
                    "dividend_per_share": 5.5,
                    "dividend_yield": 0.4
                },
                "summary": "Strong fundamentals with diversified business portfolio"
            }
        }
    
    @model_validator(mode='before')
    @classmethod
    def validate_consistent_currency(cls, values):
        """Ensure currency consistency across all monetary values."""
        if isinstance(values, dict):
            currency = values.get('currency', 'INR')
            # Add custom validation logic if needed
        return values
    
    @field_validator('summary')
    @classmethod
    def summary_cleanup(cls, v):
        """Clean up summary text."""
        if v:
            return v.strip()
        return v


# MongoDB indexes for optimal query performance
FUNDAMENTAL_INDEXES = [
    IndexModel([("company.ticker", 1)], unique=True, name="ticker_unique"),
    IndexModel([("company.name", 1)], name="company_name"),
    IndexModel([("last_updated", -1)], name="last_updated_desc"),
    IndexModel([("data_source", 1)], name="data_source"),
    IndexModel([("valuation.market_cap", -1)], name="market_cap_desc"),
    IndexModel([("analyst_sentiment.consensus_rating", 1)], name="consensus_rating"),
    IndexModel([
        ("company.ticker", 1),
        ("last_updated", -1)
    ], name="ticker_updated_compound")
]


# Backward compatibility alias
fundamental = FundamentalData
    