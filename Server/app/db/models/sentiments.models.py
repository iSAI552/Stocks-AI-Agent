"""
Production-ready Pydantic v2 models for stock sentiment analysis data.
Provides comprehensive validation, serialization, and documentation for sentiment analysis.
"""

from datetime import datetime, date as Date
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator
from pymongo import IndexModel


class SentimentEnum(str, Enum):
    """Enumeration for sentiment classifications."""
    VERY_POSITIVE = "Very Positive"
    POSITIVE = "Positive"
    NEUTRAL = "Neutral"
    NEGATIVE = "Negative"
    VERY_NEGATIVE = "Very Negative"
    BULLISH = "Bullish"
    BEARISH = "Bearish"
    MIXED = "Mixed"


class ImpactEnum(str, Enum):
    """Enumeration for impact levels."""
    HIGH = "High"
    MODERATE = "Moderate"
    LOW = "Low"
    CRITICAL = "Critical"
    MINIMAL = "Minimal"


class SourceTypeEnum(str, Enum):
    """Enumeration for source types."""
    NEWS = "news"
    SOCIAL_MEDIA = "social_media"
    ANALYST_REPORT = "analyst_report"
    EARNINGS_CALL = "earnings_call"
    PRESS_RELEASE = "press_release"
    BLOG = "blog"
    FORUM = "forum"
    WEBSITE = "website"
    OTHER = "other"


class SentimentFactor(BaseModel):
    """Model for individual sentiment factors (positive/negative/neutral)."""
    
    factor: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Brief title of the sentiment factor"
    )
    description: str = Field(
        ...,
        min_length=10,
        max_length=1000,
        description="Detailed description of the factor"
    )
    source: Optional[str] = Field(
        None,
        max_length=500,
        description="Source(s) of information"
    )
    impact: ImpactEnum = Field(
        ...,
        description="Impact level of this factor"
    )
    reference: Optional[List[str]] = Field(
        default_factory=list,
        max_items=20,
        description="Reference IDs or citations"
    )
    
    # Additional metadata
    sentiment_score: Optional[float] = Field(
        None,
        ge=-1,
        le=1,
        description="Sentiment score for this specific factor (-1 to 1)"
    )
    confidence: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="Confidence in this factor's analysis (0 to 1)"
    )
    date_identified: Optional[Date] = Field(
        None,
        description="Date when this factor was identified"
    )
    category: Optional[str] = Field(
        None,
        max_length=50,
        description="Category of the factor (e.g., 'earnings', 'market', 'regulatory')"
    )
    keywords: Optional[List[str]] = Field(
        default_factory=list,
        max_items=10,
        description="Keywords associated with this factor"
    )
    
    @field_validator('factor', 'description')
    @classmethod
    def cleanup_text(cls, v):
        """Clean up text fields."""
        return v.strip() if v else v
    
    @field_validator('keywords', 'reference')
    @classmethod
    def cleanup_lists(cls, v):
        """Clean up list items."""
        if v:
            return [item.strip() for item in v if item.strip()]
        return v


class SentimentBreakdown(BaseModel):
    """Model for sentiment breakdown into positive, negative, and neutral factors."""
    
    positive_factors: Optional[List[SentimentFactor]] = Field(
        default_factory=list,
        max_items=50,
        description="Factors contributing to positive sentiment"
    )
    negative_factors: Optional[List[SentimentFactor]] = Field(
        default_factory=list,
        max_items=50,
        description="Factors contributing to negative sentiment"
    )
    neutral_factors: Optional[List[SentimentFactor]] = Field(
        default_factory=list,
        max_items=50,
        description="Neutral factors affecting sentiment"
    )
    
    # Aggregated scores
    positive_score: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="Aggregated positive sentiment score"
    )
    negative_score: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="Aggregated negative sentiment score"
    )
    neutral_score: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="Aggregated neutral sentiment score"
    )
    
    # Factor counts
    total_positive_factors: Optional[int] = Field(
        None,
        ge=0,
        description="Total number of positive factors"
    )
    total_negative_factors: Optional[int] = Field(
        None,
        ge=0,
        description="Total number of negative factors"
    )
    total_neutral_factors: Optional[int] = Field(
        None,
        ge=0,
        description="Total number of neutral factors"
    )


class MarketContext(BaseModel):
    """Model for market context and current stock information."""
    
    current_price: Optional[str] = Field(
        None,
        max_length=50,
        description="Current stock price with currency symbol"
    )
    current_price_numeric: Optional[float] = Field(
        None,
        ge=0,
        description="Current stock price as numeric value"
    )
    
    # Price ranges
    day_high: Optional[float] = Field(None, ge=0, description="Day's high price")
    day_low: Optional[float] = Field(None, ge=0, description="Day's low price")
    week_52_high: Optional[str] = Field(None, max_length=50, description="52-week high")
    week_52_low: Optional[str] = Field(None, max_length=50, description="52-week low")
    week_52_high_numeric: Optional[float] = Field(None, ge=0, description="52-week high (numeric)")
    week_52_low_numeric: Optional[float] = Field(None, ge=0, description="52-week low (numeric)")
    
    # Market metrics
    market_cap: Optional[str] = Field(
        None,
        max_length=100,
        description="Market capitalization with currency"
    )
    market_cap_numeric: Optional[float] = Field(
        None,
        ge=0,
        description="Market capitalization as numeric value"
    )
    volume: Optional[int] = Field(None, ge=0, description="Trading volume")
    avg_volume: Optional[int] = Field(None, ge=0, description="Average trading volume")
    
    # Valuation metrics
    pe_ratio: Optional[float] = Field(None, ge=0, description="Price-to-earnings ratio")
    pb_ratio: Optional[float] = Field(None, ge=0, description="Price-to-book ratio")
    dividend_yield: Optional[float] = Field(None, ge=0, le=100, description="Dividend yield percentage")
    
    # Performance metrics
    day_change: Optional[float] = Field(None, description="Day's price change")
    day_change_percent: Optional[float] = Field(None, description="Day's percentage change")
    week_change_percent: Optional[float] = Field(None, description="Week's percentage change")
    month_change_percent: Optional[float] = Field(None, description="Month's percentage change")
    ytd_change_percent: Optional[float] = Field(None, description="Year-to-date percentage change")
    
    # Date information
    date: Optional[Union[str, Date]] = Field(
        None,
        description="Date of market context data"
    )
    last_updated: Optional[datetime] = Field(
        default_factory=datetime.utcnow,
        description="Last update timestamp"
    )
    
    # Market indices context
    index_performance: Optional[Dict[str, float]] = Field(
        default_factory=dict,
        description="Performance of relevant market indices"
    )
    sector_performance: Optional[str] = Field(
        None,
        max_length=100,
        description="Sector performance context"
    )
    
    @field_validator('date')
    @classmethod
    def validate_date(cls, v):
        """Validate and convert date format."""
        if isinstance(v, str):
            try:
                # Try to parse different date formats
                for fmt in ['%Y-%m-%d', '%d-%m-%Y', '%m/%d/%Y', '%d/%m/%Y']:
                    try:
                        return datetime.strptime(v, fmt).date()
                    except ValueError:
                        continue
                # If no format matches, return as string
                return v
            except:
                return v
        return v


class SentimentSummary(BaseModel):
    """Model for sentiment summary and outlook."""
    
    positive_drivers: Optional[str] = Field(
        None,
        max_length=1000,
        description="Summary of positive sentiment drivers"
    )
    negative_drivers: Optional[str] = Field(
        None,
        max_length=1000,
        description="Summary of negative sentiment drivers"
    )
    neutral_drivers: Optional[str] = Field(
        None,
        max_length=1000,
        description="Summary of neutral sentiment drivers"
    )
    outlook: Optional[str] = Field(
        None,
        max_length=1000,
        description="Overall outlook and future expectations"
    )
    
    # Key catalysts
    upcoming_catalysts: Optional[List[str]] = Field(
        default_factory=list,
        max_items=10,
        description="Upcoming events that could impact sentiment"
    )
    risk_factors: Optional[List[str]] = Field(
        default_factory=list,
        max_items=10,
        description="Key risk factors to monitor"
    )
    
    # Timeframe analysis
    short_term_outlook: Optional[str] = Field(
        None,
        max_length=500,
        description="Short-term sentiment outlook"
    )
    medium_term_outlook: Optional[str] = Field(
        None,
        max_length=500,
        description="Medium-term sentiment outlook"
    )
    long_term_outlook: Optional[str] = Field(
        None,
        max_length=500,
        description="Long-term sentiment outlook"
    )
    
    @field_validator('positive_drivers', 'negative_drivers', 'neutral_drivers', 'outlook')
    @classmethod
    def cleanup_text(cls, v):
        """Clean up text fields."""
        return v.strip() if v else v


class SentimentAnalysis(BaseModel):
    """Core sentiment analysis model."""
    
    # Overall sentiment
    overall_sentiment: SentimentEnum = Field(
        ...,
        description="Overall sentiment classification"
    )
    sentiment_score: float = Field(
        ...,
        ge=-1,
        le=1,
        description="Overall sentiment score (-1 to 1, where -1 is very negative, 1 is very positive)"
    )
    confidence: float = Field(
        ...,
        ge=0,
        le=1,
        description="Confidence level in the sentiment analysis (0 to 1)"
    )
    
    # Detailed breakdown
    breakdown: Optional[SentimentBreakdown] = Field(
        None,
        description="Detailed breakdown of sentiment factors"
    )
    
    # Market context
    market_context: Optional[MarketContext] = Field(
        None,
        description="Current market context and stock information"
    )
    
    # Summary
    sentiment_summary: Optional[SentimentSummary] = Field(
        None,
        description="Summary and outlook"
    )
    
    # Analysis metadata
    analysis_method: Optional[str] = Field(
        None,
        max_length=100,
        description="Method used for sentiment analysis"
    )
    data_sources: Optional[List[str]] = Field(
        default_factory=list,
        max_items=50,
        description="List of data sources used"
    )
    source_types: Optional[List[SourceTypeEnum]] = Field(
        default_factory=list,
        description="Types of sources analyzed"
    )
    
    # Temporal analysis
    sentiment_trend: Optional[str] = Field(
        None,
        max_length=50,
        description="Trend in sentiment over time (improving, declining, stable)"
    )
    previous_sentiment: Optional[SentimentEnum] = Field(
        None,
        description="Previous sentiment for comparison"
    )
    sentiment_volatility: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="Volatility in sentiment over time"
    )
    
    # Custom analysis fields
    custom_metrics: Optional[Dict[str, Union[str, int, float, bool]]] = Field(
        default_factory=dict,
        description="Additional custom sentiment metrics"
    )


class SentimentData(BaseModel):
    """
    Comprehensive model for stock sentiment analysis data.
    
    This model provides complete validation and structure for storing
    sentiment analysis data for stocks in a production environment.
    """
    
    # Core identification
    stock: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Company name"
    )
    ticker: str = Field(
        ...,
        min_length=1,
        max_length=20,
        description="Stock ticker symbol"
    )
    market: Optional[str] = Field(
        None,
        max_length=100,
        description="Market/exchange information"
    )
    
    # Core sentiment analysis
    sentiment_analysis: SentimentAnalysis = Field(
        ...,
        description="Complete sentiment analysis data"
    )
    
    # Overall summary
    summary: Optional[str] = Field(
        None,
        max_length=2000,
        description="Executive summary of sentiment analysis"
    )
    
    # Analysis metadata
    analysis_date: Optional[datetime] = Field(
        default_factory=datetime.utcnow,
        description="Date and time of sentiment analysis"
    )
    analyst: Optional[str] = Field(
        None,
        max_length=100,
        description="Analyst or system that performed the analysis"
    )
    data_source: Optional[str] = Field(
        None,
        max_length=100,
        description="Primary data source"
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
    
    # Analysis parameters
    analysis_period: Optional[str] = Field(
        None,
        max_length=50,
        description="Time period covered by the analysis"
    )
    sample_size: Optional[int] = Field(
        None,
        ge=0,
        description="Number of data points analyzed"
    )
    languages_analyzed: Optional[List[str]] = Field(
        default_factory=list,
        description="Languages of analyzed content"
    )
    
    # Disclaimer and notes
    disclaimer: Optional[str] = Field(
        None,
        max_length=1000,
        description="Disclaimer about the analysis"
    )
    notes: Optional[List[str]] = Field(
        default_factory=list,
        max_items=20,
        description="Additional notes about the analysis"
    )
    
    # Custom fields for extensibility
    custom_data: Optional[Dict[str, Union[str, int, float, bool]]] = Field(
        default_factory=dict,
        description="Additional custom data fields"
    )
    
    class Config:
        """Pydantic v2 configuration."""
        # Validate on assignment
        validate_assignment = True
        
        # Use enum values in serialization
        use_enum_values = True
        
        # Allow population by field name or alias
        populate_by_name = True
        
        # JSON encoders for special types
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            Date: lambda v: v.isoformat(),
            Decimal: lambda v: float(v)
        }
        
        # Schema extra with example
        json_schema_extra = {
            "example": {
                "stock": "Reliance Industries Ltd",
                "ticker": "RELIANCE",
                "market": "India (NSE/BSE)",
                "sentiment_analysis": {
                    "overall_sentiment": "Neutral",
                    "sentiment_score": 0.2,
                    "confidence": 0.85,
                    "breakdown": {
                        "positive_factors": [
                            {
                                "factor": "Strong market performance",
                                "description": "Stock has risen 24% in 2025",
                                "impact": "High"
                            }
                        ]
                    },
                    "market_context": {
                        "current_price": "₹1,490.00",
                        "market_cap": "₹20,21,480.21 crore"
                    }
                },
                "summary": "Neutral sentiment with balanced positive and negative factors"
            }
        }
    
    @field_validator('ticker')
    @classmethod
    def ticker_uppercase(cls, v):
        """Ensure ticker is uppercase."""
        return v.upper().strip()
    
    @field_validator('stock', 'summary')
    @classmethod
    def cleanup_text(cls, v):
        """Clean up text fields."""
        return v.strip() if v else v
    
    @field_validator('notes')
    @classmethod
    def cleanup_notes(cls, v):
        """Clean up notes list."""
        if v:
            return [note.strip() for note in v if note.strip()]
        return v
    
    @model_validator(mode='before')
    @classmethod
    def validate_sentiment_consistency(cls, values):
        """Validate sentiment score consistency with overall sentiment."""
        if isinstance(values, dict):
            sentiment_analysis = values.get('sentiment_analysis', {})
            if isinstance(sentiment_analysis, dict):
                overall_sentiment = sentiment_analysis.get('overall_sentiment')
                sentiment_score = sentiment_analysis.get('sentiment_score')
                
                if overall_sentiment and sentiment_score is not None:
                    # Validate score consistency with sentiment
                    if overall_sentiment in ['Very Positive', 'Positive', 'Bullish'] and sentiment_score < 0:
                        values['sentiment_analysis']['sentiment_score'] = abs(sentiment_score)
                    elif overall_sentiment in ['Very Negative', 'Negative', 'Bearish'] and sentiment_score > 0:
                        values['sentiment_analysis']['sentiment_score'] = -abs(sentiment_score)
        return values
    
    @model_validator(mode='after')
    def validate_data_consistency(self):
        """Validate overall data consistency."""
        # Ensure analysis_date is not in the future
        if self.analysis_date and self.analysis_date > datetime.utcnow():
            self.analysis_date = datetime.utcnow()
        
        # Validate sample size consistency
        if (self.sample_size is not None and 
            self.sentiment_analysis.breakdown and 
            self.sentiment_analysis.breakdown.positive_factors):
            total_factors = (
                len(self.sentiment_analysis.breakdown.positive_factors or []) +
                len(self.sentiment_analysis.breakdown.negative_factors or []) +
                len(self.sentiment_analysis.breakdown.neutral_factors or [])
            )
            if self.sample_size < total_factors:
                self.sample_size = total_factors
        
        return self


# MongoDB indexes for optimal query performance
SENTIMENT_INDEXES = [
    IndexModel([("ticker", 1)], unique=True, name="ticker_unique"),
    IndexModel([("analysis_date", -1)], name="analysis_date_desc"),
    IndexModel([("last_updated", -1)], name="last_updated_desc"),
    IndexModel([("sentiment_analysis.overall_sentiment", 1)], name="overall_sentiment"),
    IndexModel([("sentiment_analysis.sentiment_score", -1)], name="sentiment_score_desc"),
    IndexModel([("sentiment_analysis.confidence", -1)], name="confidence_desc"),
    IndexModel([("market", 1)], name="market"),
    IndexModel([("data_source", 1)], name="data_source"),
    IndexModel([
        ("ticker", 1),
        ("analysis_date", -1)
    ], name="ticker_analysis_compound"),
    IndexModel([
        ("sentiment_analysis.overall_sentiment", 1),
        ("sentiment_analysis.confidence", -1)
    ], name="sentiment_confidence_compound"),
    IndexModel([
        ("market", 1),
        ("sentiment_analysis.sentiment_score", -1)
    ], name="market_score_compound")
]


# Backward compatibility alias
sentiment = SentimentData