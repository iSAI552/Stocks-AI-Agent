"""
Production-ready Pydantic v2 models for stock technical analysis data.
Provides comprehensive validation, serialization, and documentation for technical indicators.
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator
from pymongo import IndexModel


class TrendEnum(str, Enum):
    """Enumeration for trend directions."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    SIDEWAYS = "sideways"
    SIDEWAYS_BULLISH = "sideways with slight bullish bias"
    SIDEWAYS_BEARISH = "sideways with slight bearish bias"
    NEUTRAL_BULLISH = "neutral to mildly bullish"
    NEUTRAL_BEARISH = "neutral to mildly bearish"
    STRONGLY_BULLISH = "strongly bullish"
    STRONGLY_BEARISH = "strongly bearish"


class SignalEnum(str, Enum):
    """Enumeration for trading signals."""
    BUY = "Buy"
    SELL = "Sell"
    HOLD = "Hold"
    STRONG_BUY = "Strong Buy"
    STRONG_SELL = "Strong Sell"
    NEUTRAL = "Neutral"


class TimeframeEnum(str, Enum):
    """Enumeration for analysis timeframes."""
    INTRADAY = "intraday"
    ONE_WEEK = "1_week"
    ONE_MONTH = "1_month"
    THREE_MONTHS = "3_months"
    SIX_MONTHS = "6_months"
    ONE_YEAR = "1_year"
    THREE_YEARS = "3_years"
    FIVE_YEARS = "5_years"


class TechnicalIndicators(BaseModel):
    """Model for technical indicators with validation."""
    
    # Momentum Indicators
    RSI: Optional[float] = Field(
        None,
        ge=0,
        le=100,
        description="Relative Strength Index (0-100)"
    )
    Stochastic_RSI: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="Stochastic RSI (0-1)"
    )
    CCI: Optional[float] = Field(
        None,
        ge=-300,
        le=300,
        description="Commodity Channel Index"
    )
    MFI: Optional[float] = Field(
        None,
        ge=0,
        le=100,
        description="Money Flow Index (0-100)"
    )
    
    # Trend Indicators
    ADX: Optional[float] = Field(
        None,
        ge=0,
        le=100,
        description="Average Directional Index"
    )
    MACD: Optional[float] = Field(
        None,
        description="MACD value"
    )
    MACD_signal: Optional[SignalEnum] = Field(
        None,
        description="MACD signal interpretation"
    )
    
    # Volume Indicators
    OBV: Optional[float] = Field(
        None,
        description="On Balance Volume"
    )
    
    # Volatility Indicators
    Bollinger_Bands: Optional[str] = Field(
        None,
        max_length=100,
        description="Bollinger Bands status"
    )
    ATR: Optional[float] = Field(
        None,
        ge=0,
        description="Average True Range"
    )
    
    # Custom indicators
    custom_indicators: Optional[Dict[str, Union[float, str]]] = Field(
        default_factory=dict,
        description="Additional custom technical indicators"
    )


class MovingAverages(BaseModel):
    """Model for moving averages analysis."""
    above: Optional[List[str]] = Field(
        default_factory=list,
        max_items=20,
        description="Moving averages that price is trading above"
    )
    below: Optional[List[str]] = Field(
        default_factory=list,
        max_items=20,
        description="Moving averages that price is trading below"
    )
    
    # Specific MA values
    MA_5: Optional[float] = Field(None, ge=0, description="5-period Moving Average")
    MA_10: Optional[float] = Field(None, ge=0, description="10-period Moving Average")
    MA_20: Optional[float] = Field(None, ge=0, description="20-period Moving Average")
    MA_50: Optional[float] = Field(None, ge=0, description="50-period Moving Average")
    MA_100: Optional[float] = Field(None, ge=0, description="100-period Moving Average")
    MA_200: Optional[float] = Field(None, ge=0, description="200-period Moving Average")
    
    # EMA values
    EMA_9: Optional[float] = Field(None, ge=0, description="9-period Exponential Moving Average")
    EMA_12: Optional[float] = Field(None, ge=0, description="12-period Exponential Moving Average")
    EMA_21: Optional[float] = Field(None, ge=0, description="21-period Exponential Moving Average")
    EMA_26: Optional[float] = Field(None, ge=0, description="26-period Exponential Moving Average")


class VolumeAnalysis(BaseModel):
    """Model for volume analysis."""
    trend: Optional[str] = Field(
        None,
        max_length=200,
        description="Volume trend description"
    )
    delivery_percentage: Optional[float] = Field(
        None,
        ge=0,
        le=100,
        description="Delivery percentage in volume"
    )
    average_volume: Optional[float] = Field(
        None,
        ge=0,
        description="Average volume"
    )
    volume_surge: Optional[bool] = Field(
        None,
        description="Whether there's a volume surge"
    )
    accumulation_distribution: Optional[str] = Field(
        None,
        max_length=100,
        description="Accumulation/Distribution pattern"
    )


class PriceLevels(BaseModel):
    """Model for support and resistance levels."""
    support: Optional[List[str]] = Field(
        default_factory=list,
        max_items=10,
        description="Support price levels"
    )
    resistance: Optional[List[str]] = Field(
        default_factory=list,
        max_items=10,
        description="Resistance price levels"
    )
    
    # Key levels as numeric values for calculations
    support_numeric: Optional[List[float]] = Field(
        default_factory=list,
        description="Support levels as numeric values"
    )
    resistance_numeric: Optional[List[float]] = Field(
        default_factory=list,
        description="Resistance levels as numeric values"
    )


class TimeframeAnalysis(BaseModel):
    """Model for timeframe-specific technical analysis."""
    
    # Basic trend and price info
    trend: Optional[Union[TrendEnum, str]] = Field(
        None,
        description="Overall trend for this timeframe"
    )
    price_range: Optional[str] = Field(
        None,
        max_length=50,
        description="Current trading price range"
    )
    current_price: Optional[float] = Field(
        None,
        ge=0,
        description="Current price (numeric)"
    )
    
    # Support and resistance
    support: Optional[List[str]] = Field(
        default_factory=list,
        description="Support levels"
    )
    resistance: Optional[List[str]] = Field(
        default_factory=list,
        description="Resistance levels"
    )
    
    # Technical indicators
    indicators: Optional[TechnicalIndicators] = Field(
        None,
        description="Technical indicators for this timeframe"
    )
    
    # Moving averages
    moving_averages: Optional[MovingAverages] = Field(
        None,
        description="Moving averages analysis"
    )
    long_term_ma: Optional[MovingAverages] = Field(
        None,
        description="Long-term moving averages"
    )
    
    # Volume analysis
    volume: Optional[Union[str, VolumeAnalysis]] = Field(
        None,
        description="Volume analysis (can be string or detailed object)"
    )
    
    # Performance metrics
    returns_percent: Optional[float] = Field(
        None,
        ge=-100,
        le=1000,
        description="Returns percentage for this timeframe"
    )
    volatility: Optional[str] = Field(
        None,
        max_length=100,
        description="Volatility description"
    )
    beta: Optional[float] = Field(
        None,
        ge=0,
        le=5,
        description="Beta coefficient"
    )
    
    # Pattern analysis
    pattern: Optional[str] = Field(
        None,
        max_length=300,
        description="Chart pattern description"
    )
    
    # Oscillators for longer timeframes
    oscillators: Optional[Dict[str, Union[str, float]]] = Field(
        default_factory=dict,
        description="Oscillator readings"
    )
    
    # Long-term specific fields
    long_term_indicators: Optional[Dict[str, str]] = Field(
        default_factory=dict,
        description="Long-term technical indicators"
    )
    support_trendline: Optional[str] = Field(
        None,
        max_length=200,
        description="Support trendline description"
    )
    critical_support: Optional[str] = Field(
        None,
        max_length=100,
        description="Critical support level"
    )
    
    # Price vs MA analysis
    current_vs_ma: Optional[str] = Field(
        None,
        max_length=200,
        description="Current price vs moving averages analysis"
    )
    
    # Comments and notes
    comment: Optional[str] = Field(
        None,
        max_length=500,
        description="Additional comments for this timeframe"
    )
    
    # Custom fields for flexibility
    custom_data: Optional[Dict[str, Union[str, float, bool]]] = Field(
        default_factory=dict,
        description="Additional custom data for this timeframe"
    )


class KeyLevels(BaseModel):
    """Model for key buy/sell levels."""
    buy: Optional[List[str]] = Field(
        default_factory=list,
        max_items=10,
        description="Key buy levels"
    )
    sell: Optional[List[str]] = Field(
        default_factory=list,
        max_items=10,
        description="Key sell levels"
    )
    
    # Numeric versions for calculations
    buy_numeric: Optional[List[float]] = Field(
        default_factory=list,
        description="Buy levels as numeric values"
    )
    sell_numeric: Optional[List[float]] = Field(
        default_factory=list,
        description="Sell levels as numeric values"
    )


class PriceTargets(BaseModel):
    """Model for price targets."""
    short_term: Optional[str] = Field(
        None,
        max_length=50,
        description="Short-term price target"
    )
    mid_term: Optional[str] = Field(
        None,
        max_length=50,
        description="Mid-term price target"
    )
    long_term: Optional[str] = Field(
        None,
        max_length=50,
        description="Long-term price target"
    )
    
    # Channel specific targets
    weekly_channel_upside: Optional[str] = Field(
        None,
        max_length=50,
        description="Weekly channel upside target"
    )
    breakout_target: Optional[str] = Field(
        None,
        max_length=50,
        description="Breakout target"
    )
    
    # Numeric targets for calculations
    short_term_numeric: Optional[float] = Field(None, ge=0, description="Short-term target (numeric)")
    mid_term_numeric: Optional[float] = Field(None, ge=0, description="Mid-term target (numeric)")
    long_term_numeric: Optional[float] = Field(None, ge=0, description="Long-term target (numeric)")
    
    # Stop loss levels
    stop_loss: Optional[str] = Field(
        None,
        max_length=50,
        description="Stop loss level"
    )
    stop_loss_numeric: Optional[float] = Field(None, ge=0, description="Stop loss (numeric)")


class TechnicalData(BaseModel):
    """
    Comprehensive model for stock technical analysis data.
    
    This model provides complete validation and structure for storing
    technical analysis data for stocks in a production environment.
    """
    
    # Core identification
    ticker: str = Field(
        ...,
        min_length=1,
        max_length=20,
        description="Stock ticker symbol"
    )
    
    # Timeframe analysis
    timeframes: Optional[Dict[str, TimeframeAnalysis]] = Field(
        default_factory=dict,
        description="Analysis for different timeframes"
    )
    
    # Key levels and targets
    key_levels: Optional[KeyLevels] = Field(
        None,
        description="Key buy and sell levels"
    )
    targets: Optional[PriceTargets] = Field(
        None,
        description="Price targets and stop losses"
    )
    
    # Risk and caveats
    caveats: Optional[List[str]] = Field(
        default_factory=list,
        max_items=20,
        description="Important caveats and risk factors"
    )
    risks: Optional[List[str]] = Field(
        default_factory=list,
        max_items=20,
        description="Technical risk factors"
    )
    
    # Overall analysis
    summary: Optional[str] = Field(
        None,
        max_length=2000,
        description="Overall technical analysis summary"
    )
    overall_signal: Optional[SignalEnum] = Field(
        None,
        description="Overall trading signal"
    )
    confidence_level: Optional[float] = Field(
        None,
        ge=0,
        le=100,
        description="Confidence level in the analysis (0-100)"
    )
    
    # Metadata
    analysis_date: Optional[datetime] = Field(
        default_factory=datetime.utcnow,
        description="Date of technical analysis"
    )
    analyst: Optional[str] = Field(
        None,
        max_length=100,
        description="Name of the analyst or system"
    )
    data_source: Optional[str] = Field(
        None,
        max_length=100,
        description="Source of the technical data"
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
    
    # Market context
    market_condition: Optional[str] = Field(
        None,
        max_length=100,
        description="Overall market condition"
    )
    sector_performance: Optional[str] = Field(
        None,
        max_length=100,
        description="Sector performance context"
    )
    
    # Custom fields for extensibility
    custom_analysis: Optional[Dict[str, Union[str, int, float, bool]]] = Field(
        default_factory=dict,
        description="Additional custom analysis fields"
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
            Decimal: lambda v: float(v)
        }
        
        # Schema extra with example
        json_schema_extra = {
            "example": {
                "ticker": "RELIANCE",
                "timeframes": {
                    "1_week": {
                        "trend": "neutral to mildly bearish",
                        "price_range": "₹1,510–₹1,520",
                        "support": ["₹1,475–₹1,480"],
                        "resistance": ["₹1,508", "₹1,517"],
                        "indicators": {
                            "RSI": 50.4,
                            "ADX": 36.3,
                            "MACD": 20.7,
                            "MFI": 60.0
                        }
                    }
                },
                "key_levels": {
                    "buy": ["₹1,480", "₹1,450–₹1,460"],
                    "sell": ["₹1,559–₹1,560", "₹1,574–₹1,596"]
                },
                "targets": {
                    "short_term": "₹1,560",
                    "mid_term": "₹1,630"
                },
                "summary": "Bullish setup with support at ₹1,480 and targets at ₹1,560"
            }
        }
    
    @field_validator('ticker')
    @classmethod
    def ticker_uppercase(cls, v):
        """Ensure ticker is uppercase."""
        return v.upper().strip()
    
    @field_validator('summary')
    @classmethod
    def summary_cleanup(cls, v):
        """Clean up summary text."""
        if v:
            return v.strip()
        return v
    
    @field_validator('caveats', 'risks')
    @classmethod
    def cleanup_lists(cls, v):
        """Clean up list items."""
        if v:
            return [item.strip() for item in v if item.strip()]
        return v
    
    @model_validator(mode='before')
    @classmethod
    def validate_timeframes(cls, values):
        """Validate timeframe data structure."""
        if isinstance(values, dict):
            timeframes = values.get('timeframes', {})
            if timeframes:
                # Ensure timeframe keys are valid
                valid_timeframes = {
                    '1_week', '1_month', '3_months', '6_months',
                    '1_year', '3_years', '5_years', 'intraday'
                }
                for tf_key in timeframes.keys():
                    if tf_key not in valid_timeframes:
                        # Allow custom timeframes but log a warning
                        pass
        return values
    
    @model_validator(mode='after')
    def validate_consistency(self):
        """Validate data consistency across timeframes."""
        # Add custom validation logic here if needed
        # For example, ensure short-term targets are realistic
        if self.targets and self.key_levels:
            # Custom validation logic can be added here
            pass
        return self


# MongoDB indexes for optimal query performance
TECHNICAL_INDEXES = [
    IndexModel([("ticker", 1)], unique=True, name="ticker_unique"),
    IndexModel([("analysis_date", -1)], name="analysis_date_desc"),
    IndexModel([("last_updated", -1)], name="last_updated_desc"),
    IndexModel([("overall_signal", 1)], name="overall_signal"),
    IndexModel([("data_source", 1)], name="data_source"),
    IndexModel([("market_condition", 1)], name="market_condition"),
    IndexModel([
        ("ticker", 1),
        ("analysis_date", -1)
    ], name="ticker_analysis_compound"),
    IndexModel([
        ("overall_signal", 1),
        ("confidence_level", -1)
    ], name="signal_confidence_compound")
]


# Backward compatibility alias
technical = TechnicalData