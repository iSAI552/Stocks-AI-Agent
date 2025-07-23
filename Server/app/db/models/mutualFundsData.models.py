"""
Production-ready Pydantic v2 models for mutual fund holdings data.
Provides comprehensive validation, serialization, and documentation for mutual fund analysis.
"""

from datetime import datetime, date as Date
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator
from pymongo import IndexModel


class HoldingChangeEnum(str, Enum):
    """Enumeration for holding change types."""
    INCREASED = "Increased"
    DECREASED = "Decreased"
    STABLE = "Stable"
    NEW_ENTRY = "New Entry"
    EXITED = "Exited"
    MAINTAINED = "Maintained"
    NOT_SPECIFIED = "Not specified"


class AnalystRatingEnum(str, Enum):
    """Enumeration for analyst ratings."""
    BUY = "Buy"
    STRONG_BUY = "Strong Buy"
    HOLD = "Hold"
    SELL = "Sell"
    STRONG_SELL = "Strong Sell"
    UNDERPERFORM = "Underperform"
    OUTPERFORM = "Outperform"
    NEUTRAL = "Neutral"


class FundCategoryEnum(str, Enum):
    """Enumeration for mutual fund categories."""
    LARGE_CAP = "Large Cap"
    MID_CAP = "Mid Cap"
    SMALL_CAP = "Small Cap"
    FLEXI_CAP = "Flexi Cap"
    MULTI_CAP = "Multi Cap"
    EQUITY_HYBRID = "Equity Hybrid"
    BLUECHIP = "Bluechip"
    SECTORAL = "Sectoral"
    THEMATIC = "Thematic"
    INDEX = "Index"
    ELSS = "ELSS"
    BALANCED = "Balanced"
    OTHER = "Other"


class WeightageChange(BaseModel):
    """Model for tracking weightage changes in mutual fund holdings."""
    
    change: HoldingChangeEnum = Field(
        ...,
        description="Type of change in holding"
    )
    percentage_change: Optional[str] = Field(
        None,
        max_length=20,
        description="Percentage change in holding (e.g., '+0.3%', '-1.2%')"
    )
    percentage_change_numeric: Optional[float] = Field(
        None,
        ge=-100,
        le=100,
        description="Numeric percentage change"
    )
    last_revision_date: Optional[Union[str, Date]] = Field(
        None,
        description="Date of last revision (YYYY-MM-DD)"
    )
    comment: Optional[str] = Field(
        None,
        max_length=500,
        description="Commentary on the change"
    )
    
    # Additional metadata
    revision_quarter: Optional[str] = Field(
        None,
        max_length=20,
        description="Quarter of revision (e.g., 'Q1 FY25')"
    )
    reason: Optional[str] = Field(
        None,
        max_length=300,
        description="Reason for the change"
    )
    confidence_level: Optional[str] = Field(
        None,
        max_length=20,
        description="Confidence in the change (High, Medium, Low)"
    )
    
    @field_validator('last_revision_date')
    @classmethod
    def validate_date(cls, v):
        """Validate and convert date format."""
        if isinstance(v, str) and v:
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
    
    @field_validator('comment')
    @classmethod
    def cleanup_comment(cls, v):
        """Clean up comment text."""
        return v.strip() if v else v


class MutualFundHolding(BaseModel):
    """Model for individual mutual fund holding data."""
    
    # Basic fund information
    fund_name: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Name of the mutual fund"
    )
    fund_house: Optional[str] = Field(
        None,
        max_length=100,
        description="Fund house/AMC name"
    )
    fund_category: Optional[FundCategoryEnum] = Field(
        None,
        description="Category of the mutual fund"
    )
    
    # Holding details
    holding_percentage: Optional[str] = Field(
        None,
        max_length=50,
        description="Holding percentage as string (e.g., '5.8%')"
    )
    holding_percentage_numeric: Optional[float] = Field(
        None,
        ge=0,
        le=100,
        description="Holding percentage as numeric value"
    )
    holding_value: Optional[str] = Field(
        None,
        max_length=50,
        description="Holding value in currency"
    )
    holding_value_numeric: Optional[float] = Field(
        None,
        ge=0,
        description="Holding value as numeric"
    )
    
    # Change tracking
    weightage_change: Optional[WeightageChange] = Field(
        None,
        description="Details of weightage changes"
    )
    
    # Price targets and analysis
    target_price: Optional[str] = Field(
        None,
        max_length=50,
        description="Fund's target price for the stock"
    )
    target_price_numeric: Optional[float] = Field(
        None,
        ge=0,
        description="Target price as numeric value"
    )
    rating: Optional[AnalystRatingEnum] = Field(
        None,
        description="Fund's rating for the stock"
    )
    
    # Fund performance metrics
    fund_aum: Optional[str] = Field(
        None,
        max_length=50,
        description="Fund's Assets Under Management"
    )
    fund_aum_numeric: Optional[float] = Field(
        None,
        ge=0,
        description="Fund AUM as numeric value"
    )
    fund_performance_1y: Optional[float] = Field(
        None,
        description="Fund's 1-year performance percentage"
    )
    fund_performance_3y: Optional[float] = Field(
        None,
        description="Fund's 3-year performance percentage"
    )
    fund_performance_5y: Optional[float] = Field(
        None,
        description="Fund's 5-year performance percentage"
    )
    
    # Source and metadata
    source: Optional[str] = Field(
        None,
        max_length=300,
        description="Source of the information"
    )
    data_date: Optional[Union[str, Date]] = Field(
        None,
        description="Date of the data"
    )
    reliability_score: Optional[float] = Field(
        None,
        ge=0,
        le=100,
        description="Reliability score of the data (0-100)"
    )
    
    # Additional insights
    fund_manager: Optional[str] = Field(
        None,
        max_length=100,
        description="Fund manager name"
    )
    investment_thesis: Optional[str] = Field(
        None,
        max_length=500,
        description="Fund's investment thesis for the stock"
    )
    risk_rating: Optional[str] = Field(
        None,
        max_length=20,
        description="Risk rating (High, Medium, Low)"
    )
    
    @field_validator('fund_name')
    @classmethod
    def cleanup_fund_name(cls, v):
        """Clean up fund name."""
        return v.strip()
    
    @field_validator('data_date')
    @classmethod
    def validate_date(cls, v):
        """Validate and convert date format."""
        if isinstance(v, str) and v:
            try:
                for fmt in ['%Y-%m-%d', '%d-%m-%Y', '%m/%d/%Y', '%d/%m/%Y']:
                    try:
                        return datetime.strptime(v, fmt).date()
                    except ValueError:
                        continue
                return v
            except:
                return v
        return v


class AnalystRatings(BaseModel):
    """Model for analyst ratings breakdown."""
    
    buy: Optional[int] = Field(None, ge=0, description="Number of buy ratings")
    strong_buy: Optional[int] = Field(None, ge=0, description="Number of strong buy ratings")
    hold: Optional[int] = Field(None, ge=0, description="Number of hold ratings")
    sell: Optional[int] = Field(None, ge=0, description="Number of sell ratings")
    strong_sell: Optional[int] = Field(None, ge=0, description="Number of strong sell ratings")
    
    # Calculated fields
    total_analysts: Optional[int] = Field(None, ge=0, description="Total number of analysts")
    consensus_rating: Optional[AnalystRatingEnum] = Field(
        None,
        description="Consensus rating"
    )
    bullish_percentage: Optional[float] = Field(
        None,
        ge=0,
        le=100,
        description="Percentage of bullish ratings"
    )
    bearish_percentage: Optional[float] = Field(
        None,
        ge=0,
        le=100,
        description="Percentage of bearish ratings"
    )


class GeneralMarketData(BaseModel):
    """Model for general market data and stock information."""
    
    # Current price information
    current_share_price: Optional[str] = Field(
        None,
        max_length=50,
        description="Current share price with currency"
    )
    current_share_price_numeric: Optional[float] = Field(
        None,
        ge=0,
        description="Current share price as numeric value"
    )
    
    # Price ranges
    week_52_high: Optional[str] = Field(None, max_length=50, description="52-week high")
    week_52_low: Optional[str] = Field(None, max_length=50, description="52-week low")
    week_52_high_numeric: Optional[float] = Field(None, ge=0, description="52-week high (numeric)")
    week_52_low_numeric: Optional[float] = Field(None, ge=0, description="52-week low (numeric)")
    
    # Holdings data
    mutual_fund_holding_percentage: Optional[str] = Field(
        None,
        max_length=100,
        description="Overall mutual fund holding percentage"
    )
    mutual_fund_holding_numeric: Optional[float] = Field(
        None,
        ge=0,
        le=100,
        description="MF holding percentage (numeric)"
    )
    total_mutual_funds_holding: Optional[int] = Field(
        None,
        ge=0,
        description="Total number of mutual funds holding the stock"
    )
    
    # Price targets
    average_target_price: Optional[str] = Field(
        None,
        max_length=100,
        description="Average target price from analysts"
    )
    average_target_price_numeric: Optional[float] = Field(
        None,
        ge=0,
        description="Average target price (numeric)"
    )
    highest_target_price: Optional[float] = Field(None, ge=0, description="Highest target price")
    lowest_target_price: Optional[float] = Field(None, ge=0, description="Lowest target price")
    
    # Analyst ratings
    analyst_ratings: Optional[AnalystRatings] = Field(
        None,
        description="Breakdown of analyst ratings"
    )
    
    # Valuation metrics
    pe_ratio: Optional[str] = Field(None, max_length=50, description="P/E ratio")
    pe_ratio_numeric: Optional[float] = Field(None, ge=0, description="P/E ratio (numeric)")
    pb_ratio: Optional[float] = Field(None, ge=0, description="Price-to-book ratio")
    market_cap: Optional[str] = Field(None, max_length=100, description="Market capitalization")
    market_cap_numeric: Optional[float] = Field(None, ge=0, description="Market cap (numeric)")
    
    # Dividend information
    dividend_yield: Optional[str] = Field(None, max_length=50, description="Dividend yield")
    dividend_yield_numeric: Optional[float] = Field(None, ge=0, le=100, description="Dividend yield (numeric)")
    dividend_per_share: Optional[float] = Field(None, ge=0, description="Dividend per share")
    
    # Trading metrics
    volume: Optional[int] = Field(None, ge=0, description="Trading volume")
    average_volume: Optional[int] = Field(None, ge=0, description="Average trading volume")
    
    # Performance metrics
    day_change: Optional[float] = Field(None, description="Day's change")
    day_change_percent: Optional[float] = Field(None, description="Day's percentage change")
    month_change_percent: Optional[float] = Field(None, description="Month's percentage change")
    ytd_change_percent: Optional[float] = Field(None, description="Year-to-date change")
    
    # Beta and volatility
    beta: Optional[float] = Field(None, ge=0, description="Beta coefficient")
    volatility: Optional[float] = Field(None, ge=0, description="Volatility measure")
    
    # Additional metrics
    book_value: Optional[float] = Field(None, ge=0, description="Book value per share")
    roce: Optional[float] = Field(None, description="Return on Capital Employed")
    roe: Optional[float] = Field(None, description="Return on Equity")
    debt_to_equity: Optional[float] = Field(None, ge=0, description="Debt to Equity ratio")


class MutualFundData(BaseModel):
    """
    Comprehensive model for mutual fund holdings and analysis data.
    
    This model provides complete validation and structure for storing
    mutual fund related data for stocks in a production environment.
    """
    
    # Core identification
    stock: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Stock name and ticker"
    )
    ticker: Optional[str] = Field(
        None,
        max_length=20,
        description="Stock ticker symbol (extracted or separate)"
    )
    current_date: Optional[Union[str, Date]] = Field(
        None,
        description="Date of the data"
    )
    
    # Mutual fund holdings
    mutual_fund_holdings: Optional[List[MutualFundHolding]] = Field(
        default_factory=list,
        max_items=100,
        description="List of mutual fund holdings"
    )
    
    # General market data
    general_market_data: Optional[GeneralMarketData] = Field(
        None,
        description="General market and stock data"
    )
    
    # Summary and analysis
    summary: Optional[str] = Field(
        None,
        max_length=3000,
        description="Summary of mutual fund holdings and analysis"
    )
    
    # Analysis insights
    key_insights: Optional[List[str]] = Field(
        default_factory=list,
        max_items=20,
        description="Key insights from the analysis"
    )
    investment_outlook: Optional[str] = Field(
        None,
        max_length=1000,
        description="Investment outlook based on MF holdings"
    )
    
    # Trends and patterns
    holding_trends: Optional[Dict[str, Union[str, float]]] = Field(
        default_factory=dict,
        description="Trends in mutual fund holdings"
    )
    top_fund_houses: Optional[List[str]] = Field(
        default_factory=list,
        max_items=10,
        description="Top fund houses holding the stock"
    )
    
    # Risk assessment
    risk_factors: Optional[List[str]] = Field(
        default_factory=list,
        max_items=15,
        description="Risk factors identified from MF analysis"
    )
    opportunity_factors: Optional[List[str]] = Field(
        default_factory=list,
        max_items=15,
        description="Opportunity factors from MF perspective"
    )
    
    # Metadata
    data_source: Optional[str] = Field(
        None,
        max_length=200,
        description="Primary data source"
    )
    data_sources: Optional[List[str]] = Field(
        default_factory=list,
        max_items=20,
        description="Multiple data sources used"
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
    
    # Analysis metadata
    analysis_date: Optional[datetime] = Field(
        default_factory=datetime.utcnow,
        description="Date of analysis"
    )
    analyst: Optional[str] = Field(
        None,
        max_length=100,
        description="Analyst or system performing analysis"
    )
    
    # Market context
    market_segment: Optional[str] = Field(
        None,
        max_length=50,
        description="Market segment (Large Cap, Mid Cap, etc.)"
    )
    sector: Optional[str] = Field(
        None,
        max_length=100,
        description="Sector classification"
    )
    industry: Optional[str] = Field(
        None,
        max_length=100,
        description="Industry classification"
    )
    
    # Quality scores
    data_quality_score: Optional[float] = Field(
        None,
        ge=0,
        le=100,
        description="Quality score of the data (0-100)"
    )
    analysis_confidence: Optional[float] = Field(
        None,
        ge=0,
        le=100,
        description="Confidence in the analysis (0-100)"
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
                "stock": "Reliance Industries Ltd. (RELIANCE)",
                "ticker": "RELIANCE",
                "current_date": "2025-07-16",
                "mutual_fund_holdings": [
                    {
                        "fund_name": "Parag Parikh Flexi Cap Fund",
                        "holding_percentage": "5.8%",
                        "weightage_change": {
                            "change": "Increased",
                            "percentage_change": "+0.3%",
                            "last_revision_date": "2025-03-31"
                        },
                        "target_price": "Not explicitly stated"
                    }
                ],
                "general_market_data": {
                    "current_share_price": "₹1496.65",
                    "pe_ratio": "24.78 (TTM)",
                    "market_cap": "₹1770034.38 crore"
                },
                "summary": "Comprehensive mutual fund holdings analysis"
            }
        }
    
    @field_validator('stock', 'summary')
    @classmethod
    def cleanup_text(cls, v):
        """Clean up text fields."""
        return v.strip() if v else v
    
    @field_validator('ticker')
    @classmethod
    def ticker_uppercase(cls, v):
        """Ensure ticker is uppercase."""
        return v.upper().strip() if v else v
    
    @field_validator('current_date')
    @classmethod
    def validate_date(cls, v):
        """Validate and convert date format."""
        if isinstance(v, str) and v:
            try:
                for fmt in ['%Y-%m-%d', '%d-%m-%Y', '%m/%d/%Y', '%d/%m/%Y']:
                    try:
                        return datetime.strptime(v, fmt).date()
                    except ValueError:
                        continue
                return v
            except:
                return v
        return v
    
    @field_validator('key_insights', 'risk_factors', 'opportunity_factors', 'top_fund_houses', 'data_sources')
    @classmethod
    def cleanup_lists(cls, v):
        """Clean up list items."""
        if v:
            return [item.strip() for item in v if item.strip()]
        return v
    
    @model_validator(mode='before')
    @classmethod
    def extract_ticker_from_stock(cls, values):
        """Extract ticker from stock name if not provided separately."""
        if isinstance(values, dict):
            stock = values.get('stock', '')
            ticker = values.get('ticker')
            
            if not ticker and stock:
                # Try to extract ticker from stock name
                import re
                ticker_match = re.search(r'\(([A-Z]+)\)', stock)
                if ticker_match:
                    values['ticker'] = ticker_match.group(1)
        return values
    
    @model_validator(mode='after')
    def validate_data_consistency(self):
        """Validate data consistency across fields."""
        # Validate mutual fund holdings count
        if self.mutual_fund_holdings:
            total_holdings = len(self.mutual_fund_holdings)
            if (self.general_market_data and 
                self.general_market_data.total_mutual_funds_holding and
                self.general_market_data.total_mutual_funds_holding < total_holdings):
                self.general_market_data.total_mutual_funds_holding = total_holdings
        
        # Ensure analysis_date is not in the future
        if self.analysis_date and self.analysis_date > datetime.utcnow():
            self.analysis_date = datetime.utcnow()
        
        return self


# MongoDB indexes for optimal query performance
MUTUAL_FUND_INDEXES = [
    IndexModel([("ticker", 1)], unique=True, name="ticker_unique"),
    IndexModel([("stock", 1)], name="stock_name"),
    IndexModel([("current_date", -1)], name="current_date_desc"),
    IndexModel([("last_updated", -1)], name="last_updated_desc"),
    IndexModel([("analysis_date", -1)], name="analysis_date_desc"),
    IndexModel([("sector", 1)], name="sector"),
    IndexModel([("market_segment", 1)], name="market_segment"),
    IndexModel([("mutual_fund_holdings.fund_name", 1)], name="fund_name"),
    IndexModel([("mutual_fund_holdings.fund_house", 1)], name="fund_house"),
    IndexModel([("general_market_data.mutual_fund_holding_numeric", -1)], name="mf_holding_desc"),
    IndexModel([
        ("ticker", 1),
        ("current_date", -1)
    ], name="ticker_date_compound"),
    IndexModel([
        ("sector", 1),
        ("general_market_data.market_cap_numeric", -1)
    ], name="sector_marketcap_compound"),
    IndexModel([
        ("mutual_fund_holdings.fund_house", 1),
        ("mutual_fund_holdings.holding_percentage_numeric", -1)
    ], name="fundhouse_holding_compound")
]


# Backward compatibility alias
mutualFund = MutualFundData