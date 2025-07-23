from pydantic import BaseModel, Field

class fundamental(BaseModel):
    """
    Model for storing fundamental data.
    """
    symbol: str = Field(..., description="Stock symbol")
    name: str = Field(..., description="Company name")
    sector: str = Field(..., description="Sector of the company")
    industry: str = Field(..., description="Industry of the company")
    market_cap: float = Field(..., description="Market capitalization in billions")
    pe_ratio: float = Field(..., description="Price-to-earnings ratio")
    dividend_yield: float = Field(..., description="Dividend yield percentage")
    earnings_per_share: float = Field(..., description="Earnings per share in dollars")
    revenue: float = Field(..., description="Annual revenue in billions")
    net_income: float = Field(..., description="Net income in billions")
    debt_to_equity: float = Field(..., description="Debt-to-equity ratio")  