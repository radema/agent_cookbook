
from enum import Enum
from typing import Optional, Dict, Any, List, Union
from pydantic import BaseModel, Field


class ShortAnswer(str, Enum):
    YES = "YES"
    NO = "NO"
    NA = "N/A"  # Not Applicable
    ND = "N/D"  # No Data/Not Determined


class InvoiceResponse(BaseModel):
    """Structured response for invoice analysis questions."""
    
    short_answer: ShortAnswer = Field(
        description="Quick categorical answer (YES/NO/N/A/N/D)"
    )
    
    extracted_information: Dict[str, Any] = Field(
        default_factory=dict,
        description="Key data points extracted from the invoice (prices, dates, items, etc.)"
    )
    
    full_answer: str = Field(
        description="Complete textual answer to the question"
    )
    
    reasoning: str = Field(
        description="Explanation of how the answer was derived from available information"
    )
    
    class Config:
        use_enum_values = True
        schema_extra = {
            "example": {
                "short_answer": "YES",
                "extracted_information": {
                    "total_cost": 250.00,
                    "vat_amount": 50.00,
                    "total_with_vat": 300.00,
                    "currency": "EUR"
                },
                "full_answer": "The total cost including VAT for the purchased items is €300.00.",
                "reasoning": "The invoice shows a subtotal of €250.00 with VAT at 20% (€50.00), resulting in a total including VAT of €300.00."
            }
        }

# Example usage
def create_invoice_response(
    query: str,
    nodes: List[Any],
    llm_response: str
) -> InvoiceResponse:
    """
    Process RAG results to create a structured invoice response.
    
    Args:
        query: The original query
        nodes: Retrieved document nodes
        llm_response: Raw text response from the LLM
        
    Returns:
        Structured InvoiceResponse object
    """
    # This would be implemented with your actual logic
    # You might use the LLM to extract structured data or implement custom parsing
    
    # Placeholder implementation
    return InvoiceResponse(
        short_answer=ShortAnswer.YES,
        extracted_information={
            "total_cost": 250.00,
            "vat_amount": 50.00, 
            "total_with_vat": 300.00
        },
        full_answer=llm_response,
        reasoning="Based on the retrieved invoice data, we found the line items and calculated the total with VAT."
    )