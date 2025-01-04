from typing import List
from pydantic import BaseModel, Field


class Metadata(BaseModel):
    """
    A data model representing key professional and educational metadata extracted from a resume.
    This class captures essential candidate information including technical/professional skills
    and the geographical distribution of their educational background.

    Attributes:
        skills (List[str]): Technical and professional competencies of the candidate
        country (List[str]): Countries where the candidate pursued formal education

    Example:
        {
            "skills": ["Python", "Machine Learning", "SQL", "Project Management"],
            "country": ["United States", "India"],
            "domain": "Information Technology"
        }
    """

    domain: str = Field(...,
                        description="The domain of the candidate can be one of SALES/ IT/ FINANCE"
                                    "Returns an empty string if no domain is identified.")

    skills: List[str] = Field(
        ...,
        description="List of technical, professional, and soft skills extracted from the resume. "
        "and domain expertise. Returns an empty list if no skills are identified."
    )

    country: List[str] = Field(
        ...,
        description="List of countries where the candidate completed their formal education, Only extract the country."
        "Returns an empty list if countries are not specified."
    )
