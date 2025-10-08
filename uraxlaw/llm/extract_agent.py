

from pydantic import BaseModel, Field, field_validator
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
class LawSchema(BaseModel):
    law_name: str
    law_id: Optional[str] = None
    document_type: Optional[str] = None  # Luật, Nghị định, Thông tư...
    issued_by: Optional[str] = None
    signer: Optional[str] = None
    issued_date: Optional[str] = None
    promulgation_date: Optional[str] = None
    effective_date: Optional[str] = None
    expiry_date: Optional[str] = None
    scope: Optional[str] = None  # Phạm vi điều chỉnh
    language: str = "vi"
    aliases: List[str] = Field(default_factory=list)
    modified_by: List[str] = Field(default_factory=list)
    articles: List[Article] = Field(default_factory=list)

    @field_validator("issued_date", "promulgation_date", "effective_date", "expiry_date")
    def _norm_date(cls, v):
        if not v:
            return None
        try:
            return parse_date(v, dayfirst=True).date().isoformat()
        except Exception:
            return v

SYSTEM_INSTRUCTIONS = (
    "You are a structured information extraction assistant for Vietnamese legal texts. "
    "Task: read a Vietnamese law or regulation and emit JSON that conforms to the provided LawSchema. "
    "Requirements:\n"
    "- Identify law_name, law_id (e.g. 13/2008/QH12 or 123/2020/NĐ-CP), document_type (Law/Decree/etc.), issued_by, signer, issued_date, promulgation_date, effective_date, expiry_date, scope, and aliases when available.\n"
    "- Keep all textual content in Vietnamese exactly as written; do not translate or paraphrase.\n"
    "- Set the language field to 'vi'.\n"
    "- Split the document into Articles (article_number, title, content).\n"
    "- Within each Article, split Clauses (clause_number, content).\n"
    "- If a Clause contains Points, list each point_symbol and content in order.\n"
    "- Detect legal relations: modify/add/repeal/replace/reference/define (type=MODIFIES|ADDS|REPEALS|REPLACES|REFERS_TO|DEFINES).\n"
    "- For every relation, provide source (originating document or provision), target (target article/clause/law), and effective_date/expiry_date when present.\n"
    "- Normalize any recognizable dates to yyyy-mm-dd.\n"
    "- When uncertain, return null. Output valid JSON only with no explanations."
)

# Configure ChatGPT/OpenAI model
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
model = OpenAIChatModel(model_name=OPENAI_MODEL,provider=OpenAIProvider(api_key=OPENAI_API_KEY))

# Create agent with structured output type
extract_agent = Agent(
    model=model,
    output_type=LawSchema,
    system_prompt=SYSTEM_INSTRUCTIONS
)

async def extract_schema(text: str) -> LawSchema:
    """Call the PydanticAI agent to get a validated LawSchema."""
    run = await extract_agent.run(text)
    return run.output  # already validated LawSchema
