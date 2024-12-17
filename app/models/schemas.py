from pydantic import BaseModel
from typing import List, Dict, Any
from typing import Optional

# Define a Pydantic model for the chat history
class ChatMessage(BaseModel):
    user: str
    bot: str
    businessId: Optional[int] = None
    businessName: Optional[str] = None


# Define another model to handle the entire request body
class AnswerRetrievalRequest(BaseModel):
    chat_history: List[ChatMessage]
    task_history: List[ChatMessage]
    query: str
    user_id: Optional[int] = None
    businessId: Optional[int] = None
    businessName: Optional[str] = None


class EditKnowledgebase(BaseModel):

    query: str
    user_id: Optional[int] = None
    businessId: Optional[int] = None
    businessName: Optional[str] = None


# Define a Pydantic model for the chat history
class ScrapingURL(BaseModel):
    url: str
    businessId: Optional[int] = None
    businessName: Optional[str] = None


# Define another model to handle the entire request body
class fileLinks(BaseModel):
    files_list: List
    user_id: Optional[int] = None
    businessId: Optional[int] = None
    businessName: Optional[str] = None
    prev_files: Optional[str] = None


# Define another model to handle the entire request body
class urlLinks(BaseModel):
    url_list: List
    user_id: Optional[int] = None
    scraping_depth: Optional[int] = None
    businessId: Optional[int] = None
    businessName: Optional[str] = None
    interval_seconds: str
    # prev_file: Optional[str] = None

# Define another model to handle the entire request body
class driveLinks(BaseModel):
    url_list: List
    user_id: Optional[int] = None
    scraping_depth: Optional[int] = None
    businessId: Optional[int] = None
    businessName: Optional[str] = None
    interval_seconds: str
    # prev_file: Optional[str] = None


# Define another model to handle the entire request body
class deleteknowledgebase(BaseModel):
    businessId: Optional[int] = None
    businessName: Optional[str] = None

# # Define another model to handle the entire request body
# class business_exp(BaseModel):
#     task_description: str
#     task_instructions: str
#     task_parameters: Dict[str, Any]  # Dictionary with dynamic keys and values of any type
#     output_format: str

# Define another model to handle the entire request body
class business_exp(BaseModel):
    task_description: str
    task_instructions: str
    task_parameters: Dict[str, Any]  # Dictionary with dynamic keys and values of any type
    user_id: Optional[int] = None
    output_format: Dict[str, Any]  # Dictionary with dynamic keys and values of any type
    businessId: Optional[int] = None
    businessName: Optional[str] = None

# Define another model to handle the entire request body
class doc_status_schema(BaseModel):
    user_id: Optional[int] = None

class business_exp_search(BaseModel):
    task_description: str
    task_instructions: str
    task_parameters: Dict[str, Any]  # Dictionary with dynamic keys and values of any type
    user_id: Optional[int] = None
    output_format: Dict[str, Any]  # Dictionary with dynamic keys and values of any type
    businessId: Optional[int] = None
    businessName: Optional[str] = None

class packaged_products_search(BaseModel):
    item: str
    user_id: Optional[int] = None
    businessId: Optional[int] = None
    businessName: Optional[str] = None

class streamlinedautofill(BaseModel):
    chat_history: List[ChatMessage]
    task_history: List[ChatMessage]
    user_id: Optional[int] = None

# Define a Pydantic model for the input
class businessNavigatorSchema(BaseModel):
    description: str
    no_of_buss:str

# Define a Pydantic model for the input
class businessRecommenderSchema(BaseModel):
    description: str

class jsonSchema(BaseModel):
    user_id: Optional[int] = None
    url: str
    signup: Optional[int] = None
    businessId: Optional[int] = None
    businessName: Optional[str] = None

class blogSchema(BaseModel):
#     user_id: int
    url:str

class trendPred(BaseModel):
#     user_id: int
    user_id: Optional[int] = None
    url:str

# Define a Pydantic model for the input
class business_navigate(BaseModel):
    description: str
    no_of_buss:str

# Define a Pydantic model for the input
class business_recommend(BaseModel):
    description: str