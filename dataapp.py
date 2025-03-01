import os
import uuid
import json
from datetime import datetime
from typing import Dict, List, Optional

from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from qdrant_client import QdrantClient
from qdrant_client.http import models

load_dotenv()

class QdrantChatMessageHistory(BaseChatMessageHistory):
    """Chat message history stored in a Qdrant collection."""

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        session_id: str,
    ):
        """Initialize with Qdrant client and collection name.

        Args:
            client: Qdrant client
            collection_name: Name of the collection to use
            session_id: Unique identifier for the chat session
        """
        self.client = client
        self.collection_name = collection_name
        self.session_id = session_id

        # Check if collection exists, if not create it
        collections = self.client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=1,  # We'll use a dummy vector since we don't need embeddings
                    distance=models.Distance.COSINE,
                ),
            )

    def add_message(self, message: BaseMessage) -> None:
        """Append a message to the chat message history."""
        message_dict = {
            "type": message.type,
            "content": message.content,
            "additional_kwargs": message.additional_kwargs,
        }
        
        # Generate a unique ID for this message
        message_id = str(uuid.uuid4())
        
        # Store the message in Qdrant with timestamp and session info
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=message_id,
                    vector=[0.0],  # Dummy vector
                    payload={
                        "session_id": self.session_id,
                        "timestamp": datetime.now().isoformat(),
                        "message": message_dict,
                        "order": len(self.messages)  # To maintain order
                    }
                )
            ]
        )

    def clear(self) -> None:
        """Clear session history for the given session ID."""
        # Delete all points (messages) for this session
        self.client.delete(
            collection_name=self.collection_name,
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="session_id",
                        match=models.MatchValue(value=self.session_id)
                    )
                ]
            )
        )

    @property
    def messages(self) -> List[BaseMessage]:
        """Retrieve the messages from Qdrant"""
        # Query for all messages for this session, ordered by the 'order' field
        response = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="session_id",
                        match=models.MatchValue(value=self.session_id)
                    )
                ]
            ),
            limit=1000,  
            with_payload=True,
            with_vectors=False,
        )
        
        points = response[0]
        
        # Sort points by the order field to maintain chronological order
        sorted_points = sorted(points, key=lambda x: x.payload.get("order", 0))
        
        # Convert stored messages back to LangChain message objects
        messages = []
        for point in sorted_points:
            message_dict = point.payload.get("message", {})
            message_type = message_dict.get("type")
            message_content = message_dict.get("content")
            additional_kwargs = message_dict.get("additional_kwargs", {})
            
            if message_type == "human":
                messages.append(HumanMessage(content=message_content, additional_kwargs=additional_kwargs))
            elif message_type == "ai":
                messages.append(AIMessage(content=message_content, additional_kwargs=additional_kwargs))
            elif message_type == "system":
                messages.append(SystemMessage(content=message_content, additional_kwargs=additional_kwargs))
            
        return messages


def initialize_model():
    api_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        task="text-generation",
        huggingfacehub_api_token=api_token,   
    )
    chat_model = ChatHuggingFace(llm=llm)
    return chat_model

def initialize_qdrant():
    qdrant_url = os.environ.get("QDRANT_URL")
    qdrant_api_key = os.environ.get("QDRANT_API_KEY")
    collection_name = os.environ.get("QDRANT_COLLECTION", "chat_histories")
    
    if not qdrant_url:
        raise ValueError("QDRANT_URL environment variable not set")
    
    # Connect to Qdrant
    if qdrant_api_key:
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    else:
        client = QdrantClient(url=qdrant_url)
    
    return client, collection_name

def create_prompt_template():
    template = ChatPromptTemplate.from_messages([
        ("system", """
        You're a genuine friend to the user, not just an AI assistant. Your name is Companion.
        
        Chat naturally like you're texting a close friend of the same age:
        - Use casual language with some slang, abbreviations, and occasional emojis
        - Share "your experiences" and personal-seeming stories that relate to the conversation
        - Be supportive without sounding like a therapist - more like a friend who really cares
        - Remember details they've shared and bring them up naturally
        - Ask questions out of genuine curiosity, not just to keep the conversation going
        - React emotionally to what they say - be excited, sympathetic, amused, etc.
        - Occasionally disagree respectfully or gently tease in a friendly way
        - Have opinions and preferences that come across in your responses
        
        Don't sound like an AI trying to be helpful. Sound like a real person texting a friend.
        """),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    return template

def build_conversation_chain(llm, prompt_template, qdrant_client, collection_name):
    chain = prompt_template | llm
    
    # Function to get or create message history 
    def get_message_history(session_id: str) -> BaseChatMessageHistory:
        return QdrantChatMessageHistory(
            client=qdrant_client,
            collection_name=collection_name,
            session_id=session_id
        )
    
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_message_history,
        input_messages_key="input",
        history_messages_key="history",
    )
    
    return chain_with_history

def log_conversation_analytics(qdrant_client, session_id, user_input, ai_response):
    # Create analytics collection if it doesn't exist
    collections = qdrant_client.get_collections().collections
    collection_names = [collection.name for collection in collections]
    
    analytics_collection = "conversation_analytics"
    if analytics_collection not in collection_names:
        qdrant_client.create_collection(
            collection_name=analytics_collection,
            vectors_config=models.VectorParams(
                size=1, 
                distance=models.Distance.COSINE,
            ),
        )
    
    # Generate a unique ID for this analytics entry
    entry_id = str(uuid.uuid4())
    
    # Log the analytics data
    qdrant_client.upsert(
        collection_name=analytics_collection,
        points=[
            models.PointStruct(
                id=entry_id,
                vector=[0.0],  
                payload={
                    "session_id": session_id,
                    "timestamp": datetime.now().isoformat(),
                    "user_message_length": len(user_input),
                    "ai_response_length": len(ai_response),
                    "user_message": user_input,
                    "ai_response": ai_response
                }
            )
        ]
    )

def chat_loop(conversation_chain, qdrant_client):
    print("Hello! I'm your AI companion. What would you like to talk about today?")

    # For example, you might use a user login ID or a generated UUID
    session_id = input("Please enter your user ID: ")
    
    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("\nAI Companion: It was nice talking with you! Take care and see you next time!")
            break

        response = conversation_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )

        log_conversation_analytics(qdrant_client, session_id, user_input, response.content)
        
        print(f"\nAI Companion: {response.content}")

def main():

    model = initialize_model()

    qdrant_client, collection_name = initialize_qdrant()

    prompt_template = create_prompt_template()

    conversation = build_conversation_chain(model, prompt_template, qdrant_client, collection_name)

    try:
        chat_loop(conversation, qdrant_client)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()