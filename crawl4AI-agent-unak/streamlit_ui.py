from __future__ import annotations
from typing import Literal, TypedDict
import asyncio
import os
from dotenv import load_dotenv

import streamlit as st
import json
import logfire
from supabase import Client
from openai import AsyncOpenAI
from pydantic_ai.messages import (
    ModelMessage, ModelRequest, ModelResponse, SystemPromptPart,
    UserPromptPart, TextPart, ToolCallPart, ToolReturnPart,
    RetryPromptPart, ModelMessagesTypeAdapter
)
from pydantic_ai_expert import pydantic_ai_expert, init_connections

load_dotenv()
logfire.configure(send_to_logfire='never')

class ChatMessage(TypedDict):
    role: Literal['user', 'model']
    timestamp: str
    content: str

def convert_to_model_messages(messages):
    """Convert Streamlit messages to PydanticAI message format."""
    model_messages = []
    for msg in messages:
        if msg["role"] == "user":
            model_messages.append(ModelRequest(parts=[UserPromptPart(content=msg["content"])]))
        elif msg["role"] == "assistant":
            model_messages.append(ModelResponse(parts=[TextPart(content=msg["content"])]))
    return model_messages

async def main():
    # Initialize connections using cached resource
    deps = init_connections()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.title("University of Akureyri Knowledge Base")

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask me about UNAK"):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            try:
                with st.spinner("Searching knowledge base..."):
                    # Convert message history to PydanticAI format
                    model_messages = convert_to_model_messages(st.session_state.messages[:-1])
                    
                    # Use run_stream for async operation
                    async with pydantic_ai_expert.run_stream(
                        prompt,
                        deps=deps,
                        message_history=model_messages
                    ) as result:
                        partial_text = ""
                        async for chunk in result.stream_text(delta=True):
                            partial_text += chunk
                            message_placeholder.markdown(partial_text)
                        
                        # Add the final response to chat history
                        st.session_state.messages.append(
                            {"role": "assistant", "content": partial_text}
                        )
            except Exception as e:
                error_message = f"Error: {str(e)}"
                message_placeholder.error(error_message)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_message}
                )

if __name__ == "__main__":
    asyncio.run(main())
