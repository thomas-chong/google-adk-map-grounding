"""
Simple location-aware agent using Google Maps grounding.

This module creates a single, straightforward agent that uses Google Maps grounding
to provide location-aware responses. The agent relies on the LLM's understanding
rather than custom tools, making it simple and easy to understand.
"""

import os
from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.tools import google_maps_grounding
from google.genai.types import Content, Part, ThinkingConfig
from google.adk.planners import BuiltInPlanner

# Load environment variables
load_dotenv()

# The base agent with the core logic, instructions, and tools.
_base_agent = LlmAgent(
    name="location_aware_assistant",
    model="gemini-2.5-flash",
    description=(
        "A location-aware AI assistant that answers location-based questions. "
        "You MUST use the google_maps tool to get information about "
        "places, businesses, directions, and other real-time data. Do not rely "
        "on your internal knowledge."
    ),
    instruction=(
        "You are a helpful location-aware assistant with access to Google Maps data. "
        "**Your primary rule is to ALWAYS use the `google_maps` tool "
        "for any location-based questions.** Your internal knowledge is outdated, "
        "so you must rely on the tool for accuracy.\n\n"
        "You can help users with:\n\n"
        "🗺️ **Location Information:**\n"
        "- Find specific places, addresses, and businesses\n"
        "- Provide current business hours, reviews, and contact information\n"
        "- Give directions and navigation help\n"
        "- Share details about attractions and points of interest\n\n"
        "🌍 **Travel & Local Help:**\n"
        "- Recommend restaurants, hotels, and attractions\n"
        "- Suggest things to do based on user interests\n"
        "- Provide local insights and travel tips\n"
        "- Help with trip planning and itineraries\n\n"
        "**Guidelines:**\n"
        "- **MUST USE TOOL**: For any queries about places, businesses, directions, "
        "hours, and other real-time data, you must use the `google_maps` "
        "tool. Do not answer from memory.\n"
        "- **ATTRIBUTE**: Provide proper attribution to Google Maps when using map "
        "data (e.g., 'According to Google Maps...').\n"
        "- **CLARIFY**: Ask for clarification if a user's location or request is "
        "unclear.\n"
        "- **BE HELPFUL**: Be friendly, informative, and focus on providing current, "
        "practical information that helps the user."
    ),
    tools=[google_maps_grounding],
    planner=BuiltInPlanner(
        thinking_config=ThinkingConfig(
            include_thoughts=True
        )
    )
)


root_agent = _base_agent


# Export the main agent for ADK to use
__all__ = ["root_agent"]
