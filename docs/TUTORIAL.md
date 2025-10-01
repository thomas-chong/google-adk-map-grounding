# Complete Tutorial: Building Location-Aware AI Agents with Google ADK

This tutorial walks you through building a simple, effective location-aware AI agent using Google ADK and Maps grounding. We'll focus on creating a single, straightforward agent that leverages the LLM's natural understanding rather than complex custom tools.

## Table of Contents

1. [What We're Building](#what-were-building)
2. [Why Keep It Simple](#why-keep-it-simple)
3. [Setting Up the Project](#setting-up-the-project)
4. [Creating the Agent](#creating-the-agent)
5. [Testing Your Agent](#testing-your-agent)
6. [Understanding Google Maps Grounding](#understanding-google-maps-grounding)
7. [Best Practices](#best-practices)
8. [Deployment](#deployment)

## What We're Building

We're creating a location-aware AI assistant that can:
- 🗺️ Find places, businesses, and attractions using Google Maps
- 🍽️ Recommend restaurants, hotels, and local services
- 🕐 Provide information about time zones and local conditions
- 🧭 Give directions and navigation help
- ✈️ Help with travel planning and local insights

**Key Philosophy**: Rather than building complex custom tools, we let the LLM use its natural language understanding combined with Google Maps data to provide intelligent responses.

## Why Keep It Simple

### The Problem with Complex Approaches
Many developers try to create elaborate multi-agent systems with custom tools, but this often leads to:
- Confusing architecture that's hard to understand
- More potential failure points
- Unnecessary complexity for users
- Difficult maintenance and debugging

### The Power of Simplicity
A single agent with Google Maps grounding can:
- Leverage the LLM's built-in knowledge about locations, time zones, and travel
- Use Google Maps data for current, accurate information
- Provide natural, conversational responses
- Be easier to deploy and maintain

## Setting Up the Project

### Step 1: Environment Setup

```bash
mkdir location-aware-agent
cd location-aware-agent
python -m venv venv
source venv/bin/activate  # Windows: venv\\Scripts\\activate
```

### Step 2: Install Dependencies

```bash
pip install google-adk[agents] python-dotenv
```

### Step 3: Project Structure

Create this simple structure:

```
location-aware-agent/
├── location_agent/
│   ├── __init__.py
│   └── agent.py
├── examples/
│   ├── quick_start.py
│   └── simple_usage.py
├── .env.template
├── requirements.txt
└── README.md
```

### Step 4: Environment Configuration

Create `.env.template`:

```env
# For Vertex AI (Recommended)
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=global
GOOGLE_GENAI_USE_VERTEXAI=True

# OR for Google AI Studio
# GOOGLE_API_KEY=your-api-key
```

Copy to `.env` and add your actual credentials.

## Creating the Agent

### Step 1: Understanding the Agent Structure

Our agent has three main components:
1. **Model**: We use `gemini-2.5-flash` which supports Google Maps grounding
2. **Instructions**: Clear guidance on what the agent can do
3. **Google Maps Tool**: The built-in `google_maps_grounding` tool for location data

### Step 2: The Complete Agent Implementation

Create `location_agent/agent.py`:

```python
"""
Simple location-aware agent using Google Maps grounding.
"""

import os
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.tools import google_maps_grounding

# Load environment variables
load_dotenv()

# Simple location-aware agent with Google Maps grounding
root_agent = Agent(
    name="location_aware_assistant",
    model="gemini-2.5-flash",
    description=(
        "A location-aware AI assistant that provides information about places, "
        "travel recommendations, local information, and answers location-based "
        "questions using Google Maps data. please use the google_maps_grounding tool to get the information."
    ),
    instruction=(
        "You are a helpful location-aware assistant with access to Google Maps data. "
        "You can help users with:\\n\\n"
        "🗺️ **Location Information:**\\n"
        "- Find specific places, addresses, and businesses\\n"
        "- Provide current business hours, reviews, and contact information\\n"
        "- Give directions and navigation help\\n"
        "- Share details about attractions and points of interest\\n\\n"
        "🌍 **Travel & Local Help:**\\n"
        "- Recommend restaurants, hotels, and attractions\\n"
        "- Suggest things to do based on user interests\\n"
        "- Provide local insights and travel tips\\n"
        "- Help with trip planning and itineraries\\n\\n"
        "🕐 **General Assistance:**\\n"
        "- Answer questions about time zones and local time\\n"
        "- Provide weather information when available\\n"
        "- Help with distance and travel time estimates\\n"
        "- Offer cultural and practical local advice\\n\\n"
        "**Guidelines:**\\n"
        "- Always use Google Maps data when available for the most accurate information\\n"
        "- Provide proper attribution to Google Maps when using map data\\n"
        "- Be helpful, informative, and friendly\\n"
        "- Ask for clarification if location context is unclear\\n"
        "- Suggest specific coordinates or addresses when helpful\\n"
        "- Focus on current, practical information that helps users"
    ),
    tools=[google_maps_grounding]
)
```

### Step 3: Package Initialization

Create `location_agent/__init__.py`:

```python
"""
Location-aware AI agent package using Google ADK and Maps grounding.
"""

from . import agent

__version__ = "1.0.0"
```

## Testing Your Agent

### Step 1: Quick Test

Create `examples/quick_start.py` to verify everything works:

```python
import asyncio
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part
from location_agent.agent import root_agent

async def quick_test():
    # Setup
    session_service = InMemorySessionService()
    await session_service.create_session("test_app", "user", "session")

    runner = Runner(root_agent, "test_app", session_service)

    # Test query
    content = Content(role='user', parts=[Part(text="What are some good restaurants in Paris?")])
    events = runner.run_async("user", "session", content)

    # Get response
    async for event in events:
        if event.is_final_response():
            print(f"Response: {event.content.parts[0].text}")

asyncio.run(quick_test())
```

### Step 2: Run the Test

```bash
python examples/quick_start.py
```

### Step 3: Test with ADK Web Interface

```bash
adk web
```

This opens a web interface where you can chat with your agent directly.

## Understanding Google Maps Grounding

### What It Provides

Google Maps grounding gives your agent access to:
- **Real-time business information**: Hours, reviews, contact details
- **Current place data**: What's open now, popular times
- **Navigation information**: Directions, traffic, transit options
- **Rich place details**: Photos, descriptions, amenities

### How Location Context Works

With the simplified approach, location context is provided naturally through user messages:

1. **Natural Language**: Users include location information in their queries
2. **Coordinate Context**: Include coordinates in messages when precise location is needed

```python
# Examples of providing location context in user messages:
"What restaurants are near Times Square in New York?"
"I'm at coordinates 48.8566, 2.3522 - what's around me?"
"Find coffee shops near the Eiffel Tower in Paris"
```

The LLM naturally understands location references and uses the Google Maps grounding tool to get current, accurate information.

### What the LLM Understands Naturally

The LLM already knows a lot without custom tools:
- **Geography**: Cities, countries, landmarks, distances
- **Time zones**: Local time calculations, scheduling
- **Culture**: Local customs, languages, currencies
- **Travel**: Transportation options, typical costs, seasons

Combined with Google Maps data, this creates a powerful location assistant.

## Best Practices

### 1. Clear Instructions

Write clear, specific instructions for your agent:

```python
instruction=(
    "You are a helpful location-aware assistant. "
    "Use Google Maps data for current information. "
    "Provide specific, actionable advice. "
    "Always cite Google Maps when using map data."
)
```

### 2. Natural Location Context

Let users provide location context naturally:
- Include location names in queries: "restaurants in Paris"
- Provide coordinates when precise: "what's near 40.7128, -74.006?"
- The LLM handles location understanding automatically

### 3. Error Handling

The LLM naturally handles many error cases:
- Unknown locations: "I don't have current information about that place"
- Ambiguous queries: "Could you be more specific about which [city] you mean?"
- No data available: "Let me suggest some alternatives"

### 4. Attribution

Always instruct your agent to credit Google Maps:

```python
"Provide proper attribution to Google Maps when using map data"
```

### 5. Helpful Context

Encourage the agent to ask for clarification:

```python
"Ask for clarification if location context is unclear"
"Suggest specific coordinates or addresses when helpful"
```

## Deployment

### Local Development

```bash
# Test locally
adk web

# CLI interface
adk run location_agent
```

### Production Deployment

```bash
# API server
adk api_server location_agent --port 8000
```

### Vertex AI (Recommended for production)

```python
from vertexai import agent_engines

remote_app = agent_engines.create(
    agent_engine=root_agent,
    requirements=["google-cloud-aiplatform[adk,agent_engines]"]
)
```

## Example Interactions

Here are some example interactions with your agent:

**User**: "I'm visiting Tokyo next week. What should I do?"

**Agent**: "Tokyo is an amazing city! Here are some must-do experiences:

🏛️ **Cultural Sites**: Visit Senso-ji Temple in Asakusa, the Meiji Shrine, and the Imperial Palace East Gardens.

🍣 **Food**: Try authentic sushi at Tsukiji Outer Market, visit a traditional izakaya in Shibuya, and experience a proper ramen shop.

🛍️ **Shopping**: Explore Harajuku for youth culture, Ginza for luxury brands, and Akihabara for electronics and anime culture.

🌸 **Districts**: Spend time in Shibuya (famous crossing), Shinjuku (nightlife), and Ueno (museums and park).

*Data sourced from Google Maps*

What specific interests do you have? I can provide more targeted recommendations!"

---

**User**: "Find me a good coffee shop near the Eiffel Tower"

**Agent**: "Here are some excellent coffee shops near the Eiffel Tower according to Google Maps:

☕ **Café de l'Homme** - Upscale café with Eiffel Tower views (in Palais de Chaillot)
⭐ **Du Pain et des Idées** - Artisanal bakery and coffee (short Metro ride)
🥐 **Café Constant** - Charming local café with great pastries

For the closest option, head to **Trocadéro Gardens** area where you'll find several cafés with perfect Eiffel Tower views. The **Café de l'Homme** offers an unforgettable experience if you're looking for something special.

*Information from Google Maps. Hours and availability may vary.*

Would you like directions to any of these, or are you looking for a specific type of coffee experience?"

## Common Questions

### Q: Why not use custom tools for time zones?
**A**: The LLM already understands time zones very well. Adding a custom tool adds complexity without significant benefit.

### Q: Can I add weather information?
**A**: Google Maps grounding sometimes includes weather data. For dedicated weather, you could add a weather API tool, but keep the architecture simple.

### Q: How accurate is the location data?
**A**: Google Maps grounding provides real-time, accurate data including current business hours, reviews, and availability.

### Q: Can I use this for multiple languages?
**A**: Yes! Change the `language_code` in the configuration and update the instructions in your target language.

## Conclusion

By keeping the agent design simple and leveraging the LLM's natural understanding combined with Google Maps grounding, you get:

✅ **Powerful functionality** without complex architecture
✅ **Easy maintenance** with minimal moving parts
✅ **Natural interactions** that users understand intuitively
✅ **Reliable performance** with fewer potential failure points

The key insight is that modern LLMs already understand location, time, and travel concepts very well. Adding real-time Google Maps data creates a compelling location assistant without unnecessary complexity.

Start simple, test thoroughly, and add complexity only when you have a clear need that can't be met with the basic approach.