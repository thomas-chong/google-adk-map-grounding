# Building Location-Aware AI Agents with Google ADK and Maps Grounding

In this comprehensive tutorial, you'll learn how to create intelligent AI agents using Google's Agent Development Kit (ADK) with Google Maps grounding capabilities. We'll build a practical location-aware assistant that can answer questions about places, provide recommendations, and deliver contextually relevant information based on geographical data.

## What is Google ADK?

The Agent Development Kit (ADK) is Google's open-source, code-first Python toolkit for building, evaluating, and deploying sophisticated AI agents. ADK makes agent development feel more like traditional software development, providing flexibility and control over your AI applications.

### Key Features of ADK:
- **Code-first approach**: Define agents with clear, maintainable Python code
- **Flexible orchestration**: Support for sequential, parallel, and loop-based agent workflows
- **Multi-agent architecture**: Enable agent-to-agent communication and delegation
- **Rich tool ecosystem**: Integrate with various APIs and services
- **Gemini optimized**: Built specifically for Google's Gemini models

## What is Google Maps Grounding?

Google Maps grounding enhances AI responses with real-world location and place information from Google's database of 250+ million places. This feature enables agents to provide:

- Accurate location data and addresses
- Business hours and contact information
- Reviews and ratings
- Accessibility features
- Payment options
- Navigation and proximity information

## Important ADK Limitations

⚠️ **Critical Constraints for Google Maps Grounding:**

1. **One Built-in Tool Per Agent**: Each agent can only use ONE built-in tool (like Google Maps grounding)
2. **No Mixed Tools**: You cannot combine built-in tools with custom tools in the same agent
3. **No Built-in Tools in Sub-Agents**: Sub-agents cannot use built-in tools like Google Maps grounding
4. **Architecture Requirement**: You must use separate agents for built-in tools and coordinate through a parent agent

## Prerequisites

Before we start, ensure you have:

- Python 3.9 or higher
- A Google Cloud Project with Vertex AI enabled
- Google AI Studio API key (alternative to Vertex AI)
- Basic knowledge of Python and async programming

## Project Setup

### 1. Environment Setup

First, create a new project directory and set up a virtual environment:

```bash
mkdir location-aware-agent
cd location-aware-agent
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

Install the Google ADK with required dependencies:

```bash
pip install google-adk[agents]
pip install python-dotenv
```

### 3. Project Structure

Create the following project structure:

```
location-aware-agent/
├── .env
├── location_agent/
│   ├── __init__.py
│   ├── agent.py
│   └── tools.py
└── README.md
```

### 4. Environment Configuration

Create a `.env` file with your credentials:

```bash
# .env
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=global
GOOGLE_GENAI_USE_VERTEXAI=True

# Alternative: If using Google AI Studio
# GOOGLE_API_KEY=your-api-key
```

## Understanding the Required Architecture

Due to ADK limitations with built-in tools like Google Maps grounding, we need to design our agent architecture carefully:

### Architecture Pattern:
```
Root Coordinator Agent
├── Maps Specialist Agent (Google Maps grounding only)
└── Custom Tools Agent (local time, recommendations, etc.)
```

### Key Design Principles:
1. **Separation of Concerns**: Built-in tools and custom tools must be in separate agents
2. **Coordination**: Use a parent agent to coordinate between specialists
3. **No Mixing**: Never combine built-in tools with custom tools in the same agent
4. **No Sub-Agent Built-ins**: Sub-agents cannot use built-in tools

## Building the Location-Aware Agent

### Step 1: Create Custom Tools

First, let's create some location-aware tools in `location_agent/tools.py`:

```python
# location_agent/tools.py
import datetime
from typing import Dict, List
from zoneinfo import ZoneInfo

def get_local_time(city: str) -> Dict:
    """Get the current local time for a specified city.

    Args:
        city (str): The name of the city

    Returns:
        dict: Current time information or error message
    """
    timezone_map = {
        "new york": "America/New_York",
        "london": "Europe/London",
        "tokyo": "Asia/Tokyo",
        "paris": "Europe/Paris",
        "sydney": "Australia/Sydney",
        "los angeles": "America/Los_Angeles",
        "chicago": "America/Chicago",
        "berlin": "Europe/Berlin",
        "mumbai": "Asia/Kolkata",
        "shanghai": "Asia/Shanghai"
    }

    city_lower = city.lower()
    if city_lower in timezone_map:
        tz = ZoneInfo(timezone_map[city_lower])
        now = datetime.datetime.now(tz)
        return {
            "status": "success",
            "city": city,
            "current_time": now.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "timezone": str(tz)
        }
    else:
        return {
            "status": "error",
            "error_message": f"Timezone information for '{city}' is not available."
        }

def get_travel_recommendation(city: str, interest: str = "general") -> Dict:
    """Get travel recommendations for a city based on interests.

    Args:
        city (str): The destination city
        interest (str): Type of interest (food, culture, nature, shopping)

    Returns:
        dict: Travel recommendations or error message
    """

    recommendations = {
        "new york": {
            "food": ["Joe's Pizza", "Katz's Delicatessen", "Xi'an Famous Foods"],
            "culture": ["Metropolitan Museum", "Broadway Shows", "Central Park"],
            "shopping": ["Fifth Avenue", "SoHo", "Brooklyn Flea Market"],
            "nature": ["Central Park", "Brooklyn Bridge Park", "The High Line"]
        },
        "paris": {
            "food": ["Le Comptoir du Relais", "L'As du Fallafel", "Breizh Café"],
            "culture": ["Louvre Museum", "Eiffel Tower", "Notre-Dame Cathedral"],
            "shopping": ["Champs-Élysées", "Le Marais", "Galeries Lafayette"],
            "nature": ["Luxembourg Gardens", "Tuileries Garden", "Bois de Vincennes"]
        },
        "tokyo": {
            "food": ["Tsukiji Outer Market", "Ramen Yokocho", "Izakayas in Shibuya"],
            "culture": ["Senso-ji Temple", "Imperial Palace", "Meiji Shrine"],
            "shopping": ["Harajuku", "Ginza", "Akihabara"],
            "nature": ["Ueno Park", "Shinjuku Gyoen", "Mount Takao"]
        }
    }

    city_lower = city.lower()
    if city_lower in recommendations:
        city_recs = recommendations[city_lower]
        if interest in city_recs:
            return {
                "status": "success",
                "city": city,
                "interest": interest,
                "recommendations": city_recs[interest]
            }
        else:
            # Return general recommendations
            return {
                "status": "success",
                "city": city,
                "interest": "general",
                "recommendations": {k: v for k, v in city_recs.items()}
            }
    else:
        return {
            "status": "error",
            "error_message": f"No recommendations available for '{city}'"
        }

def find_nearby_places(place_type: str, location: str) -> Dict:
    """Simulate finding nearby places of a specific type.

    Args:
        place_type (str): Type of place (restaurant, hotel, attraction)
        location (str): Location to search near

    Returns:
        dict: List of nearby places or error message
    """

    # Simulated data - in real implementation, this would use Google Places API
    mock_places = {
        "restaurant": [
            {"name": "The Local Bistro", "rating": 4.5, "price": "$$"},
            {"name": "Corner Café", "rating": 4.2, "price": "$"},
            {"name": "Fine Dining House", "rating": 4.8, "price": "$$$"}
        ],
        "hotel": [
            {"name": "City Center Hotel", "rating": 4.3, "price": "$$$"},
            {"name": "Budget Inn", "rating": 3.9, "price": "$"},
            {"name": "Luxury Resort", "rating": 4.9, "price": "$$$$"}
        ],
        "attraction": [
            {"name": "Historic Downtown", "rating": 4.6, "type": "Cultural"},
            {"name": "City Park", "rating": 4.4, "type": "Nature"},
            {"name": "Art Museum", "rating": 4.7, "type": "Cultural"}
        ]
    }

    if place_type.lower() in mock_places:
        return {
            "status": "success",
            "place_type": place_type,
            "location": location,
            "places": mock_places[place_type.lower()]
        }
    else:
        return {
            "status": "error",
            "error_message": f"Place type '{place_type}' not supported. Try: restaurant, hotel, attraction"
        }
```

### Step 2: Create the Main Agent

Now, let's create the main agent in `location_agent/agent.py`:

```python
# location_agent/agent.py
import os
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.tools import GoogleMapsGroundingTool
from google.genai.types import GenerateContentConfig, GoogleMaps, Tool, ToolConfig, RetrievalConfig, LatLng

# Import our custom tools
from .tools import get_local_time, get_travel_recommendation, find_nearby_places

# Load environment variables
load_dotenv()

# Initialize Google Maps Grounding Tool
google_maps_grounding = GoogleMapsGroundingTool()

# Define the root agent with Google Maps grounding
# Note: Due to ADK limitations, we can only use ONE built-in tool per agent
# Google Maps grounding is a built-in tool, so we create a Maps-only agent
maps_agent = Agent(
    name="maps_specialist",
    model="gemini-2.0-flash",
    description="Specialist agent for Google Maps location queries and place information",
    instruction=(
        "You are a Google Maps specialist. Use Google Maps data to provide "
        "accurate location information, place details, and geographical context. "
        "Always acknowledge the source and provide attribution to Google Maps."
    ),
    generate_content_config=GenerateContentConfig(
        tools=[Tool(google_maps=GoogleMaps())],
        tool_config=ToolConfig(
            retrieval_config=RetrievalConfig(
                lat_lng=LatLng(latitude=40.7128, longitude=-74.006),
                language_code="en_US"
            )
        )
    )
)

# Create a custom tools agent (cannot include built-in tools)
custom_tools_agent = Agent(
    name="location_tools_specialist",
    model="gemini-2.0-flash",
    description="Specialist for local time, travel recommendations, and nearby places",
    instruction=(
        "You are a location tools specialist. Provide local time information, "
        "travel recommendations, and nearby place suggestions using your available tools."
    ),
    tools=[
        get_local_time,
        get_travel_recommendation,
        find_nearby_places
    ]
)

# Root coordinator agent that delegates to specialists
root_agent = Agent(
    name="location_aware_assistant",
    model="gemini-2.0-flash",
    description=(
        "A location-aware AI assistant that coordinates between Google Maps "
        "and custom location tools to provide comprehensive location services."
    ),
    instruction=(
        "You are a helpful location-aware assistant coordinator. You can: \n"
        "1. Delegate Google Maps queries to the maps_specialist \n"
        "2. Delegate time/recommendation queries to location_tools_specialist \n"
        "3. Combine information from both specialists for comprehensive answers \n\n"
        "For location-based queries about places, addresses, or geographical info, "
        "use the maps_specialist. For local time, travel tips, or nearby suggestions, "
        "use the location_tools_specialist. Always provide clear, helpful responses."
    ),
    sub_agents=[maps_agent, custom_tools_agent]
)
```

### Step 3: Initialize the Package

Create the package initializer in `location_agent/__init__.py`:

```python
# location_agent/__init__.py
from . import agent
```

### Step 4: Advanced Agent with Dynamic Location

For more sophisticated location handling, let's create an enhanced version:

```python
# location_agent/advanced_agent.py
import os
import asyncio
from typing import Dict, Optional
from dotenv import load_dotenv

from google.adk.agents import Agent
from google.adk.tools import GoogleMapsGroundingTool
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import (
    GenerateContentConfig, GoogleMaps, Tool, ToolConfig,
    RetrievalConfig, LatLng, Content, Part
)

from .tools import get_local_time, get_travel_recommendation, find_nearby_places

load_dotenv()

class LocationAwareAgent:
    """Enhanced location-aware agent with proper ADK architecture."""

    def __init__(self):
        self.session_service = InMemorySessionService()
        self.app_name = "location_aware_app"

        # Default coordinates (New York)
        self.default_lat = 40.7128
        self.default_lng = -74.006

        self.agent = self._create_agent()
        self.runner = Runner(
            agent=self.agent,
            app_name=self.app_name,
            session_service=self.session_service
        )

    def _create_agent(self) -> Agent:
        """Create the location-aware agent."""
        # Create Maps specialist (built-in tool only)
        maps_agent = Agent(
            name="maps_specialist",
            model="gemini-2.0-flash",
            description="Google Maps specialist for location queries",
            instruction=(
                "You are a Google Maps specialist. Provide accurate location "
                "information, place details, and geographical context using Google Maps data."
            )
        )

        # Create custom tools specialist (no built-in tools)
        tools_agent = Agent(
            name="tools_specialist",
            model="gemini-2.0-flash",
            description="Local information specialist",
            instruction=(
                "You provide local time, travel recommendations, and nearby place "
                "suggestions using your available tools."
            ),
            tools=[
                get_local_time,
                get_travel_recommendation,
                find_nearby_places
            ]
        )

        # Return coordinator agent
        return Agent(
            name="advanced_location_assistant",
            model="gemini-2.0-flash",
            description=(
                "An advanced location-aware AI assistant coordinator that delegates "
                "between Google Maps and custom location tools."
            ),
            instruction=(
                "You are an expert location-aware assistant coordinator. You can:\n"
                "1. Delegate Google Maps queries to maps_specialist\n"
                "2. Delegate local info queries to tools_specialist\n"
                "3. Combine information for comprehensive responses\n\n"
                "Always:\n"
                "- Use appropriate specialists for different query types\n"
                "- Provide proper attribution to Google Maps when using that data\n"
                "- Ask for clarification if location context is unclear\n"
                "- Be helpful and informative while being concise"
            ),
            sub_agents=[maps_agent, tools_agent]
        )

    async def chat_with_location(
        self,
        user_message: str,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        user_id: str = "default_user",
        session_id: str = "default_session"
    ) -> str:
        """
        Chat with the agent using specific location context.

        Args:
            user_message: The user's message
            latitude: Optional latitude for location context
            longitude: Optional longitude for location context
            user_id: User identifier
            session_id: Session identifier

        Returns:
            Agent's response as string
        """

        # Use provided coordinates or defaults
        lat = latitude or self.default_lat
        lng = longitude or self.default_lng

        # Create session if it doesn't exist
        try:
            await self.session_service.get_session(
                app_name=self.app_name,
                user_id=user_id,
                session_id=session_id
            )
        except:
            await self.session_service.create_session(
                app_name=self.app_name,
                user_id=user_id,
                session_id=session_id
            )

        # Update agent config with location context
        config = GenerateContentConfig(
            tools=[Tool(google_maps=GoogleMaps())],
            tool_config=ToolConfig(
                retrieval_config=RetrievalConfig(
                    lat_lng=LatLng(latitude=lat, longitude=lng),
                    language_code="en_US"
                )
            )
        )

        # Create content for the message
        content = Content(role='user', parts=[Part(text=user_message)])

        # Run the agent
        events = self.runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=content,
            config=config
        )

        # Collect the response
        response_parts = []
        async for event in events:
            if event.is_final_response():
                response_parts.append(event.content.parts[0].text)

        return '\n'.join(response_parts) if response_parts else "No response received"

# Create a global instance for easy access
location_agent = LocationAwareAgent()

async def ask_location_agent(
    message: str,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None
) -> str:
    """
    Convenience function to ask the location agent a question.

    Args:
        message: User's question
        latitude: Optional latitude
        longitude: Optional longitude

    Returns:
        Agent's response
    """
    return await location_agent.chat_with_location(
        user_message=message,
        latitude=latitude,
        longitude=longitude
    )
```

## Working with the Separated Architecture

Due to the ADK limitations, you'll need to work with two separate agents:

### Option 1: Use the Coordinator Agent (Recommended)
The `root_agent` coordinates between specialists and provides the best user experience:

```python
# The root_agent automatically delegates to appropriate specialists
# Use this for most interactions
```

### Option 2: Use Individual Specialists Directly

For Google Maps queries only:
```python
# Use maps_agent directly for pure Google Maps queries
# This agent only has access to Google Maps data
```

For custom tools only:
```python
# Use custom_tools_agent for time, recommendations, etc.
# This agent cannot access Google Maps data
```

## Running and Testing the Agent

### Method 1: Using ADK Web UI

The easiest way to test your agent is using the ADK web interface:

```bash
# From your project root directory
adk web
```

This will start a web interface where you can interact with your root coordinator agent directly. The coordinator will automatically delegate to the appropriate specialists.

### Method 2: Command Line

You can also run the agent from the command line:

```bash
adk run location_agent
```

### Method 3: Programmatic Usage

Create a test script `test_agent.py`:

```python
# test_agent.py
import asyncio
from location_agent.advanced_agent import ask_location_agent

async def main():
    # Test basic location questions
    print("Testing location-aware agent...\n")

    # Test 1: General location query
    response1 = await ask_location_agent(
        "What are some good restaurants near Times Square?",
        latitude=40.7580,  # Times Square coordinates
        longitude=-73.9855
    )
    print("Q: What are some good restaurants near Times Square?")
    print(f"A: {response1}\n")

    # Test 2: Travel recommendations
    response2 = await ask_location_agent(
        "I'm visiting Paris next week. What cultural attractions should I see?",
        latitude=48.8566,  # Paris coordinates
        longitude=2.3522
    )
    print("Q: I'm visiting Paris next week. What cultural attractions should I see?")
    print(f"A: {response2}\n")

    # Test 3: Local time query
    response3 = await ask_location_agent("What time is it in Tokyo?")
    print("Q: What time is it in Tokyo?")
    print(f"A: {response3}\n")

    # Test 4: Nearby places
    response4 = await ask_location_agent(
        "Find me hotels near the Eiffel Tower",
        latitude=48.8584,  # Eiffel Tower coordinates
        longitude=2.2945
    )
    print("Q: Find me hotels near the Eiffel Tower")
    print(f"A: {response4}\n")

if __name__ == "__main__":
    asyncio.run(main())
```

Run the test:

```bash
python test_agent.py
```

## Advanced Features and Customization

### 1. Multi-Agent Architecture

You can create specialized agents for different location-related tasks:

```python
# location_agent/specialized_agents.py
from google.adk.agents import Agent
from google.adk.tools import GoogleMapsGroundingTool
from .tools import get_local_time, get_travel_recommendation, find_nearby_places

# IMPORTANT: Due to ADK limitations, we cannot mix built-in tools with custom tools
# Each agent can only have ONE built-in tool, and sub-agents cannot use built-in tools

# Travel recommendation specialist (custom tools only)
travel_agent = Agent(
    name="travel_specialist",
    model="gemini-2.0-flash",
    description="Specialist in travel recommendations and itinerary planning",
    instruction=(
        "You are a travel specialist. Focus on providing detailed travel "
        "recommendations, itineraries, and local insights for destinations."
    ),
    tools=[get_travel_recommendation]
)

# Local information specialist (custom tools only)
local_info_agent = Agent(
    name="local_info_specialist",
    model="gemini-2.0-flash",
    description="Specialist in local time and current information",
    instruction=(
        "You are a local information specialist. Provide accurate local "
        "time and current information about places."
    ),
    tools=[get_local_time]
)

# Google Maps specialist (built-in tool only - cannot be a sub-agent)
maps_only_agent = Agent(
    name="maps_only_agent",
    model="gemini-2.0-flash",
    description="Dedicated Google Maps specialist",
    instruction=(
        "You are a Google Maps specialist. Provide location information, "
        "place details, and geographical context using Google Maps data."
    ),
    generate_content_config=GenerateContentConfig(
        tools=[Tool(google_maps=GoogleMaps())],
        tool_config=ToolConfig(
            retrieval_config=RetrievalConfig(
                lat_lng=LatLng(latitude=40.7128, longitude=-74.006),
                language_code="en_US"
            )
        )
    )
)

# Coordinator agent (custom tools only, cannot include built-in tools as sub-agents)
coordinator_agent = Agent(
    name="location_coordinator",
    model="gemini-2.0-flash",
    description="Coordinates location-related queries across specialist agents",
    instruction=(
        "You coordinate location-related queries. Delegate travel questions "
        "to the travel specialist and local information questions to the "
        "local info specialist. For Google Maps queries, inform users they "
        "need to use the dedicated maps_only_agent separately."
    ),
    sub_agents=[travel_agent, local_info_agent],
    tools=[find_nearby_places]
)
```

### 2. Custom Location Tools

Extend the agent with more sophisticated location tools:

```python
# location_agent/advanced_tools.py
import math
from typing import Dict, Tuple

def calculate_distance(
    lat1: float, lon1: float,
    lat2: float, lon2: float
) -> Dict:
    """Calculate distance between two coordinates using Haversine formula."""

    # Convert latitude and longitude from degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = (math.sin(dlat/2)**2 +
         math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2)
    c = 2 * math.asin(math.sqrt(a))

    # Radius of earth in kilometers
    r = 6371
    distance = c * r

    return {
        "status": "success",
        "distance_km": round(distance, 2),
        "distance_miles": round(distance * 0.621371, 2)
    }

def get_coordinates_from_address(address: str) -> Dict:
    """Simulate geocoding an address to coordinates."""

    # Mock geocoding data
    known_addresses = {
        "times square, new york": (40.7580, -73.9855),
        "eiffel tower, paris": (48.8584, 2.2945),
        "big ben, london": (51.4994, -0.1245),
        "sydney opera house": (-33.8568, 151.2153),
        "statue of liberty": (40.6892, -74.0445)
    }

    address_lower = address.lower()
    for known_addr, coords in known_addresses.items():
        if known_addr in address_lower:
            return {
                "status": "success",
                "address": address,
                "latitude": coords[0],
                "longitude": coords[1]
            }

    return {
        "status": "error",
        "error_message": f"Could not geocode address: {address}"
    }
```

## Best Practices and Tips

### 1. Error Handling

Always implement proper error handling in your tools:

```python
def robust_tool_example(location: str) -> Dict:
    try:
        # Tool logic here
        result = perform_operation(location)
        return {"status": "success", "data": result}
    except Exception as e:
        return {
            "status": "error",
            "error_message": f"An error occurred: {str(e)}"
        }
```

### 2. Rate Limiting and Caching

For production applications, implement rate limiting and caching:

```python
import time
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_location_lookup(location: str) -> Dict:
    """Cache location lookups to reduce API calls."""
    # Expensive operation here
    time.sleep(0.1)  # Simulate API call
    return {"location": location, "data": "cached_result"}
```

### 3. Input Validation

Validate inputs to prevent errors:

```python
def validate_coordinates(latitude: float, longitude: float) -> bool:
    """Validate latitude and longitude values."""
    return (-90 <= latitude <= 90) and (-180 <= longitude <= 180)
```

### 4. Attribution and Compliance

Always provide proper attribution when using Google Maps data:

```python
def format_maps_response(data: Dict) -> str:
    """Format response with proper Google Maps attribution."""
    response = f"Location information: {data}\n\n"
    response += "Data provided by Google Maps. "
    response += "© Google Maps contributors."
    return response
```

## Deployment Options

### 1. Local Development

For development and testing:

```bash
adk web  # Web interface
adk run location_agent  # CLI interface
```

### 2. API Server

Deploy as an API server:

```bash
adk api_server location_agent --port 8000
```

### 3. Vertex AI Agent Engine

Deploy to Google Cloud:

```python
from vertexai import agent_engines

# Deploy to Vertex AI
remote_app = agent_engines.create(
    agent_engine=root_agent,
    requirements=[
        "google-cloud-aiplatform[adk,agent_engines]",
        "python-dotenv"
    ]
)
```

## Troubleshooting Common Issues

### 1. Authentication Issues

- Ensure your Google Cloud credentials are properly set
- Check that Vertex AI API is enabled in your project
- Verify the `GOOGLE_GENAI_USE_VERTEXAI=True` environment variable

### 2. Model Compatibility

- Google Maps grounding only works with Gemini 2.0+ models
- Ensure you're using a supported model like "gemini-2.0-flash"

### 3. Tool Integration Issues

- Check that all tool functions have proper type hints
- Ensure tool functions return dictionaries with consistent structure
- Validate that import statements are correct

### 4. ADK Limitation Issues

- **Error: "Built-in tool cannot be used with other tools"**
  - Solution: Separate built-in tools (Google Maps) into dedicated agents
  - Use a coordinator agent to manage multiple specialists

- **Error: "Built-in tools not supported in sub-agents"**
  - Solution: Never put built-in tools in sub-agents
  - Keep Google Maps grounding in root-level agents only

- **Agent not responding properly**
  - Check that you're using the coordinator pattern correctly
  - Ensure proper delegation instructions in your coordinator agent

## Conclusion

You've now built a comprehensive location-aware AI agent using Google ADK and Maps grounding! This agent can:

- Answer location-based questions with real Google Maps data
- Provide travel recommendations
- Find nearby places and attractions
- Give local time information
- Handle complex location queries with proper attribution

The combination of ADK's flexible agent framework with Google Maps' rich location data creates powerful, contextually aware AI applications. You can extend this foundation to build more sophisticated location-based services, travel assistants, or local discovery applications.

### Next Steps

1. **Extend functionality**: Add more location-based tools like weather, traffic, or event information
2. **Improve UI**: Create a custom web interface for your agent
3. **Add persistence**: Integrate with databases for user preferences and history
4. **Deploy to production**: Use Vertex AI Agent Engine for scalable deployment
5. **Monitor and optimize**: Implement logging, analytics, and performance monitoring

Remember to always follow Google Maps attribution guidelines and respect usage limits when deploying to production environments.