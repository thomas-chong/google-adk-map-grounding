# Location-Aware AI Agent with Google ADK and Maps Grounding

A simple, effective location-aware AI agent using Google's Agent Development Kit (ADK) with Google Maps grounding. This implementation focuses on simplicity and leverages the LLM's natural understanding rather than complex custom tools.

## 🌟 Key Features

- **🗺️ Google Maps Integration**: Real-time place information, business details, and navigation
- **🤖 Single Agent Design**: Simple, straightforward architecture that's easy to understand
- **🧠 LLM-Powered**: Leverages the model's built-in knowledge about locations, time zones, and travel
- **📍 Dynamic Location Context**: Support for location-specific queries
- **🚀 Production Ready**: Easy to deploy and maintain

## 🏗️ Simple Architecture

Unlike complex multi-agent systems, this project uses a single agent approach:

```
Single Location-Aware Agent
└── Google Maps Grounding (built-in tool)
```

**Why Simple Works Better:**
- ✅ Easier to understand and maintain
- ✅ Fewer potential failure points
- ✅ Leverages LLM's natural knowledge
- ✅ Quick to deploy and test

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone or create the project
mkdir location-aware-agent
cd location-aware-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install google-adk[agents] python-dotenv
```

### 2. Configure Credentials

Create a `.env` file with your credentials:

```env
# For Vertex AI (Recommended)
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=global
GOOGLE_GENAI_USE_VERTEXAI=True

# OR for Google AI Studio
# GOOGLE_API_KEY=your-api-key
```

### 3. Test the Setup

```bash
# Quick test
python examples/quick_start.py

# Try some examples
python examples/simple_usage.py
```

### 4. Run with ADK Web Interface

```bash
# Start the web interface
adk web
```

This opens a web interface where you can chat with your agent directly!

## 📁 Project Structure

```
location-aware-agent/
├── location_agent/           # 🤖 Simple agent package
│   ├── __init__.py          # Package initialization
│   └── agent.py             # Single agent implementation
├── examples/                # 📚 Usage examples
│   ├── quick_start.py       # Quick setup verification
│   └── simple_usage.py      # Usage examples
├── docs/                    # 📖 Documentation
│   └── TUTORIAL.md          # Complete tutorial
├── .env.template            # 🔧 Environment template
├── requirements.txt         # 📦 Dependencies
└── README.md               # 📋 This file
```

## 🤖 What the Agent Can Do

The agent combines Google Maps data with the LLM's natural understanding to help with:

### 🗺️ **Location Information**
- Find specific places, addresses, and businesses
- Provide current business hours, reviews, and contact information
- Give directions and navigation help
- Share details about attractions and points of interest

### 🌍 **Travel & Local Help**
- Recommend restaurants, hotels, and attractions
- Suggest things to do based on user interests
- Provide local insights and travel tips
- Help with trip planning and itineraries

### 🕐 **General Assistance**
- Answer questions about time zones and local time
- Provide weather information when available
- Help with distance and travel time estimates
- Offer cultural and practical local advice

## 📖 Usage Examples

### Basic Usage

```python
# The agent is automatically configured and ready to use
# Just run: adk web
```

### Programmatic Usage

```python
import asyncio
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part
from location_agent.agent import root_agent

async def ask_agent(question):
    # Setup
    session_service = InMemorySessionService()
    await session_service.create_session("app", "user", "session")
    runner = Runner(root_agent, "app", session_service)

    # Ask question
    content = Content(role='user', parts=[Part(text=question)])
    events = runner.run_async("user", "session", content)

    # Get response
    async for event in events:
        if event.is_final_response():
            return event.content.parts[0].text

# Example usage
response = await ask_agent("What are good restaurants in Paris?")
print(response)
```

### Location-Specific Queries

```python
# The agent automatically uses Google Maps grounding
# Just include location context in your messages for better results

response = await ask_agent(
    "What restaurants are near the Eiffel Tower in Paris?",
    # Location context is handled naturally by the LLM + Google Maps
)

# Or provide coordinates in your message:
response = await ask_agent(
    "What's around me? [I'm at coordinates 48.8566, 2.3522]"
)
```

## 🧪 Testing

Run the example scripts to test functionality:

```bash
# Quick verification that everything works
python examples/quick_start.py

# Try various location-based queries
python examples/simple_usage.py

# Test in the web interface
adk web
```

## 🚀 Deployment Options

### 1. Local Development
```bash
adk web          # Web interface
adk run location_agent  # CLI interface
```

### 2. API Server
```bash
adk api_server location_agent --port 8000
```

### 3. Vertex AI (Production)
```python
from vertexai import agent_engines

remote_app = agent_engines.create(
    agent_engine=root_agent,
    requirements=["google-cloud-aiplatform[adk,agent_engines]"]
)
```

## 🔧 Troubleshooting

### Common Issues

1. **Authentication Issues**
   - Verify your `.env` file is configured correctly
   - For Vertex AI: Ensure `GOOGLE_CLOUD_PROJECT` is set
   - For AI Studio: Ensure `GOOGLE_API_KEY` is set

2. **Model Compatibility**
   - Google Maps grounding requires Gemini 2.0+ models
   - The agent uses `gemini-2.0-flash` by default

3. **No Response or Errors**
   - Check internet connection
   - Verify API credentials are valid
   - Ensure required APIs are enabled in Google Cloud

### Getting Help

1. Run `python examples/quick_start.py` to test basic functionality
2. Check the [tutorial](docs/TUTORIAL.md) for detailed setup instructions
3. Review the [ADK documentation](https://google.github.io/adk-docs/)

## 💡 Example Interactions

**You**: "I'm visiting Tokyo next week. What should I do?"

**Agent**: "Tokyo is an amazing city! Here are some must-do experiences:

🏛️ **Cultural Sites**: Visit Senso-ji Temple in Asakusa, the Meiji Shrine, and the Imperial Palace East Gardens.

🍣 **Food**: Try authentic sushi at Tsukiji Outer Market, visit a traditional izakaya in Shibuya, and experience a proper ramen shop.

🛍️ **Shopping**: Explore Harajuku for youth culture, Ginza for luxury brands, and Akihabara for electronics and anime culture.

*Data sourced from Google Maps*

What specific interests do you have? I can provide more targeted recommendations!"

---

**You**: "Find me a coffee shop near the Eiffel Tower"

**Agent**: "Here are some excellent coffee shops near the Eiffel Tower according to Google Maps:

☕ **Café de l'Homme** - Upscale café with Eiffel Tower views
🥐 **Du Pain et des Idées** - Artisanal bakery and coffee
☕ **Café Constant** - Charming local café with great pastries

*Information from Google Maps. Hours and availability may vary.*

Would you like directions to any of these?"

## 🤝 Why This Approach Works

### LLM Natural Knowledge + Google Maps Data = Powerful Assistant

Modern LLMs already understand:
- ✅ Geography and landmarks
- ✅ Time zones and scheduling
- ✅ Travel and transportation
- ✅ Cultural context and customs

Adding Google Maps provides:
- ✅ Real-time business information
- ✅ Current hours and availability
- ✅ Reviews and ratings
- ✅ Navigation and directions

**Result**: A sophisticated location assistant without complex architecture!

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🔮 Next Steps

- **Try it out**: Run `adk web` and start asking location-based questions
- **Customize**: Modify the agent instructions for your specific use case
- **Deploy**: Use Vertex AI Agent Engine for production deployment
- **Extend**: Add additional tools only if you have specific needs the LLM can't handle

---

**Note**: This implementation demonstrates that simple designs often work better than complex ones. The LLM's natural understanding combined with Google Maps data creates a powerful location assistant without unnecessary complexity.