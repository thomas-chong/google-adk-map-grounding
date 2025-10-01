"""
Simple usage examples for the location-aware agent.

This script demonstrates how to use the simplified single agent
with Google Maps grounding.
"""

import asyncio
import sys
import os

# Add the parent directory to the path so we can import our agent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part
from location_agent.agent import root_agent


async def create_agent_runner():
    """Create and return an agent runner."""
    session_service = InMemorySessionService()
    app_name = "location_aware_app"

    runner = Runner(
        agent=root_agent,
        app_name=app_name,
        session_service=session_service
    )

    # Create a session
    user_id = "demo_user"
    session_id = "demo_session"

    await session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id
    )

    return runner, user_id, session_id


async def ask_agent(runner, user_id, session_id, question, latitude=None, longitude=None):
    """Ask the agent a question."""

    # Include location context in the message if coordinates are provided
    if latitude and longitude:
        question_with_context = f"{question}\n\n[Location context: I'm near coordinates {latitude}, {longitude}]"
    else:
        question_with_context = question

    # Create the message
    content = Content(role='user', parts=[Part(text=question_with_context)])

    # Run the agent
    events = runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=content
    )

    # Collect the response
    response_parts = []
    async for event in events:
        if event.is_final_response():
            response_parts.append(event.content.parts[0].text)

    return '\n'.join(response_parts) if response_parts else "No response received"


async def run_examples():
    """Run simple usage examples."""
    print("🌍 Location-Aware Agent - Simple Usage Examples")
    print("=" * 60)

    # Create the agent runner
    runner, user_id, session_id = await create_agent_runner()

    # Example 1: Find restaurants
    print("\n1. 🍽️ Finding Restaurants")
    print("-" * 30)

    response = await ask_agent(
        runner, user_id, session_id,
        "What are some good restaurants near Times Square in New York?",
        latitude=40.7580,  # Times Square coordinates
        longitude=-73.9855
    )
    print("Q: What are some good restaurants near Times Square in New York?")
    print(f"A: {response}\n")

    # Example 2: Travel recommendations
    print("2. ✈️ Travel Recommendations")
    print("-" * 30)

    response = await ask_agent(
        runner, user_id, session_id,
        "I'm visiting Paris for 3 days. What are the must-see attractions and good places to eat?",
        latitude=48.8566,  # Paris coordinates
        longitude=2.3522
    )
    print("Q: I'm visiting Paris for 3 days. What are the must-see attractions and good places to eat?")
    print(f"A: {response}\n")

    # Example 3: Local time and weather
    print("3. 🕐 Local Information")
    print("-" * 25)

    response = await ask_agent(
        runner, user_id, session_id,
        "What time is it in Tokyo right now? Also, what's the weather like there?"
    )
    print("Q: What time is it in Tokyo right now? Also, what's the weather like there?")
    print(f"A: {response}\n")

    # Example 4: Directions and navigation
    print("4. 🗺️ Directions and Navigation")
    print("-" * 35)

    response = await ask_agent(
        runner, user_id, session_id,
        "How do I get from the Eiffel Tower to the Louvre Museum in Paris? What's the best way to travel?",
        latitude=48.8584,  # Eiffel Tower coordinates
        longitude=2.2945
    )
    print("Q: How do I get from the Eiffel Tower to the Louvre Museum in Paris?")
    print(f"A: {response}\n")

    # Example 5: Local recommendations
    print("5. 🏨 Hotel and Accommodation")
    print("-" * 30)

    response = await ask_agent(
        runner, user_id, session_id,
        "I need a hotel near Central Park in New York. What are some good options with good reviews?",
        latitude=40.7829,  # Central Park coordinates
        longitude=-73.9654
    )
    print("Q: I need a hotel near Central Park in New York. What are some good options?")
    print(f"A: {response}\n")

    # Example 6: Distance and travel time
    print("6. 📏 Distance and Travel Time")
    print("-" * 35)

    response = await ask_agent(
        runner, user_id, session_id,
        "How far is it from London to Paris? What are the different ways to travel between these cities?"
    )
    print("Q: How far is it from London to Paris? What are the travel options?")
    print(f"A: {response}\n")


async def main():
    """Run all examples."""
    try:
        print("Starting Location-Aware Agent Examples...")
        print("Note: Make sure you have set up your .env file with proper credentials.\n")

        await run_examples()

        print("\n✅ All examples completed successfully!")
        print("\n💡 Tips:")
        print("- The agent uses Google Maps data for accurate, real-time information")
        print("- Provide specific locations or coordinates for better results")
        print("- The agent can help with restaurants, hotels, attractions, directions, and local info")
        print("- Try asking about business hours, reviews, and current information")

        print("\n🚀 Next Steps:")
        print("- Run 'adk web' to test in the ADK web interface")
        print("- Try your own location-based questions")
        print("- Experiment with different cities and types of queries")

    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("\nTroubleshooting:")
        print("1. Check your .env file configuration")
        print("2. Ensure you have proper Google Cloud/AI Studio credentials")
        print("3. Verify your internet connection")
        print("4. Make sure all dependencies are installed: pip install -r requirements.txt")


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())