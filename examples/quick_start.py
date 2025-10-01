"""
Quick start script for testing the location-aware agent.

This script provides a simple way to test the agent setup and basic functionality.
Run this first to verify everything is working correctly.
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


async def quick_test():
    """Run a quick test of the location-aware agent."""
    print("🚀 Location-Aware Agent - Quick Start Test")
    print("=" * 50)

    try:
        # Setup agent runner
        session_service = InMemorySessionService()
        app_name = "quick_test_app"
        user_id = "test_user"
        session_id = "test_session"

        # Create session
        await session_service.create_session(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id
        )

        # Create runner
        runner = Runner(
            agent=root_agent,
            app_name=app_name,
            session_service=session_service
        )

        # Test 1: Simple query
        print("\n1. 🌍 Testing basic functionality...")
        content = Content(role='user', parts=[Part(text="What are some famous landmarks in New York City?")])

        events = runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=content
        )

        response_parts = []
        async for event in events:
            if event.is_final_response():
                response_parts.append(event.content.parts[0].text)

        response = '\n'.join(response_parts) if response_parts else "No response received"
        print(f"✅ Success! Response: {response[:100]}...")

        # Test 2: Location-specific query with context in message
        print("\n2. 📍 Testing location context...")

        content = Content(role='user', parts=[Part(text="What restaurants are near Times Square in New York? I'm looking for good places to eat.")])

        events = runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=content
        )

        response_parts = []
        async for event in events:
            if event.is_final_response():
                response_parts.append(event.content.parts[0].text)

        response = '\n'.join(response_parts) if response_parts else "No response received"
        print(f"✅ Success! Response: {response[:100]}...")

        print("\n🎉 Quick test completed successfully!")
        print("\n✅ Your location-aware agent is working correctly!")

        print("\n📚 What you can do next:")
        print("• Run 'python examples/simple_usage.py' for more examples")
        print("• Use 'adk web' to test in the web interface")
        print("• Try asking about restaurants, hotels, attractions, and directions")
        print("• Ask for travel recommendations for different cities")

    except Exception as e:
        print(f"\n❌ Quick test failed: {e}")
        print("\n🔧 Troubleshooting steps:")
        print("1. Check your .env file configuration:")
        print("   - Copy .env.template to .env")
        print("   - Add your Google Cloud project ID key and GOOGLE_GENAI_USE_VERTEXAI=True")
        print("2. Ensure you have the required dependencies:")
        print("   - Run: pip install -r requirements.txt")
        print("3. Verify your credentials:")
        print("   - For Vertex AI: Set GOOGLE_CLOUD_PROJECT and GOOGLE_GENAI_USE_VERTEXAI=True")
        print("4. Check internet connection and API access")


if __name__ == "__main__":
    asyncio.run(quick_test())