"""Simple Kafka connectivity test.

Just tests that we can connect to Kafka and send a basic message.
"""

import asyncio
from ai_architect_demo.streaming.kafka_client import KafkaProducer, KafkaConfig
from ai_architect_demo.streaming.event_schemas import EventFactory, EventType


async def test_kafka_connectivity():
    """Simple test to verify Kafka is working."""
    print("🎯 Simple Kafka Connectivity Test")
    print("=" * 40)
    
    # Create producer with correct config
    config = KafkaConfig(bootstrap_servers="localhost:9092")
    producer = KafkaProducer(config)
    
    try:
        # Test connection
        print("📤 Testing Kafka connection...")
        await producer.connect()
        
        if producer.is_connected:
            print("✅ Successfully connected to Kafka!")
        else:
            print("❌ Failed to connect to Kafka")
            return False
        
        # Create a test event
        print("\n🎭 Creating test event...")
        event = EventFactory.create_document_event(
            event_type=EventType.DOCUMENT_UPLOADED,
            document_id="connectivity_test",
            document_name="test.pdf",
            document_type="pdf",
            source="connectivity_test"
        )
        print(f"✅ Created event: {event.event_type}")
        
        # Send the event
        print("\n📨 Sending test event to Kafka...")
        await producer.send_event(topic="test_documents", event=event)
        print("✅ Event sent successfully!")
        
        # Check producer stats
        print("\n📊 Producer Statistics:")
        stats = producer.get_stats()
        for key, value in stats.items():
            print(f"  • {key}: {value}")
        
        print("\n🎉 KAFKA CONNECTIVITY TEST PASSED! 🎉")
        return True
        
    except Exception as e:
        print(f"\n💥 Test failed: {e}")
        return False
        
    finally:
        # Cleanup
        if producer.is_connected:
            await producer.disconnect()
            print("\n🧹 Producer disconnected")


async def main():
    success = await test_kafka_connectivity()
    
    if success:
        print("\n🏆 Kafka is working correctly!")
        print("The streaming architecture can communicate with Kafka.")
    else:
        print("\n💥 Kafka connectivity failed!")
        print("Check that Kafka is running on localhost:9092")


if __name__ == "__main__":
    asyncio.run(main())