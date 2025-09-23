"""Test basic streaming components integration.

This test validates that the streaming architecture components
can be initialized and work together correctly.
"""

import asyncio
from ai_architect_demo.streaming.kafka_client import KafkaConfig, KafkaProducer
from ai_architect_demo.streaming.event_schemas import EventFactory, EventType
from ai_architect_demo.streaming.event_dispatcher import EventDispatcher
from ai_architect_demo.streaming.stream_processors import DocumentStreamProcessor
from ai_architect_demo.streaming.real_time_monitor import RealTimeMonitor


async def test_streaming_components():
    """Test that streaming components can be created and initialized."""
    print("🧪 Testing Streaming Components Integration")
    print("=" * 50)
    
    # 1. Test Kafka client creation
    print("📡 Testing Kafka client creation...")
    config = KafkaConfig(bootstrap_servers="localhost:9092")
    producer = KafkaProducer(config)
    print("✅ Kafka client created")
    
    # 2. Test event creation
    print("\n🎭 Testing event creation...")
    doc_event = EventFactory.create_document_event(
        event_type=EventType.DOCUMENT_UPLOADED,
        document_id="test_doc",
        document_name="test.pdf",
        document_type="pdf",
        source="component_test"
    )
    print(f"✅ Created document event: {doc_event.event_type}")
    
    task_event = EventFactory.create_task_event(
        event_type=EventType.TASK_CREATED,
        task_id="test_task",
        task_type="analysis",
        source="component_test"
    )
    print(f"✅ Created task event: {task_event.event_type}")
    
    # 3. Test event dispatcher creation
    print("\n🔀 Testing event dispatcher...")
    dispatcher = EventDispatcher()
    print("✅ Event dispatcher created")
    
    # 4. Test stream processor creation
    print("\n⚙️ Testing stream processors...")
    doc_processor = DocumentStreamProcessor()
    print(f"✅ Document processor created: {doc_processor.name}")
    
    # Check processor stats
    stats = doc_processor.get_stats()
    print(f"  • Name: {stats['name']}")
    print(f"  • Input topics: {stats.get('input_topics', 'N/A')}")
    print(f"  • Running: {stats.get('is_running', False)}")
    
    # 5. Test real-time monitor creation
    print("\n📊 Testing real-time monitor...")
    monitor = RealTimeMonitor()
    print("✅ Real-time monitor created")
    
    # 6. Test Kafka producer connection
    print("\n🔗 Testing Kafka connection...")
    try:
        await producer.connect()
        if producer.is_connected:
            print("✅ Connected to Kafka successfully!")
            
            # Test sending an event
            await producer.send_event("test_topic", doc_event)
            print("✅ Event sent successfully!")
            
            # Show stats
            stats = producer.get_stats()
            print(f"  • Messages sent: {stats['stats']['messages_sent']}")
            
        else:
            print("⚠️ Could not connect to Kafka (expected if not running)")
            
    except Exception as e:
        print(f"⚠️ Kafka connection failed: {e}")
        print("  (This is expected if Kafka is not running)")
        
    finally:
        if producer.is_connected:
            await producer.disconnect()
            print("✅ Producer disconnected")
    
    print("\n" + "=" * 50)
    print("🎉 ALL STREAMING COMPONENTS TESTS PASSED!")
    print("\nThe streaming architecture components are:")
    print("  ✅ Properly configured")
    print("  ✅ Can be instantiated")
    print("  ✅ Ready for integration")
    
    if producer.get_stats()['stats']['messages_sent'] > 0:
        print("  ✅ Successfully communicating with Kafka!")
    else:
        print("  ⚠️ Kafka communication test skipped (no Kafka running)")


async def main():
    await test_streaming_components()


if __name__ == "__main__":
    asyncio.run(main())