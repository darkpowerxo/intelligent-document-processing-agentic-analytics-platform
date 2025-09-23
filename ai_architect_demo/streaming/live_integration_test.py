"""Live streaming integration test with real Kafka instance.

This test runs against the actual Kafka instance in our Docker setup
to validate the complete streaming architecture.
"""

import asyncio
import json
import time
from typing import List

from ai_architect_demo.streaming.kafka_client import KafkaProducer, KafkaConsumer, KafkaConfig
from ai_architect_demo.streaming.event_schemas import EventFactory, EventType


class LiveStreamingTest:
    """Test streaming with real Kafka instance."""
    
    def __init__(self):
        self.producer_config = KafkaConfig(
            bootstrap_servers="localhost:9092"
        )
        self.consumer_config = KafkaConfig(
            bootstrap_servers="localhost:9092",
            auto_offset_reset="earliest"
        )
        self.producer = KafkaProducer(self.producer_config)
        self.consumer = None
        self.received_events = []
    
    async def test_end_to_end_streaming(self):
        """Test complete streaming pipeline with Kafka."""
        print("🚀 Starting live streaming integration test...")
        print("=" * 60)
        
        try:
            # 1. Connect producer
            print("📤 Connecting producer...")
            await self.producer.connect()
            assert self.producer.is_connected
            print("✅ Producer connected successfully")
            
            # 2. Create test events
            print("\n🎭 Creating test events...")
            events = self._create_test_events()
            print(f"✅ Created {len(events)} test events")
            
            # 3. Produce events
            print("\n📨 Sending events to Kafka...")
            for event in events:
                topic = self._get_topic_for_event(event.event_type)
                await self.producer.send_event(
                    topic=topic,
                    event=event
                )
                print(f"  ➤ Sent {event.event_type} to topic '{topic}'")
            
            print("✅ All events sent successfully")
            
            # 4. Set up consumer and consume events
            print("\n📥 Setting up consumer...")
            self.consumer = KafkaConsumer(
                topics=["documents", "tasks", "agents", "analytics"],
                config=self.consumer_config,
                group_id="test_group"
            )
            
            await self.consumer.connect()
            print("✅ Consumer connected successfully")
            
            # 5. Consume events with timeout
            print("\n🎧 Consuming events...")
            timeout = 15  # seconds
            start_time = time.time()
            consumed_count = 0
            
            async for message in self.consumer.consume():
                if message:
                    consumed_count += 1
                    event_data = json.loads(message.value)
                    print(f"  ✅ Received: {event_data['event_type']}")
                    self.received_events.append(event_data)
                    
                    # Stop when we've received all events
                    if consumed_count >= len(events):
                        print(f"✅ Successfully received all {consumed_count} events!")
                        break
                
                # Timeout check
                if time.time() - start_time > timeout:
                    print(f"⚠️  Timeout reached. Received {consumed_count}/{len(events)} events")
                    break
                
                # Small delay to prevent busy loop
                await asyncio.sleep(0.1)
            
            # 6. Validate received events
            print(f"\n📊 Validation Results:")
            print(f"  • Events sent: {len(events)}")
            print(f"  • Events received: {len(self.received_events)}")
            print(f"  • Success rate: {len(self.received_events) / len(events) * 100:.1f}%")
            
            # Check event types
            sent_types = [e.event_type for e in events]
            received_types = [e['event_type'] for e in self.received_events]
            
            print(f"\n📋 Event Type Summary:")
            for event_type in set(sent_types):
                sent_count = sent_types.count(event_type)
                received_count = received_types.count(event_type)
                print(f"  • {event_type}: {received_count}/{sent_count}")
            
            if len(self.received_events) >= len(events) * 0.8:  # 80% success rate
                print("\n🎉 STREAMING TEST PASSED! 🎉")
                return True
            else:
                print("\n❌ STREAMING TEST FAILED - Too few events received")
                return False
                
        except Exception as e:
            print(f"\n💥 Test failed with error: {e}")
            return False
        
        finally:
            # Cleanup
            print("\n🧹 Cleaning up...")
            if self.producer.is_connected:
                await self.producer.disconnect()
                print("  • Producer disconnected")
            
            if self.consumer and self.consumer.is_connected:
                await self.consumer.disconnect()
                print("  • Consumer disconnected")
    
    def _create_test_events(self) -> List:
        """Create a variety of test events."""
        events = []
        
        # Document events
        for i in range(2):
            event = EventFactory.create_document_event(
                event_type=EventType.DOCUMENT_UPLOADED,
                document_id=f"doc_{i}",
                document_name=f"test_document_{i}.pdf",
                document_type="pdf",
                file_size=1024 * (i + 1),
                source="live_test",
                metadata={"test": True, "index": i}
            )
            events.append(event)
        
        # Task events
        for i in range(2):
            event = EventFactory.create_task_event(
                event_type=EventType.TASK_CREATED,
                task_id=f"task_{i}",
                task_type="document_analysis",
                agent_id=f"agent_{i}",
                source="live_test",
                metadata={"test": True, "index": i}
            )
            events.append(event)
        
        # Agent events
        for i in range(2):
            event = EventFactory.create_agent_event(
                event_type=EventType.AGENT_STATUS_CHANGED,
                agent_id=f"agent_{i}",
                agent_name=f"Test Agent {i}",
                agent_role="document_analyzer",
                status="working",
                current_tasks=i + 1,
                source="live_test",
                metadata={"test": True, "index": i}
            )
            events.append(event)
        
        # Analytics events
        for i in range(2):
            event = EventFactory.create_analytics_event(
                metric_name=f"test_metric_{i}",
                metric_value=42.0 + i,
                metric_type="gauge",
                service="live_test",
                source="live_test",
                metadata={"test": True, "index": i}
            )
            events.append(event)
        
        return events
    
    def _get_topic_for_event(self, event_type: str) -> str:
        """Map event types to topics."""
        if "document" in event_type:
            return "documents"
        elif "task" in event_type:
            return "tasks"
        elif "agent" in event_type:
            return "agents"
        elif "analytics" in event_type:
            return "analytics"
        else:
            return "general"


async def main():
    """Run the live streaming test."""
    print("🎯 Live Streaming Architecture Integration Test")
    print("Testing against real Kafka instance at localhost:9092")
    print("=" * 60)
    
    test = LiveStreamingTest()
    success = await test.test_end_to_end_streaming()
    
    if success:
        print("\n🏆 OVERALL RESULT: SUCCESS! 🏆")
        print("The streaming architecture is working correctly with Kafka!")
    else:
        print("\n💥 OVERALL RESULT: FAILED! 💥")
        print("Check Kafka connection and configuration.")
    
    return success


if __name__ == "__main__":
    asyncio.run(main())