import os
import asyncio
from dotenv import load_dotenv
from supabase import create_client, Client
from openai import AsyncOpenAI

load_dotenv()

async def test_connection():
    print("\n=== Testing Supabase Connection ===")
    
    # Initialize clients
    try:
        supabase: Client = create_client(
            os.environ.get("SUPABASE_URL"),
            os.environ.get("SUPABASE_SERVICE_KEY")
        )
        openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        print("✓ Clients initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize clients: {e}")
        return

    # Test 1: Basic connection and content check
    try:
        # First check total count
        result = supabase.table('site_pages').select("count").execute()
        print(f"✓ Connected to database. Found {len(result.data)} rows in site_pages table")
        
        # Then check actual content
        content = supabase.table('site_pages').select("id, url, title").limit(1).execute()
        if content.data:
            print("Sample entry:")
            print(f"ID: {content.data[0]['id']}")
            print(f"URL: {content.data[0]['url']}")
            print(f"Title: {content.data[0]['title']}")
        else:
            print("⚠️ Table is empty! Need to run the crawler first.")
    except Exception as e:
        print(f"✗ Failed to query site_pages: {e}")

    # Test 2: Test embedding generation with a more specific UNAK query
    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input="What student services are available at UNAK?"  # More specific query about content we can see
        )
        embedding = response.data[0].embedding
        print(f"✓ Generated embedding vector of size {len(embedding)}")
    except Exception as e:
        print(f"✗ Failed to generate embedding: {e}")

    # Test 3: Test vector search
    try:
        result = supabase.rpc(
            'match_site_pages',
            {
                "query_embedding": embedding,
                "match_count": 3,
                "filter": {}  # Optional filter parameter
            }
        ).execute()
        print(f"✓ Vector search successful. Found {len(result.data)} matches")
        
        # Print first result if available
        if result.data:
            print("\nSample result:")
            print(f"Title: {result.data[0]['title']}")
            print(f"URL: {result.data[0]['url']}")
            print(f"Similarity: {result.data[0]['similarity']:.2f}")
    except Exception as e:
        print(f"✗ Failed to perform vector search: {e}")

    print("\n=== Environment Variables ===")
    print(f"SUPABASE_URL: {'Set' if os.getenv('SUPABASE_URL') else 'Not set'}")
    print(f"SUPABASE_SERVICE_KEY: {'Set' if os.getenv('SUPABASE_SERVICE_KEY') else 'Not set'}")
    print(f"OPENAI_API_KEY: {'Set' if os.getenv('OPENAI_API_KEY') else 'Not set'}")

if __name__ == "__main__":
    asyncio.run(test_connection()) 