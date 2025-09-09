from models.pipeline import IntentionPipeline

def test_basic_query():
    pipeline = IntentionPipeline()
    query = "Show me when the speaker explains PLA vs ABS printing."
    result = pipeline.process(query)
    print(result)
    assert "intent" in result or "llm_output" in result
