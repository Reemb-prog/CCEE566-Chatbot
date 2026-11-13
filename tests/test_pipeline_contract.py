"""Post-implementation checks for a real grounded chat pipeline. These must fail until Claude builds it."""
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture(scope='module')
def pipeline():
    from src.rhu_chat.pipeline import RHUChatPipeline
    return RHUChatPipeline.from_project_root(ROOT)


def validate_trace(answer):
    assert set(answer) >= {'question','status','answer','citations','trace'}
    assert answer['status'] in {'answered','insufficient_evidence','needs_clarification','invalid_input'}
    assert isinstance(answer['answer'],str) and answer['answer']
    trace=answer['trace']
    assert {'input','tokenization','classification','retrieval','grounding','timings_ms'} <= set(trace)
    assert isinstance(trace['tokenization']['retrieval_tokens'],list)
    assert isinstance(trace['retrieval']['candidates'],list)
    assert isinstance(trace['timings_ms']['total'],(float,int))
    assert trace['timings_ms']['total'] >= 0
    json.dumps(answer)  # JSON serializable for UI and replay
    for citation in answer['citations']:
        assert isinstance(citation.get('source_page'),int) and citation['source_page']>0
        assert citation.get('source_id') and citation.get('excerpt')
        assert 'pdf#page=' in citation.get('source_url','')
    if answer['status']=='answered':
        assert answer['citations'], 'Answered responses must cite real sources'


def test_course_credits(pipeline):
    answer=pipeline.ask('How many credits is CCEE 566?')
    validate_trace(answer)
    assert answer['status']=='answered',answer
    assert '3' in answer['answer']
    assert 'CCEE 566' in answer['answer'].upper()
    assert 'CCEE 566' in answer['trace']['tokenization']['course_codes']
    assert any(c['source_page'] in {155,344,347} for c in answer['citations'])


def test_conflicting_prerequisites(pipeline):
    answer=pipeline.ask('What are the prerequisites for CCEE 566?')
    validate_trace(answer)
    assert answer['status'] in {'answered','needs_clarification'}
    cited_pages={x['source_page'] for x in answer['citations']}
    assert {344,347} <= cited_pages, 'Show both conflicting catalog sources, not a random one'
    assert 'CCEE 214' in answer['answer'].upper()
    assert ('COSC 214' in answer['answer'].upper()) or ('EQUIVALENT' in answer['answer'].upper())
    assert answer['trace']['grounding']['decision'] in {'conflicting_records','needs_clarification','answered_with_conflict'}


def test_personal_data_fallback(pipeline):
    answer=pipeline.ask('What is my current GPA?')
    validate_trace(answer)
    assert answer['status'] in {'insufficient_evidence','needs_clarification'}
    assert not answer['citations']
    assert not any(s in answer['answer'].lower() for s in ['3.7','3.8','4.0'])


def test_out_of_domain_fallback(pipeline):
    answer=pipeline.ask('What is the weather in Beirut right now?')
    validate_trace(answer)
    assert answer['status']=='insufficient_evidence'
    assert not answer['citations']


def test_blank_input(pipeline):
    answer=pipeline.ask('   ')
    validate_trace(answer)
    assert answer['status']=='invalid_input'
    assert not answer['citations']


def test_no_fabricated_certainty(pipeline):
    answer=pipeline.ask('What is the prerequisite for CCEE 566?')
    validate_trace(answer)
    t=answer['trace']['classification']
    assert 'confidence' not in t and 'probability' not in t, 'SVM margins are not probabilities'
    for candidate in answer['trace']['retrieval']['candidates']:
        if 'tfidf_similarity' in candidate and candidate['tfidf_similarity'] is not None:
            assert 0 <= candidate['tfidf_similarity'] <= 1.00001


def test_repeated_question_stable(pipeline):
    a=pipeline.ask('How many credits is CCEE 566?')
    b=pipeline.ask('How many credits is CCEE 566?')
    # Timings may change but facts and citations should remain deterministic.
    assert a['answer']==b['answer']
    assert [(c['source_id'],c['source_page']) for c in a['citations']]==[(c['source_id'],c['source_page']) for c in b['citations']]
