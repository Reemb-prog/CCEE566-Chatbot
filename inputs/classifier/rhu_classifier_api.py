# Portable RHU V3.1 classifier inference. Needs pandas, numpy, scikit-learn, joblib.
# Only load the accompanying joblib file from a trusted source.
import re
import numpy as np
import joblib


def normalize_questions(texts, regex_steps):
    out = []
    for text in texts:
        s = str(text).lower()
        for pat,rep in regex_steps:
            s = re.sub(pat,rep,s)
        out.append(s.strip())
    return out


def _scores(model, normalized):
    x = model.named_steps['vectorizer'].transform(normalized)
    estimator = model.named_steps['classifier']
    if hasattr(estimator,'decision_function'):
        scores = np.asarray(estimator.decision_function(x))
    elif hasattr(estimator,'predict_log_proba'):
        scores = np.asarray(estimator.predict_log_proba(x))
    else:
        raise TypeError('Classifier cannot rank alternate intent classes')
    if scores.ndim == 1:
        scores = np.column_stack([-scores,scores])
    return scores,np.asarray(estimator.classes_)


def load_bundle(path='rhu_classifier_v31.joblib'):
    return joblib.load(path) # trusted local artifact ONLY


def predict_questions(bundle, questions):
    if isinstance(questions,str):
        questions = [questions]
    else:
        questions = list(questions)
    if any(not isinstance(q,str) for q in questions):
        raise TypeError('Each input question must be a string')
    results = [{'question_text':q,'predicted_category':None,'predicted_intent':None,
                'raw_category':None,'raw_intent':None,'routing_status':'invalid_input',
                'review_needed':True} for q in questions]
    valid_idx = [i for i,q in enumerate(questions) if q.strip() and re.search(r'[A-Za-z]',q)]
    if not valid_idx:
        return results
    normalized = normalize_questions([questions[i] for i in valid_idx],bundle['regex_steps'])
    cats = bundle['models']['category'].predict(normalized)
    ints = bundle['models']['intent'].predict(normalized)
    scores,classes = _scores(bundle['models']['intent'],normalized)
    mapping=bundle['intent_to_category']
    policy=bundle['routing_policy']
    for j,i in enumerate(valid_idx):
        cat, intent = str(cats[j]),str(ints[j])
        raw_cat,raw_int = cat,intent
        status='ok'
        if mapping.get(intent)!=cat:
            if policy=='intent_governs':
                if intent in mapping:
                    cat=mapping[intent]
                    status='reconciled_from_intent'
                else:
                    intent=None
                    status='review_missing_mapping'
            elif policy=='category_governs':
                candidates = [k for k,x in enumerate(classes) if mapping.get(str(x))==cat]
                if candidates:
                    intent=str(classes[max(candidates,key=lambda k:scores[j,k])])
                    status='reconciled_from_category'
                else:
                    intent=None
                    status='review_no_intent_for_category'
            elif policy=='review_on_conflict':
                intent=None
                status='review_conflicting_intent'
            else:
                raise ValueError('Unexpected routing policy')
        assert intent is None or mapping[intent]==cat
        results[i]={'question_text':questions[i], 'predicted_category':cat,
            'predicted_intent':intent,'raw_category':raw_cat,'raw_intent':raw_int,
            'routing_status':status,'review_needed':status!='ok'}
    return results


def predict_one(bundle,question):
    return predict_questions(bundle,[question])[0]


def make_qa_request(prediction):
    # Soft routing hints only. The retrieval module must independently verify evidence and cite pages.
    if not isinstance(prediction,dict) or 'question_text' not in prediction:
        raise TypeError('Provide one predict_one/predict_questions result dictionary')
    valid = prediction.get('routing_status') != 'invalid_input'
    return {
        'original_question': prediction['question_text'],
        'input_valid': bool(valid),
        'category_hint': prediction.get('predicted_category') if valid else None,
        'intent_hint': prediction.get('predicted_intent') if valid else None,
        'routing_status': prediction.get('routing_status'),
        'routing_review_needed': bool(prediction.get('review_needed')),
        'category_hard_filter': False,
        'intent_hard_filter': False,
        'retrieve_from_full_catalog': bool(valid),
        'requires_verified_evidence': True,
        'requires_source_citation': True,
        'fallback_if_no_evidence': True,
    }
