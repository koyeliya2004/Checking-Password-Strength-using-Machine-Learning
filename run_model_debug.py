from pathlib import Path
import joblib, sys

paths = [Path('password-ml-app/models/enhanced_password_model_1.pkl'), Path('enhanced_password_model_1.pkl'), Path('enhancedpasswordmodel.pkl'), Path('Uenhancedpasswordmodel.pkl'), Path('enhancedpasswordmodel.pkl')]
model = None
for p in paths:
    print('TRY_MODEL', p)
    if p.exists():
        try:
            model = joblib.load(str(p))
            print('LOADED_MODEL', p)
            break
        except Exception as e:
            print('MODEL_LOAD_ERROR', p, e)

if model is None:
    print('NO_MODEL_FOUND')
    sys.exit(2)

vec_paths = [Path('password-ml-app/models/tfidf_vectorizer_1.pkl'), Path('tfidf_vectorizer_1.pkl'), Path('password-ml-app/models/tfidf_vectorizer.pkl'), Path('tfidf_vectorizer.pkl'), Path('tfidf_vectorizer (2).pkl')]
vec = None
for vp in vec_paths:
    print('TRY_VEC', vp)
    if vp.exists():
        try:
            vec = joblib.load(str(vp))
            print('LOADED_VEC', vp)
            break
        except Exception as e:
            print('VEC_LOAD_ERROR', vp, e)

pwd = 'koyel'
print('PASSWORD', pwd)
X = None
if vec is not None:
    try:
        X = vec.transform([pwd])
        print('X_SHAPE', getattr(X, 'shape', None))
    except Exception as e:
        print('VEC_TRANSFORM_ERROR', e)
else:
    X = [pwd]

# Try predict
try:
    out = model.predict(X)
    print('PREDICT_OUTPUT_TYPE', type(out))
    print('PREDICT_OUTPUT', out)
except Exception as e:
    print('PREDICT_ERROR', e)

# Try predict_proba
try:
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X)
        print('PREDICT_PROBA_TYPE', type(proba))
        try:
            import numpy as np
            print('PREDICT_PROBA', np.array(proba).shape, proba)
        except Exception:
            print('PREDICT_PROBA', proba)
    else:
        print('NO_PREDICT_PROBA')
except Exception as e:
    print('PREDICT_PROBA_ERROR', e)

# Inspect some attrs
for a in ('n_features_in_', 'classes_', 'feature_names_in_', 'vocabulary_'):
    print('ATTR', a, getattr(model, a, None))

print('MODEL_TYPE', type(model))
