from pathlib import Path
import joblib, sys, types

# fallback tokenizer
def char_tokenizer(s):
    try:
        return list(s)
    except Exception:
        return [c for c in str(s)]

if '__main__' not in sys.modules:
    sys.modules['__main__'] = types.ModuleType('__main__')
setattr(sys.modules['__main__'], 'char_tokenizer', char_tokenizer)

BASE = Path('/workspaces/Checking-Password-Strength-using-Machine-Learning')
model_path = BASE / 'password-ml-app' / 'models' / 'enhanced_password_model_1.pkl'
vec_candidates = [BASE / 'tfidf_vectorizer_1.pkl', BASE / 'tfidf_vectorizer.pkl', BASE / 'password-ml-app' / 'models' / 'tfidf_vectorizer_1.pkl', BASE / 'tfidf_vectorizer.pkl']

print('MODEL_PATH', model_path.exists(), model_path)
model = joblib.load(str(model_path))
print('LOADED MODEL TYPE', type(model))

vec = None
for vp in vec_candidates:
    print('TRY_VEC', vp, vp.exists())
    if vp.exists():
        try:
            vec = joblib.load(str(vp))
            print('LOADED_VEC', vp)
            break
        except Exception as e:
            print('VEC_LOAD_ERROR', vp, e)

pwd = 'koyel'
print('PASSWORD',pwd)
X = None
if vec is not None:
    X = vec.transform([pwd])
    print('X shape', getattr(X,'shape',None))
else:
    print('NO_VECTORIER_LOADED')
    X = [pwd]

try:
    out = model.predict(X)
    print('predict ->', out)
except Exception as e:
    print('PREDICT_ERROR', e)

if hasattr(model, 'predict_proba'):
    try:
        print('predict_proba ->', model.predict_proba(X))
    except Exception as e:
        print('PREDICT_PROBA_ERROR', e)

print('model.n_features_in_', getattr(model,'n_features_in_', None))
print('model.classes_', getattr(model,'classes_', None))
