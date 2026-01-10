from pathlib import Path
import joblib, sys, types

# Provide fallback tokenizer used by pickled vectorizers
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
vec_path = BASE / 'password-ml-app' / 'models' / 'tfidf_vectorizer_1.pkl'
print('MODEL_PATH', model_path.exists(), model_path)
print('VEC_PATH', vec_path.exists(), vec_path)
model = joblib.load(str(model_path))
print('LOADED MODEL TYPE', type(model))
vec = joblib.load(str(vec_path))
print('LOADED VEC TYPE', type(vec))

pwd = 'koyel'
X = vec.transform([pwd])
print('X shape', X.shape)
out = model.predict(X)
print('predict ->', out)
if hasattr(model, 'predict_proba'):
    print('predict_proba ->', model.predict_proba(X))

print('model.attrs n_features_in_', getattr(model, 'n_features_in_', None))
print('model.classes_', getattr(model, 'classes_', None))
