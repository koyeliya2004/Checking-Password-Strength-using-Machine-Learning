from pathlib import Path
import joblib, sys
BASE = Path('/workspaces/Checking-Password-Strength-using-Machine-Learning')
model_paths = [BASE / 'password-ml-app' / 'models' / 'enhanced_password_model_1.pkl',
               BASE / 'enhanced_password_model_1.pkl',
               BASE / 'enhancedpasswordmodel.pkl',
               BASE / 'Uenhancedpasswordmodel.pkl',
               BASE / 'password_strength_model.pkl']
model=None
for p in model_paths:
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

vec_paths = [BASE / 'tfidf_vectorizer_1.pkl', BASE / 'tfidf_vectorizer.pkl', BASE / 'password-ml-app' / 'models' / 'tfidf_vectorizer_1.pkl']
vec=None
for vp in vec_paths:
    print('TRY_VEC', vp)
    if vp.exists():
        try:
            vec = joblib.load(str(vp))
            print('LOADED_VEC', vp)
            break
        except Exception as e:
            print('VEC_LOAD_ERROR', vp, e)

pwd='koyel'
print('PASSWORD',pwd)
X=None
if vec is not None:
    try:
        X=vec.transform([pwd])
        print('X_SHAPE', getattr(X,'shape',None))
    except Exception as e:
        print('VEC_TRANSFORM_ERROR', e)
else:
    X=[pwd]

try:
    out=model.predict(X)
    print('PREDICT_OUTPUT_TYPE', type(out))
    print('PREDICT_OUTPUT', out)
except Exception as e:
    print('PREDICT_ERROR', e)

try:
    if hasattr(model,'predict_proba'):
        proba=model.predict_proba(X)
        import numpy as np
        print('PREDICT_PROBA', np.asarray(proba).shape, proba)
    else:
        print('NO_PREDICT_PROBA')
except Exception as e:
    print('PREDICT_PROBA_ERROR', e)

for a in ('n_features_in_','classes_','feature_names_in_','vocabulary_'):
    print('ATTR',a, getattr(model,a,None))
print('MODEL_TYPE', type(model))
