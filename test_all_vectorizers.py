import joblib
from pathlib import Path
import numpy as np

# Test passwords with different expected strengths
test_passwords = [
    "weak",           # Should be Weak
    "koyel",          # Should be Weak
    "Password1!",     # Should be Medium or Strong
    "MyP@ssw0rd123!", # Should be Strong
    "Tr0ub4dor&3",    # Should be Strong
]

model_path = Path("/workspaces/Checking-Password-Strength-using-Machine-Learning/password-ml-app/models/enhanced_password_model_1.pkl")
model = joblib.load(str(model_path))

print(f"Model type: {type(model)}")
print(f"Model n_features_in_: {model.n_features_in_}")
print(f"Model classes_: {model.classes_}")
print()

# Find all vectorizers
vectorizer_paths = []
repo_root = Path("/workspaces/Checking-Password-Strength-using-Machine-Learning")
vectorizer_paths.extend(repo_root.glob("tfidf_vectorizer*.pkl"))
vectorizer_paths.extend((repo_root / "password-ml-app/models").glob("*vectorizer*.pkl"))

label_map = {0: "Weak", 1: "Medium", 2: "Strong"}

for vec_path in vectorizer_paths:
    print(f"\n{'='*80}")
    print(f"Testing vectorizer: {vec_path.name}")
    print(f"Path: {vec_path}")
    
    try:
        vectorizer = joblib.load(str(vec_path))
        
        # Get feature count
        try:
            n_features = len(vectorizer.vocabulary_)
        except:
            try:
                n_features = len(vectorizer.get_feature_names_out())
            except:
                n_features = vectorizer.transform(["x"]).shape[1]
        
        print(f"Vectorizer n_features: {n_features}")
        
        # Test predictions
        predictions = []
        for pwd in test_passwords:
            X = vectorizer.transform([pwd])
            
            # Reshape if needed
            if X.shape[1] != model.n_features_in_:
                from scipy.sparse import hstack, csr_matrix
                diff = model.n_features_in_ - X.shape[1]
                if diff > 0:
                    # Pad
                    padding = csr_matrix((X.shape[0], diff))
                    X = hstack([X, padding])
                else:
                    # Truncate
                    X = X[:, :model.n_features_in_]
            
            pred = model.predict(X)[0]
            pred_label = label_map.get(int(pred), str(pred))
            predictions.append(pred_label)
            print(f"  {pwd:20s} -> {pred_label}")
        
        # Check if all predictions are the same
        unique_preds = set(predictions)
        if len(unique_preds) == 1:
            print(f"  ⚠️  WARNING: All predictions are '{predictions[0]}' - likely wrong vectorizer")
        else:
            print(f"  ✓ GOOD: Multiple predictions ({unique_preds}) - this vectorizer may be correct!")
            
    except Exception as e:
        print(f"  ERROR: {e}")

print(f"\n{'='*80}")
print("DONE")
