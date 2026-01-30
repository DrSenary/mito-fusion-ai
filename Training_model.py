import warnings
warnings.filterwarnings('ignore')
from rdkit import rdBase
rdBase.DisableLog('rdApp.warning')
# train_scaffold_model.py (SIMPLIFIED VERSION)
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import joblib
from collections import defaultdict

print("=== 1. LOAD CLEANED DATA ===")
df = pd.read_csv('clean_dataset_simple.csv')

print("=== 2. SCAFFOLD SPLIT ===")
# Get scaffolds
def get_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold)

df['scaffold'] = df['smiles'].apply(get_scaffold)
df = df.dropna(subset=['scaffold'])

# Group by scaffold
scaffold_to_indices = defaultdict(list)
for idx, scaffold in enumerate(df['scaffold']):
    scaffold_to_indices[scaffold].append(idx)

scaffold_groups = list(scaffold_to_indices.values())
np.random.shuffle(scaffold_groups)

# Split 80/20
train_idx, test_idx = [], []
for group in scaffold_groups:
    if len(test_idx) < len(df) * 0.2:
        test_idx.extend(group)
    else:
        train_idx.extend(group)

print(f"Train: {len(train_idx)}, Test: {len(test_idx)}")

print("=== 3. CONVERT TO FINGERPRINTS ===")
def smiles_to_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024))

X = pd.DataFrame([smiles_to_fp(s) for s in df['smiles']])
y = df['active'].values

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

print("=== 4. TRAIN XGBOOST ===")
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]),
    random_state=42
)
model.fit(X_train, y_train)

print("=== 5. EVALUATE ===")
y_pred = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred)
print(f"Scaffold-split AUC: {auc:.3f}")

print("=== 6. SAVE MODEL ===")
joblib.dump(model, 'phase1_model.pkl')
print("Model saved as 'phase1_model.pkl'")
