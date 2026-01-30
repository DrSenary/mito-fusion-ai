# Project: MitoFusionAI - AI-Powered Drug Repurposing for Mitochondrial Dynamics Regulation

## Scientific Context
Parkinson's disease and other neurodegenerative disorders are characterized by impaired mitochondrial dynamics, particularly dysfunctional mitochondrial fission/fusion balance. DRP1 (Dynamin-Related Protein 1) mediates mitochondrial fission, and enhancing its activity may promote mitochondrial health in neurons. This project uses machine learning to identify FDA-approved drugs that may inhibit mitochondrial fusion, thereby indirectly promoting DRP1-mediated fission for potential neuroprotective effects.

## Project Goal
Develop a machine learning pipeline to predict mitochondrial fusion inhibitors from high-throughput screening data, then repurpose these predictions to identify FDA-approved drugs with potential neuroprotective properties through mitochondrial dynamics modulation.

## Dataset & Methods
- Primary Dataset: PubChem BioAssay AID 743254 - Mitochondrial Fusion Inhibitors Screen (~200,000 compounds)
- Target: Indirect DRP1 activation via fusion inhibition
- ML Approach: XGBoost with scaffold-split validation (AUC: 0.783)
- Fingerprints: Morgan fingerprints (1024 bits)
- Validation: Strict scaffold splitting ensuring generalization to novel chemical spaces

## Technical Pipeline
1. Data Processing: PubChem data cleaning, activity labeling (≥57.94% inhibition = active)
2. Model Training: XGBoost with Morgan fingerprints, scaffold-split validation
3. Drug Repurposing: Prediction on FDA-approved drugs from DrugBank
4. Candidate Prioritization: Rank by predicted probability, filter for CNS-penetrant compounds

## Key Results
- Model Performance: 0.783 AUC on scaffold-split test set
- Top Candidates: Fluphenazine, prochlorperazine and other FDA drugs with predicted fusion inhibition
- Biological Relevance: Several top predictions have known mitochondrial effects

## Biological Significance
This work bridges computational drug discovery with mitochondrial biology, offering:
1. Novel repurposing candidates for neurodegenerative diseases
2. Insights into chemical features promoting mitochondrial fission
3. Framework for similar target-agnostic phenotypic screening repurposing

## Applications
- Drug Repurposing: Immediate candidates for experimental validation
- Chemical Biology: Understanding structure-activity relationships for fusion inhibition
- Therapeutic Development: Potential leads for mitochondrial-targeted neuroprotectives

## Technologies Used
- Python (RDKit, XGBoost, scikit-learn, pandas)
- Cheminformatics (Morgan fingerprints, scaffold splitting)
- Machine Learning (XGBoost with class imbalance handling)
- Data from PubChem, DrugBank

## Contributing
This project is part of Chemical biology research in mitochondrial-targeted neurotherapeutics. Suggestions and collaborations welcome!

## License
Academic/Research Use

