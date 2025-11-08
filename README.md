# 🔍 ML05:2023 Model Theft - Stealing an AI Model
🔥 Lab Overview This lab demonstrates Model Stealing Attacks (ML05) in the OWASP ML Security Top 10, where an adversary steals a proprietary handwritten digit recognition model by querying its prediction API and training a substitute model.

## 🚨 Why This Matters  
The Perfect Storm:  
📈 AI adoption exploding → More valuable targets  
🔓 APIs everywhere → Easy access points  
💰 Economic pressure → Strong incentives to steal  
🛠️ Tools improving → Easier to execute attacks  
⚖️ Legal uncertainty → Lower risk for attackers  

## 🎯 Lab Objectives
1. Understand how model stealing attacks work against ML-as-a-Service platforms  
2. Learn to extract model functionality through strategic API queries  
3. Implement and evaluate defense mechanisms against model extraction  
4. Demonstrate the business impact of model intellectual property theft  

## ⚡ Quick Demo
## 🏃‍♂️ How to Run
1. Open Google Colab: Go to Google Colab  
2. Create a New Notebook: Click File > New notebook  
3. Run the Code: Copy the entire code block from demo.py and paste it into a single cell in your Colab notebook  
4. Execute: Click the play (▶️) button or go to Runtime > Run all  

The code is completely self-contained and will install all necessary dependencies, train the digit recognition model, execute the stealing attack, and display the results.

## 📊 Attack Scenario
The Threat: A company deploys a proprietary handwritten digit recognition service that can identify digits 0-9 with 97.4% accuracy. An attacker discovers the public API and decides to steal the model instead of building their own.    

The Attack:  
1. Attacker generates 300 synthetic digit-like images  
2. Queries the prediction API with each image  
3. Records the model's predictions  
4. Trains a stolen Logistic Regression model on the collected data  

## 🔍 Attack Mechanics in Detail
What the Attacker Steals:
1. Decision Boundaries: By probing with many inputs, the attacker learns where the model draws boundaries between classes
2. Feature Importance: Which features matter most for predictions
3. Model Behavior: How the model responds to different inputs

Why This Works:
1. Black Box Access: Attacker only needs prediction API, no model internals
2. Input-Output Pairs: Each query gives (features → predicted_class) training data
3. Synthetic Data: Attacker doesn't need real training data - can generate it
4. Model Transfer: Simple models can often approximate complex ones given enough examples

## The Impact:

1. Without defenses: Attacker steals a functional model with 49.4% accuracy  
2. With defenses: Attacker gets a useless model with only 17.4% accuracy  
3. 64.8% reduction in attack effectiveness  

## 🛡️ Defense Mechanisms
This lab implements multiple defense strategies:  

1. Rate Limiting: Detects and blocks suspicious query patterns  
2. Output Noise: Adds random noise to predictions to obscure decision boundaries  
3. Precision Reduction: Limits output decimal places to reduce information leakage  
4. Hard Labels: Returns only top predictions without confidence scores  
5. Response Normalization: Maintains valid probability distributions while protecting the model

## 📈 Results Summary  
Scenario	Model Accuracy	Query Success	Agreement with Original  
Original Model	97.4%  
Stolen Model (No Defense)	49.4%  
Stolen Model (With Defenses)	17.4%  
Defense Effectiveness: 64.8% reduction in stolen model accuracy 
  
<img width="716" height="607" alt="image" src="https://github.com/user-attachments/assets/d6e94960-f61c-4da4-9e4b-eba0c605f420" />

## 📊 What You Will See  
🔬 OWASP ML05: Model Stealing Attack Demo - Handwritten Digits  
=============================================  
📊 Loading handwritten digits dataset...  
📁 Dataset: 1797 digits, 64 pixels each  
🔢 Digits: 0-9  
  
=============================================  
🔥 SCENARIO 1: Model Stealing - NO Defenses  
=============================================  

🏢 Training proprietary digit recognition model...  
✅ Model trained with 100.0% training accuracy  
🎯 Original Model Test Accuracy: 0.974  
  
👤 Attacker stealing digit recognition model...  
📊 Generating 300 synthetic digit queries  
🔍 Queried 100/300 synthetic digits...  
🔍 Queried 200/300 synthetic digits...  
🔄 Training stolen digit recognition model...  
✅ Stolen model trained successfully  
✅ Successful queries: 300/300  
❌ Failed queries: 0  
🎯 Stolen Model Accuracy: 0.494  
🤝 Agreement with Original: 0.504  
  
=============================================  
🛡️  SCENARIO 2: Model Stealing - WITH Defenses  
=============================================  

🏢 Training proprietary digit recognition model...  
✅ Model trained with 100.0% training accuracy  
  
👤 Attacker stealing digit recognition model...  
📊 Generating 300 synthetic digit queries  
🔍 Queried 100/300 synthetic digits...  
🚫 Defenses blocked some queries: Service temporarily unavailable  
🔍 Queried 200/300 synthetic digits...  
🔄 Training stolen digit recognition model...  
✅ Stolen model trained successfully  
✅ Successful queries: 182/300  
❌ Failed queries: 118  
🎯 Stolen Model Accuracy: 0.174  
🤝 Agreement with Original: 0.172  
  
=============================================  
📊 RESULTS SUMMARY  
=============================================  

  
## 📖 Further Reading  
   -   [OWASP ML Top 10: ML05:2023 Model Theft](https://owasp.org/www-project-machine-learning-security-top-10/docs/ML05_2023-Model_Theft.html)  

⭐ If you find this lab helpful, please give it a star on GitHub!

## 🔗 Related Labs  

This demo is the fifth one in the OWASP ML Top 10 series, previous demos are here:  
-   [**ML01:2023 Input Manipulation Attack**](https://github.com/l0renz02017/OWASP-Machine-Learning-Security-ml01-input-manipulation-attack) - Fooling models during inference  
-   [**ML02:2023 Data Poisoning Attack**](https://github.com/l0renz02017/OWASP-Machine-Learning-Security-ml02-Data-Poisoning-Attack) - Corrupting models during training  
-   [**ML03:2023 Model Inversion Attack**](https://github.com/l0renz02017/OWASP-Machine-Learning-Security-ml03-Model-Inversion-Attack) - Stealing training data from models
-   [**ML04:2023 Membership Inference Attack**](https://github.com/l0renz02017/OWASP-Machine-Learning-Security-ml04-Membership-Inference-Attack) - Detecting Training Data Secrets


