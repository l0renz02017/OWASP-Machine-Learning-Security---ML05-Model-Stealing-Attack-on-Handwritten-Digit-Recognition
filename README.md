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
Original Model	97.4%	-	-
Stolen Model (No Defense)	49.4%	100.0%	60.7%
Stolen Model (With Defenses)	17.4%	60.7%	27.7%
Defense Effectiveness: 64.8% reduction in stolen model accuracy

## 📖 Further Reading  
   -   [OWASP ML Top 10: ML05:2023 Model Theft](https://owasp.org/www-project-machine-learning-security-top-10/docs/ML05_2023-Model_Theft.html)  

⭐ If you find this lab helpful, please give it a star on GitHub!

## 🔗 Related Labs  

This demo is the fifth one in the OWASP ML Top 10 series, previous demos are here:  
-   [**ML01:2023 Input Manipulation Attack**](https://github.com/l0renz02017/OWASP-Machine-Learning-Security-ml01-input-manipulation-attack) - Fooling models during inference  
-   [**ML02:2023 Data Poisoning Attack**](https://github.com/l0renz02017/OWASP-Machine-Learning-Security-ml02-Data-Poisoning-Attack) - Corrupting models during training  
-   [**ML03:2023 Model Inversion Attack**](https://github.com/l0renz02017/OWASP-Machine-Learning-Security-ml03-Model-Inversion-Attack) - Stealing training data from models
-   [**ML04:2023 Membership Inference Attack**](https://github.com/l0renz02017/OWASP-Machine-Learning-Security-ml04-Membership-Inference-Attack) - Detecting Training Data Secrets


