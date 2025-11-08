# OWASP-Machine-Learning-Security---ML05-Model-Stealing-Attack-on-Handwritten-Digit-Recognition
🔥 Lab Overview This lab demonstrates Model Stealing Attacks (ML05) in the OWASP ML Security Top 10, where an adversary steals a proprietary handwritten digit recognition model by querying its prediction API and training a substitute model.

🎯 Lab Objectives
Understand how model stealing attacks work against ML-as-a-Service platforms

Learn to extract model functionality through strategic API queries

Implement and evaluate defense mechanisms against model extraction

Demonstrate the business impact of model intellectual property theft

⚡ Quick Demo
🏃‍♂️ How to Run
1. Open Google Colab: Go to Google Colab  
2. Create a New Notebook: Click File > New notebook  
3. Run the Code: Copy the entire code block from demo.py and paste it into a single cell in your Colab notebook  
4. Execute: Click the play (▶️) button or go to Runtime > Run all  

The code is completely self-contained and will install all necessary dependencies, train the digit recognition model, execute the stealing attack, and display the results.

📊 Attack Scenario
The Threat: A company deploys a proprietary handwritten digit recognition service that can identify digits 0-9 with 97.4% accuracy. An attacker discovers the public API and decides to steal the model instead of building their own.

The Attack:

Attacker generates 300 synthetic digit-like images

Queries the prediction API with each image

Records the model's predictions

Trains a stolen Logistic Regression model on the collected data

The Impact:

Without defenses: Attacker steals a functional model with 49.4% accuracy

With defenses: Attacker gets a useless model with only 17.4% accuracy

64.8% reduction in attack effectiveness

🛡️ Defense Mechanisms
This lab implements multiple defense strategies:

Rate Limiting: Detects and blocks suspicious query patterns

Output Noise: Adds random noise to predictions to obscure decision boundaries

Precision Reduction: Limits output decimal places to reduce information leakage

Hard Labels: Returns only top predictions without confidence scores

Response Normalization: Maintains valid probability distributions while protecting the model

📈 Results Summary
Scenario	Model Accuracy	Query Success	Agreement with Original
Original Model	97.4%	-	-
Stolen Model (No Defense)	49.4%	100.0%	60.7%
Stolen Model (With Defenses)	17.4%	60.7%	27.7%
Defense Effectiveness: 64.8% reduction in stolen model accuracy
