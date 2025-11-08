# -*- coding: utf-8 -*-
"""
OWASP Machine Learning Security - ML05: Model Stealing Attack Demo
Handwritten Digit Recognition Version
Complete self-contained demo that runs in Google Colab
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

print("🔬 OWASP ML05: Model Stealing Attack Demo - Handwritten Digits")
print("=" * 60)

class DigitModel:
    """Proprietary handwritten digit recognition model"""
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=50, random_state=42)
        self.is_trained = False
        self.query_counts = {}
        
    def train(self, X, y):
        """Train the digit recognition model"""
        print("🏢 Training proprietary digit recognition model...")
        self.model.fit(X, y)
        self.is_trained = True
        accuracy = self.model.score(X, y)
        print(f"✅ Model trained with {accuracy:.1%} training accuracy")
        
    def predict_digit(self, digit_image, defense_mode=False, client_id="default"):
        """Predict digit from image (8x8 pixels)"""
        if not self.is_trained:
            raise Exception("Model not trained")
            
        # Initialize client tracking
        if client_id not in self.query_counts:
            self.query_counts[client_id] = 0
        self.query_counts[client_id] += 1
        
        # Reshape for prediction
        if len(digit_image.shape) > 1:
            digit_image = digit_image.flatten()
        digit_image = digit_image.reshape(1, -1)
        
        # Get original prediction
        original_prediction = self.model.predict_proba(digit_image)[0]
        predicted_digit = np.argmax(original_prediction)
        
        if not defense_mode:
            return original_prediction, predicted_digit
        
        # =============================================================
        # DEFENSE MECHANISMS AGAINST MODEL STEALING
        # =============================================================
        
        # Defense 1: Rate limiting
        if self.query_counts[client_id] > 150:
            if np.random.random() < 0.8:  # 80% block chance after limit
                raise Exception("Service temporarily unavailable")
        
        # Defense 2: Add noise to probabilities
        noise = np.random.normal(0, 0.2, original_prediction.shape)
        defended_prediction = np.clip(original_prediction + noise, 0.01, 0.99)
        
        # Defense 3: Reduce precision
        defended_prediction = np.round(defended_prediction, 1)
        
        # Defense 4: Sometimes return only top prediction
        if np.random.random() < 0.4:
            hard_prediction = np.zeros_like(defended_prediction)
            hard_prediction[predicted_digit] = 1.0
            defended_prediction = hard_prediction
        
        # Normalize
        defended_prediction = defended_prediction / defended_prediction.sum()
        defended_digit = np.argmax(defended_prediction)
        
        return defended_prediction, defended_digit

class DigitStealingAttack:
    """Adversary trying to steal the digit recognition model"""
    def __init__(self, target_model):
        self.target_model = target_model
        self.stolen_model = LogisticRegression(max_iter=1000, random_state=42)
        self.stolen_images = []
        self.stolen_labels = []
        self.failed_queries = 0
        
    def generate_synthetic_digit(self):
        """Generate synthetic digit-like images"""
        # Create random patterns that resemble digits
        synthetic_digit = np.random.uniform(0, 1, 64)  # 8x8 flattened
        
        # Add some structure to make it more digit-like
        if np.random.random() < 0.7:
            # Simulate strokes by having correlated pixels
            center_x, center_y = np.random.randint(2, 6, 2)
            for i in range(8):
                for j in range(8):
                    distance = np.sqrt((i - center_x)**2 + (j - center_y)**2)
                    if distance < 3:
                        synthetic_digit[i*8 + j] += np.random.uniform(0.3, 0.7)
        
        return np.clip(synthetic_digit, 0, 1)
    
    def execute_attack(self, num_queries=300, defense_mode=False, client_id="attacker"):
        """Execute model stealing attack"""
        print(f"\n👤 Attacker stealing digit recognition model...")
        print(f"📊 Generating {num_queries} synthetic digit queries")
        
        self.stolen_images = []
        self.stolen_labels = []
        self.failed_queries = 0
        
        for i in range(num_queries):
            if i % 100 == 0 and i > 0:
                print(f"🔍 Queried {i}/{num_queries} synthetic digits...")
                
            synthetic_digit = self.generate_synthetic_digit()
            
            try:
                # Query the target model
                probabilities, predicted_digit = self.target_model.predict_digit(
                    synthetic_digit, defense_mode, client_id
                )
                
                # Store stolen knowledge
                self.stolen_images.append(synthetic_digit)
                self.stolen_labels.append(predicted_digit)
                
            except Exception as e:
                self.failed_queries += 1
                if "unavailable" in str(e) and self.failed_queries == 1:
                    print(f"🚫 Defenses blocked some queries: {e}")
        
        # Train stolen model
        if len(self.stolen_images) > 50:
            print("🔄 Training stolen digit recognition model...")
            X_stolen = np.array(self.stolen_images)
            y_stolen = np.array(self.stolen_labels)
            self.stolen_model.fit(X_stolen, y_stolen)
            print("✅ Stolen model trained successfully")
            
        return len(self.stolen_images)
    
    def evaluate_stealing(self, X_test, y_test):
        """Evaluate stolen model performance"""
        if len(self.stolen_images) < 20:
            return 0, 0
            
        # Original model predictions
        y_pred_original = self.target_model.model.predict(X_test)
        acc_original = accuracy_score(y_test, y_pred_original)
        
        # Stolen model predictions
        y_pred_stolen = self.stolen_model.predict(X_test)
        acc_stolen = accuracy_score(y_test, y_pred_stolen)
        
        # Agreement between models
        agreement = np.mean(y_pred_original == y_pred_stolen)
        
        return acc_stolen, agreement

def plot_digit_comparison(original_model, stolen_model_no_defense, stolen_model_defended, X_test, y_test):
    """Visual comparison of digit recognition performance"""
    # Sample some test digits
    sample_indices = np.random.choice(len(X_test), 12, replace=False)
    
    fig, axes = plt.subplots(3, 4, figsize=(15, 10))
    
    for i, idx in enumerate(sample_indices):
        digit_image = X_test[idx]
        true_label = y_test[idx]
        
        # Get predictions from all models
        pred_original = original_model.predict([digit_image])[0]
        pred_stolen_no_def = stolen_model_no_defense.predict([digit_image])[0]
        pred_stolen_def = stolen_model_defended.predict([digit_image])[0]
        
        # Plot digit
        ax = axes[i // 4, i % 4]
        ax.imshow(digit_image.reshape(8, 8), cmap='gray')
        ax.set_title(f'True: {true_label}\n'
                    f'Original: {pred_original} | '
                    f'Stolen: {pred_stolen_no_def} | '
                    f'Stolen+Def: {pred_stolen_def}', 
                    fontsize=9)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    print("📊 Loading handwritten digits dataset...")
    digits = load_digits()
    X, y = digits.data, digits.target
    
    # Normalize pixel values
    X = X / 16.0  # Max value is 16 in this dataset
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print(f"📁 Dataset: {X.shape[0]} digits, {X.shape[1]} pixels each")
    print(f"🔢 Digits: 0-9")
    
    # =========================================================================
    # SCENARIO 1: No defenses
    # =========================================================================
    print("\n" + "="*60)
    print("🔥 SCENARIO 1: Model Stealing - NO Defenses")
    print("="*60)
    
    original_model = DigitModel()
    original_model.train(X_train, y_train)
    
    # Test original model
    original_accuracy = original_model.model.score(X_test, y_test)
    print(f"🎯 Original Model Test Accuracy: {original_accuracy:.3f}")
    
    attacker_no_defense = DigitStealingAttack(original_model)
    successful_queries_no_def = attacker_no_defense.execute_attack(
        num_queries=300, defense_mode=False, client_id="attacker1"
    )
    
    print(f"✅ Successful queries: {successful_queries_no_def}/300")
    print(f"❌ Failed queries: {attacker_no_defense.failed_queries}")
    
    if successful_queries_no_def > 50:
        stolen_acc_no_def, agreement_no_def = attacker_no_defense.evaluate_stealing(X_test, y_test)
        print(f"🎯 Stolen Model Accuracy: {stolen_acc_no_def:.3f}")
        print(f"🤝 Agreement with Original: {agreement_no_def:.3f}")
    else:
        stolen_acc_no_def, agreement_no_def = 0, 0
        print("💥 Attack failed - not enough data")
    
    # =========================================================================
    # SCENARIO 2: With defenses
    # =========================================================================
    print("\n" + "="*60)
    print("🛡️  SCENARIO 2: Model Stealing - WITH Defenses")
    print("="*60)
    
    defended_model = DigitModel()
    defended_model.train(X_train, y_train)
    
    attacker_defended = DigitStealingAttack(defended_model)
    successful_queries_def = attacker_defended.execute_attack(
        num_queries=300, defense_mode=True, client_id="attacker2"
    )
    
    print(f"✅ Successful queries: {successful_queries_def}/300")
    print(f"❌ Failed queries: {attacker_defended.failed_queries}")
    
    if successful_queries_def > 50:
        stolen_acc_def, agreement_def = attacker_defended.evaluate_stealing(X_test, y_test)
        print(f"🎯 Stolen Model Accuracy: {stolen_acc_def:.3f}")
        print(f"🤝 Agreement with Original: {agreement_def:.3f}")
    else:
        stolen_acc_def, agreement_def = 0, 0
        print("💥 Attack largely blocked by defenses!")
    
    # =========================================================================
    # VISUALIZATION
    # =========================================================================
    print("\n" + "="*60)
    print("📊 RESULTS SUMMARY")
    print("="*60)
    
    # Accuracy comparison
    scenarios = ['Original Model', 'Stolen (No Defense)', 'Stolen (With Defense)']
    accuracies = [original_accuracy, stolen_acc_no_def, stolen_acc_def]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Accuracy plot
    colors = ['green', 'red', 'orange']
    bars1 = ax1.bar(scenarios, accuracies, color=colors, alpha=0.8)
    ax1.set_ylabel('Digit Recognition Accuracy', fontweight='bold')
    ax1.set_title('Model Stealing Impact on Digit Recognition', fontweight='bold', fontsize=12)
    ax1.set_ylim(0, 1)
    
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Query success plot
    query_success = [
        (successful_queries_no_def/300)*100,
        (successful_queries_def/300)*100
    ]
    
    bars2 = ax2.bar(['No Defense', 'With Defenses'], query_success, color=['red', 'orange'], alpha=0.8)
    ax2.set_ylabel('Query Success Rate (%)', fontweight='bold')
    ax2.set_title('Attack Difficulty: Successful Queries', fontweight='bold', fontsize=12)
    ax2.set_ylim(0, 100)
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # =========================================================================
    # KEY INSIGHTS
    # =========================================================================
    print("\n💡 KEY INSIGHTS:")
    print(f"• Original digit recognition accuracy: {original_accuracy:.3f}")
    print(f"• Stolen model accuracy (no defense): {stolen_acc_no_def:.3f}")
    print(f"• Stolen model accuracy (with defense): {stolen_acc_def:.3f}")
    
    if stolen_acc_no_def > 0:
        effectiveness_reduction = ((stolen_acc_no_def - stolen_acc_def) / stolen_acc_no_def * 100)
        print(f"• Defense effectiveness: {effectiveness_reduction:.1f}% reduction")
    
    print(f"• Queries successful (no defense): {successful_queries_no_def}/300 ({successful_queries_no_def/300*100:.1f}%)")
    print(f"• Queries successful (with defense): {successful_queries_def}/300 ({successful_queries_def/300*100:.1f}%)")
    
    print("\n🛡️  DEFENSE EFFECTIVENESS:")
    if stolen_acc_def < stolen_acc_no_def * 0.6:
        print("✅ Defenses are HIGHLY effective!")
        print("   - Noise and precision reduction protect model knowledge")
        print("   - Rate limiting prevents mass data collection")
        print("   - Hard labels confuse the stealing process")
    elif stolen_acc_def < stolen_acc_no_def * 0.8:
        print("✅ Defenses are effective")
    else:
        print("⚠️  Defenses need improvement")
    
    # Show sample digit predictions
    if successful_queries_no_def > 50 and successful_queries_def > 50:
        print(f"\n🎯 SAMPLE DIGIT PREDICTIONS:")
        plot_digit_comparison(
            original_model.model, 
            attacker_no_defense.stolen_model, 
            attacker_defended.stolen_model,
            X_test, 
            y_test
        )

if __name__ == "__main__":
    main()
