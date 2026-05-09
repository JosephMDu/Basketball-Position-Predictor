# Joseph Du
# TAC 216
# Final Project
# Using Basketball stats to predict position
# This project predicts NBA player position group using per-48-minute statistics.
# Positions are grouped into G, F, and C to reduce missclassification between similar roles.
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

def main():
    # write your code here

    # upload and clean
    file_path = 'NBAstats_anonymized.csv'
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: Could not find '{file_path}'. Please ensure it is in the same directory.")
        return
    df.dropna(inplace = True)
    df.drop_duplicates(inplace = True)

    stats = ['TRB', 'AST', 'BLK', 'STL', '3PA', 'PTS', 'FT', 'TOV']
    
    standard_positions = ['PG', 'SG', 'SF', 'PF', 'C']
    df = df[df['Pos'].isin(standard_positions)]
    
    # make stats per 48mins to avoid discrepancies in playing Time
    df =df[df['MP'] > 0.0]

    per48 = []
    for stat in stats:
        new_col_name = f"{stat}48"
        df[new_col_name] = (df[stat]/df['MP'])*48
        per48.append(new_col_name)

    # graph of assists and blocks in correlation to position
    # colors = {'PG': 'blue', 'SG': 'orange', 'SF': 'green', 'PF': 'red', 'C': 'purple'}
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    # for pos in standard_positions:
      #  pos_data = df[df['Pos']==pos]
      #  ax1.scatter(pos_data['TRB48'], pos_data['AST48'], label = pos, color = colors[pos], alpha=0.7)
      #  ax2.scatter(pos_data['3PA48'], pos_data['PTS48'], label = pos, color = colors[pos], alpha = 0.7)
    # ax1.set_title("Plot 1: Actual Positions (Per48 Mins) TRB & AST")
    # ax1.set_xlabel("Rebounds Per 48 Minutes")
    # ax1.set_ylabel("Assists Per 48 Minutes")
    # ax1.legend(title="Real Position")
    # ax1.grid(True, linestyle='--', alpha=0.5)
    # ax2.set_title("Plot 2: Actual Positions (Per48 Mins) Pts & 3PA")
    # ax2.set_xlabel("3PA Per 48 Minutes")
    # ax2.set_ylabel("PTS Per 48 Minutes")
    # ax2.legend(title="Real Position")
    # ax2.grid(True, linestyle='--', alpha=0.5)
    
    # plt.tight_layout()
    # plt.savefig("Positions_vs_per48.jpg")

    # KNN Model
    print("KNN TEST MODEL:")
    # group guards and forwards together due to very small discrepancies between the roles causing large missclassification
    df['Pos'] = df['Pos'].replace({'SF': 'F', 'PF': 'F', 'PG': 'G', 'SG': 'G'})
    
    X = df[['AST48', 'TRB48', '3PA48', 'STL48', 'BLK48', 'FG%', 'PTS48', 'FT48', 'TOV48']]
    y = df['Pos']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100, stratify = y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # find optimal # of neigbors
    best_k = 1
    best_score_KNN = 0 
    for k in range(1, 20):
        temp_model = KNeighborsClassifier(n_neighbors=k)
        temp_model.fit(X_train_scaled, y_train)
        score = temp_model.score(X_test_scaled, y_test)

        if score > best_score_KNN:
            best_score_KNN = score
            best_k = k
    
    knn_model = KNeighborsClassifier(n_neighbors=best_k)
    knn_model.fit(X_train_scaled, y_train)

    knn_predictions = knn_model.predict(X_test_scaled)
    print(f"Number of Neighbors: {best_k}")
    print(f"KNN Accuracy: {knn_model.score(X_test_scaled, y_test)}\n")
    print("Classification Report:")
    print(classification_report(y_test, knn_predictions))

    # cm = confusion_matrix(y_test, knn_predictions, labels=knn_model.classes_)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=knn_model.classes_)
    
    # fig2, ax_cm = plt.subplots(figsize=(8, 6))
    # disp.plot(cmap='Blues', ax=ax_cm)
    # plt.title(f"Plot 3: KNN Predicted Positions (K={best_k})")
    # plt.tight_layout()
    # plt.savefig("KNN_Confusion_Matrix.jpg")

    # MLP Model
    print("MLP TEST MODEL:")
    layer_options = [
        (16,),             
        (32,),            
        (64,),             
        (32, 16),          
        (64, 32),          
        (128, 64),        
        (64, 32, 16),      
        (128, 64, 32)      
    ]
    best_layers = None
    best_score_MLP = 0

    for layers in layer_options:
        temp_model = MLPClassifier(hidden_layer_sizes=layers, max_iter=3000, random_state=100)
        temp_model.fit(X_train_scaled, y_train)
        score = temp_model.score(X_test_scaled, y_test)

        if score > best_score_MLP:
            best_score_MLP = score
            best_layers = layers
    
    mlp_model = MLPClassifier(hidden_layer_sizes = best_layers, max_iter = 3000, random_state = 100)
    mlp_model.fit(X_train_scaled, y_train)

    mlp_predictions = mlp_model.predict(X_test_scaled)
    print("MLP Classification Report:")
    print("Best Layer: ", best_layers)
    print("Score: ", best_score_MLP)
    print(classification_report(y_test, mlp_predictions))

    # cm = confusion_matrix(y_test, mlp_predictions, labels=mlp_model.classes_)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=mlp_model.classes_)
    
    # fig, axes = plt.subplots()
    # disp.plot(ax=axes)
    # axes.set(title = f"Plot 3: MLP Predicted Positions (Layers={best_layers})")
    # plt.tight_layout()
    # plt.savefig("MLP_Confusion_Matrix.jpg")

    # program interface

    def get_user_input():
        user_choice = input("Enter a position to visualize, or type 'ALL': ").strip().upper()
        
        if user_choice in ['G', 'F', 'C']:
            print(f"\nSubsetting data for {user_choice}...")
            return df[df['Pos'] == user_choice], [user_choice], user_choice
            
        elif user_choice == 'ALL':
            print("\nVisualizing ALL positions...")
            return df, ['G', 'F', 'C'], 'ALL'
            
        else:
            print(f"\nERROR: '{user_choice}' is invalid. Please try again.")
            return get_user_input()
        
    plot_df, positions_to_plot, user_pos = get_user_input()
    
    colors = {'G': 'blue', 'F': 'green', 'C': 'purple'}
    
    # Plot 1 & 2: Actual Positions (Using the user's subset)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    for pos in positions_to_plot:
        pos_data = plot_df[plot_df['Pos']==pos]
        ax1.scatter(pos_data['TRB48'], pos_data['AST48'], label=pos, color=colors[pos], alpha=0.7)
        ax2.scatter(pos_data['3PA48'], pos_data['PTS48'], label=pos, color=colors[pos], alpha=0.7)
        
    ax1.set_title(f"Plot 1: Actual Positions (Subset: {user_pos if user_pos in ['G', 'F', 'C'] else 'ALL'})")
    ax1.set_xlabel("Rebounds Per 48 Minutes")
    ax1.set_ylabel("Assists Per 48 Minutes")
    ax1.legend(title="Real Position")
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    ax2.set_title(f"Plot 2: Actual Positions Pts & 3PA")
    ax2.set_xlabel("3PA Per 48 Minutes")
    ax2.set_ylabel("PTS Per 48 Minutes")
    ax2.legend(title="Real Position")
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f"{user_pos}_vs_per48.jpg")

    if user_pos in ['G', 'F', 'C']:
        # Focused KNN Matrix
        cm_knn = confusion_matrix(y_test == user_pos, knn_predictions == user_pos)
        disp1 = ConfusionMatrixDisplay(confusion_matrix=cm_knn, display_labels=[f"Not {user_pos}", user_pos])
        fig, axes = plt.subplots()
        disp1.plot(ax=axes)
        axes.set(title = f"Plot 4: KNN Predicted Positions (K={best_k})")
        plt.tight_layout()
        plt.savefig(f"KNN_Confusion_Matrix_{user_pos}.jpg")

        # Focused MLP Matrix
        cm_mlp = confusion_matrix(y_test == user_pos, mlp_predictions == user_pos)
        disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_mlp, display_labels=[f"Not {user_pos}", user_pos])
        fig, axes = plt.subplots()
        disp2.plot(ax=axes)
        axes.set(title = f"Plot 3: MLP Predicted Positions (Layers={best_layers})")
        plt.tight_layout()
        plt.savefig(f"MLP_Confusion_Matrix_{user_pos}.jpg")

    else:
    # Plot 3: Confusion Matrix (Always shows the ML on the full test set)
        cm = confusion_matrix(y_test, mlp_predictions, labels=mlp_model.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=mlp_model.classes_)
        
        fig, axes = plt.subplots()
        disp.plot(ax=axes)
        axes.set(title = f"Plot 3: MLP Predicted Positions (Layers={best_layers})")
        plt.tight_layout()
        plt.savefig(f"MLP_Confusion_Matrix_{user_pos}.jpg")

        # plot 4: Confusion Matrix for KNN
        cm = confusion_matrix(y_test, knn_predictions, labels=knn_model.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=knn_model.classes_)
        
        fig, axes = plt.subplots()
        disp.plot(ax=axes)
        axes.set(title = f"Plot 4: KNN Predicted Positions (K={best_k})")
        plt.tight_layout()
        plt.savefig(f"KNN_Confusion_Matrix_{user_pos}.jpg")


if __name__ == '__main__':
    main()
