from flask import Flask, request, jsonify, render_template, send_from_directory
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree, _tree
import os

app = Flask(__name__)

# Helper functions
def load_data(latih_path, uji_path):
    data_latih = pd.read_excel(latih_path)
    data_uji = pd.read_excel(latih_path)
    return data_latih, data_uji

def prepare_training_data(data_latih):
    X_train = data_latih.drop(columns=['Skenario', 'Label', 'Banjir Historis'])
    y_train = data_latih['Label']
    feature_names = X_train.columns.tolist()
    return X_train, y_train, feature_names

def train_random_forest(X_train, y_train):
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    return rf_model

def save_model(model, file_name):
    joblib.dump(model, file_name)

def save_decision_tree_as_png(tree, feature_names, class_names, file_name):
    plt.figure(figsize=(20,10))
    plot_tree(
        tree,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        proportion=True,
        fontsize=10
    )
    plt.savefig(file_name)
    plt.close()

def format_rule(path, class_names):
    """
    Format a single path into a readable rule string.
    """
    rule = []
    indent = ""
    for p in path:
        if p[0] == "leaf":
            rule.append(f"{indent}THEN {class_names[p[1].argmax()]}")
        else:
            value = p[2] if isinstance(p[2], str) else round(p[2], 4)
            rule.append(f"{indent}IF {p[0]} {p[1]} {value}")
            indent += "    "
    return "\n".join(rule)

def extract_rules(tree, feature_names, class_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    paths = []
    path = []

    def recurse(node, depth):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = round(tree_.threshold[node], 4)
            path.append((name, "<=", threshold))
            recurse(tree_.children_left[node], depth + 1)
            path.pop()
            path.append((name, ">", threshold))
            recurse(tree_.children_right[node], depth + 1)
            path.pop()
        else:
            path.append(("leaf", tree_.value[node]))
            paths.append(list(path))
            path.pop()

    recurse(0, 1)
    rules = [format_rule(p, class_names) for p in paths]
    return rules

def save_rules(rules, file_name):
    with open(file_name, 'w') as f:
        for path in rules:
            f.write("RULE:\n")
            for p in path:
                if len(p) == 3:
                    f.write(f"  {p[0]} {p[1]} {p[2]}\n")
                else:
                    f.write(f"  {p[0]}: {p[1]}\n")
            f.write("\n")

def evaluate_model(model, X_train, y_train):
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_pred)
    train_report = classification_report(y_train, train_pred)
    conf_matrix = confusion_matrix(y_train, train_pred)
    return cv_scores, train_accuracy, train_report, conf_matrix

def prepare_testing_data(data_uji):
    X_test = data_uji.drop(columns=['Kecamatan', 'Latitude', 'Longitude', 'Banjir Historis'])
    kecamatan = data_uji['Kecamatan']
    latitude = data_uji['Latitude']
    longitude = data_uji['Longitude']
    return X_test, kecamatan, latitude, longitude

def make_predictions(model, X_test):
    return model.predict(X_test)

def save_prediction_results(kecamatan, y_pred, file_name):
    results = pd.DataFrame({
        'Kecamatan': kecamatan,
        'Prediksi Banjir': y_pred
    })
    results.to_excel(file_name, index=False)
    return results

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/load_data', methods=['GET'])
def load_data_route():
    latih_path = 'data/data_latih.xlsx'
    uji_path = 'data/data_uji.xlsx'
    data_latih, data_uji = load_data(latih_path, uji_path)
    return jsonify({
        'data_latih_head': data_latih.head().to_dict(),
        'data_uji_head': data_uji.head().to_dict()
    })

@app.route('/train', methods=['GET'])
def train_route():
    latih_path = 'data/data_latih.xlsx'
    data_latih, _ = load_data(latih_path, latih_path)
    X_train, y_train, feature_names = prepare_training_data(data_latih)
    rf_model = train_random_forest(X_train, y_train)
    save_model(rf_model, 'rf_model.pkl')
    return "Model trained and saved!"

@app.route('/save_decision_trees', methods=['GET'])
def save_decision_trees_route():
    rf_model = joblib.load('rf_model.pkl')
    feature_names = ['Curah Hujan', 'Suhu', 'Kelembaban', 'Kelembaban', 'Tinggi Muka Air Sungai', 'Ketinggian Air Tanah']
    class_names = rf_model.classes_.astype(str).tolist()  # Convert to list
    output_dir = 'decision_trees'
    os.makedirs(output_dir, exist_ok=True)

    for i in range(len(rf_model.estimators_)):
        file_name = os.path.join(output_dir, f'tree_{i + 1}.png')
        save_decision_tree_as_png(rf_model.estimators_[i], feature_names, class_names, file_name)

    return "Decision trees saved!"


@app.route('/save_rules', methods=['GET'])
def save_rules_route():
    rf_model = joblib.load('rf_model.pkl')
    feature_names = ['Curah Hujan', 'Suhu', 'Kelembaban', 'Kecepatan Angin', 'Tinggi Muka Air Sungai', 'Ketinggian Air Tanah']
    class_names = rf_model.classes_.astype(str).tolist()  # Convert to list
    output_dir = 'decision_trees'
    os.makedirs(output_dir, exist_ok=True)

    all_rules = []

    for i in range(len(rf_model.estimators_)):
        rules = extract_rules(rf_model.estimators_[i], feature_names, class_names)
        all_rules.append((i + 1, rules))

    # Save all rules into a single file
    rules_file_name = os.path.join(output_dir, 'all_rules.txt')
    with open(rules_file_name, 'w') as f:
        for tree_index, rules in all_rules:
            f.write(f"Rules for tree {tree_index}:\n")
            for rule in rules:
                f.write(f"{rule}\n\n")

    return "Rules saved!"

@app.route('/view_rules', methods=['GET'])
def view_rules_route():
    rules_file_name = 'decision_trees/all_rules.txt'
    with open(rules_file_name, 'r') as f:
        rules_content = f.read()
    return render_template('view_rules.html', rules=rules_content)

@app.route('/view_trees', methods=['GET'])
def view_trees_route():
    image_folder = 'decision_trees'
    # Ensure the directory exists
    if not os.path.exists(image_folder):
        return f"Error: Directory '{image_folder}' does not exist", 404
    
    images = sorted([img for img in os.listdir(image_folder) if img.startswith('tree_') and img.endswith('.png')])
    return render_template('view_trees.html', images=images)

@app.route('/decision_trees/<filename>')
def decision_tree_file(filename):
    return send_from_directory('decision_trees', filename)

@app.route('/evaluate', methods=['GET'])
def evaluate_route():
    latih_path = 'data/data_latih.xlsx'
    data_latih, _ = load_data(latih_path, latih_path)
    X_train, y_train, feature_names = prepare_training_data(data_latih)
    rf_model = joblib.load('rf_model.pkl')
    cv_scores, train_accuracy, train_report, conf_matrix = evaluate_model(rf_model, X_train, y_train)
    return jsonify({
        'cv_scores': cv_scores.tolist(),
        'mean_cv_score': cv_scores.mean(),
        'train_accuracy': train_accuracy,
        'train_report': train_report,
        'conf_matrix': conf_matrix.tolist()
    })

@app.route('/predict', methods=['GET'])
def predict_route():
    uji_path = 'data/data_uji.xlsx'
    _, data_uji = load_data(uji_path, uji_path)
    X_test, kecamatan, latitude, longitude = prepare_testing_data(data_uji)
    rf_model = joblib.load('rf_model.pkl')
    y_pred = make_predictions(rf_model, X_test)
    prediction_results = save_prediction_results(kecamatan, y_pred, 'prediction_results.xlsx')
    return jsonify(prediction_results.head().to_dict())

if __name__ == '__main__':
    app.run(debug=True)
