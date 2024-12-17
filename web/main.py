from flask import Flask, request, jsonify, render_template, send_from_directory, stream_with_context, Response
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.tree import plot_tree, _tree
import matplotlib.pyplot as plt
import os
import time

app = Flask(__name__)

# Pemetaan header untuk training dan test data
training_header_mapping = {
    'Wilayah': 'Kecamatan',
    'Latitude': 'Latitude',
    'Longitude': 'Longitude',
    'Curah Hujan (mm/hari)': 'Curah Hujan',
    'Suhu (°C)': 'Suhu',
    'Kelembaban (%)': 'Kelembaban',
    'Kecepatan Angin (km/jam)': 'Kecepatan Angin',
    'Tinggi Muka Air Sungai (m)': 'Tinggi Air Sungai',
    'Ketinggian Air Tanah (m)': 'Ketinggian Air Tanah',
    'Banjir Historis': 'Banjir Historis'
}

test_header_mapping = training_header_mapping  # Jika header sama, gunakan pemetaan yang sama

# Helper functions
def load_data(latih_path, uji_path):
    data_latih = pd.read_excel(latih_path)
    data_uji = pd.read_excel(uji_path)
    return data_latih, data_uji

def load_data_latih(latih_path):
    data_latih = pd.read_excel(latih_path)
    return data_latih
def load_data_uji(uji_path):
    data_uji = pd.read_excel(uji_path)
    return data_uji
def prepare_training_data(data_latih):
    X_train = data_latih.drop(columns=['Skenario', 'Label', 'Banjir Historis'])
    y_train = data_latih['Label']
    feature_names = X_train.columns.tolist()
    return X_train, y_train, feature_names

def train_random_forest(X_train, y_train):
    rf_model = RandomForestClassifier(n_estimators=200, random_state=50)
    rf_model.fit(X_train, y_train)
    return rf_model

def save_model(model, file_name):
    joblib.dump(model, file_name)

def save_decision_tree_as_png(tree, feature_names, class_names, file_name, max_depth=None):
    plt.figure(figsize=(20,10))
    plot_tree(
        tree,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        proportion=True,
        fontsize=6,  # Mengurangi ukuran font agar tidak bertumpuk
        max_depth=max_depth  # Menambahkan parameter max_depth untuk mengatur kedalaman tree
    )
    plt.tight_layout()  # Menambahkan tight_layout untuk mengatur tata letak agar tidak saling menimpa
    plt.savefig(file_name)
    plt.close()

def format_rule(path, class_names):
    rule = []
    indent = ""
    for i, p in enumerate(path):
        if p[0] == "leaf":
            rule.append(f"{indent}MAKA {class_names[p[1].argmax()]}")
        else:
            condition = f"{p[0]} {p[1]} {p[2]}" if not isinstance(p[2], bool) else f"{p[0]} IS {'TRUE' if p[2] else 'FALSE'}"
            if i > 0 and path[i-1][0] != "leaf":
                rule.append(f"{indent}DAN {condition}")
            else:
                rule.append(f"{indent}JIKA {condition}")
            
            # Check for the next condition to append "JIKA TIDAK MAKA"
            if i < len(path) - 1 and path[i + 1][0] != "leaf" and path[i + 1][1] == p[1] and path[i + 1][2] != p[2]:
                rule.append(f"{indent}JIKA TIDAK MAKA {path[i + 1][0]} {path[i + 1][1]} {path[i + 1][2]}")
                indent += "    "
                i += 1  # Skip the next condition as it has already been included in the rule
            else:
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

def generate_log():
    def log_stream():
        output_dir = 'decision_trees'
        if os.path.exists(output_dir):
            for file in os.listdir(output_dir):
                os.remove(os.path.join(output_dir, file))
            yield "data: Deleted existing decision trees and rules.\n\n"

        rf_model_path = 'rf_model.pkl'
        if not os.path.exists(rf_model_path):
            yield "data: Model file not found.\n\n"
            return

        rf_model = joblib.load(rf_model_path)
        feature_names = ['Curah Hujan', 'Suhu', 'Kelembaban', 'Kecepatan Angin', 'Tinggi Muka Air Sungai', 'Ketinggian Air Tanah']
        class_names = rf_model.classes_.astype(str).tolist()  # Convert to list

        os.makedirs(output_dir, exist_ok=True)

        yield "data: Starting extraction of decision trees...\n\n"
        for i in range(len(rf_model.estimators_)):
            file_name = os.path.join(output_dir, f'tree_{i + 1}.png')
            save_decision_tree_as_png(rf_model.estimators_[i], feature_names, class_names, file_name)
            yield f"data: Saved decision tree {i + 1} as {file_name}.\n\n"
            time.sleep(1)  # Simulate delay

        yield "data: Extracting rules...\n\n"
        all_rules = []

        for i in range(len(rf_model.estimators_)):
            rules = extract_rules(rf_model.estimators_[i], feature_names, class_names)
            all_rules.append((i + 1, rules))
            yield f"data: Extracted rules for tree {i + 1}.\n\n"
            time.sleep(1)  # Simulate delay

        rules_file_name = os.path.join(output_dir, 'all_rules.txt')
        with open(rules_file_name, 'w') as f:
            for tree_index, rules in all_rules:
                f.write(f"Rules for tree {tree_index}:\n")
                for rule in rules:
                    f.write(f"{rule}\n\n")

        yield "data: Saved all rules.\n\n"
    return Response(stream_with_context(log_stream()), content_type='text/event-stream')



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
        'Prediksi': y_pred
    })
    results.to_excel(file_name, index=False)
    return results

# Routes
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/view_training_data')
def view_training_data():
    latih_path = '../data_latih.xlsx'
    data_latih = load_data_latih(latih_path)
    data_html = data_latih.to_html(classes='table table-striped table-bordered', index=False)
    return render_template('view_data.html', data_table=data_html)

@app.route('/view_test_data')
def view_test_data():
    # Load data test
    uji_path = '../data_uji.xlsx'
    data_uji = load_data_uji(uji_path)

    # Rename kolom sesuai header_mapping dan filter kolom yang ditampilkan
    data_renamed = data_uji.rename(columns={v: k for k, v in test_header_mapping.items()})
    display_columns = list(test_header_mapping.keys())
    data_final = data_renamed[display_columns]

    # Konversi ke HTML
    data_html = data_final.to_html(classes='table table-striped table-bordered', index=False)
    return render_template('view_data.html', data_table=data_html)


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


@app.route('/extract_and_log', methods=['GET'])
def extract_and_log_route():
    return generate_log()

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
    if not os.path.exists(image_folder):
        return f"Error: Directory '{image_folder}' does not exist", 404
    
    images = sorted([img for img in os.listdir(image_folder) if img.startswith('tree_') and img.endswith('.png')])
    return render_template('view_trees.html', images=images)

@app.route('/decision_trees/<filename>')
def decision_tree_file(filename):
    return send_from_directory('decision_trees', filename)


@app.route('/predict', methods=['GET'])
def predict_route():
    uji_path = '../data_uji.xlsx'
    _, data_uji = load_data(uji_path, uji_path)
    X_test, kecamatan, latitude, longitude = prepare_testing_data(data_uji)
    rf_model = joblib.load('rf_model.pkl')
    y_pred = make_predictions(rf_model, X_test)
    prediction_results = save_prediction_results(kecamatan, y_pred, '../hasil_prediksi.xlsx')
    return jsonify(prediction_results.to_dict())

# Training
@app.route('/train', methods=['GET'])
def train_route():
    latih_path = '../data_latih.xlsx'
    model_file_path = 'rf_model.pkl'
    
    # Logging steps
    log = []
    log.append("Training started...")

    # Remove existing model file if it exists
    if os.path.exists(model_file_path):
        os.remove(model_file_path)
        log.append("Existing model file removed.")
    
    # Load data
    data_latih, _ = load_data(latih_path, latih_path)
    X_train, y_train, feature_names = prepare_training_data(data_latih)
    
    # Train model
    rf_model = train_random_forest(X_train, y_train)
    log.append("Model training completed.")
    
    # Save model
    save_model(rf_model, model_file_path)
    log.append("Model saved successfully.")
    
    # Evaluate model
    cv_scores, train_accuracy, train_report, conf_matrix = evaluate_model(rf_model, X_train, y_train)
    log.append("Model evaluation completed.")
    
    log.append(f"Cross-validation scores: {cv_scores}")
    log.append(f"Mean cross-validation score: {cv_scores.mean()}")
    log.append(f"Training accuracy: {train_accuracy}")
    log.append(f"Classification report:\n{train_report}")
    log.append(f"Confusion matrix:\n{conf_matrix}")
    
    return jsonify(log)

@app.route('/training', methods=['GET'])
def training_page():
    return render_template('training.html')



# Result



@app.route('/result', methods=['GET'])
def view_result():
    # Load data
    data_uji = pd.read_excel('../data_uji.xlsx')
    hasil_prediksi = pd.read_excel('../hasil_prediksi.xlsx')

    # Merge data uji dengan hasil prediksi
    data_merged = pd.merge(data_uji, hasil_prediksi, on='Kecamatan')

    # Hitung batas tampilan peta berdasarkan Latitude dan Longitude
    min_lat, max_lat = data_merged['Latitude'].min(), data_merged['Latitude'].max()
    min_lng, max_lng = data_merged['Longitude'].min(), data_merged['Longitude'].max()

    # Tambahkan tombol "View In Maps"
    data_merged['View In Maps'] = data_merged.apply(
        lambda row: f'<button class="view-in-maps btn btn-primary btn-sm" data-lat="{row.Latitude}" data-lng="{row.Longitude}">View In Maps</button>',
        axis=1
    )

    # Daftar header yang diinginkan
    header_labels = {
        'Kecamatan': 'Wilayah',
        'Latitude': 'Latitude',
        'Longitude': 'Longitude',
        'Curah Hujan': 'Curah Hujan (mm/hari)',
        'Suhu': 'Suhu (°C)',
        'Kelembaban': 'Kelembaban (%)',
        'Kecepatan Angin': 'Kecepatan Angin (km/jam)',
        'Tinggi Air Sungai': 'Tinggi Muka Air Sungai (m)',
        'Ketinggian Air Tanah': 'Ketinggian Air Tanah (m)',
        'Banjir Historis': 'Banjir Historis',
        'View In Maps': 'View In Maps'
    }

    # Konversi data ke tabel HTML dengan mengganti header
    data_table = data_merged.to_html(
        classes='table table-striped table-bordered',
        index=False, 
        escape=False, 
        table_id="dataTable",
        header=True
    )

    # Ganti header HTML secara manual sesuai pemetaan
    for original, new_label in header_labels.items():
        data_table = data_table.replace(f'<th>{original}</th>', f'<th>{new_label}</th>')

    # Konversi data ke format untuk JavaScript
    map_data = data_merged.to_dict(orient='records')

    # Render template HTML
    return render_template('view_result.html', 
                           map_data=map_data, 
                           min_latitude=min_lat, max_latitude=max_lat,
                           min_longitude=min_lng, max_longitude=max_lng, 
                           data_table=data_table)




if __name__ == '__main__':
    app.run(debug=True)
