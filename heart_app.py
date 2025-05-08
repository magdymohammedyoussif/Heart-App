import streamlit as st
import pandas as pd
import numpy as np

import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import os
import logging
import plotly.express as px

PAGE_CONFIG = {"page_title": "Heart App", "page_icon": "❤️", "layout": "wide"}
DEFAULT_DATA_FILE = 'heart.csv'
MODEL_ARTIFACTS_DIR = "model_artifacts"
NUMERICAL_COLS_FOR_OUTLIERS = ['RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
CATEGORICAL_COLS = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
TARGET_COL = 'HeartDisease'

SKEY_DF_ORIGINAL = 'df_original'
SKEY_ACTIVE_DATA_SOURCE_NAME = 'active_data_source_name'
SKEY_TRAINED = 'trained'
SKEY_MODEL_RESULTS = 'model_results'
SKEY_DF_OUTLIER_REMOVED_DATA = 'df_after_outlier_removal_data'
SKEY_ENCODERS = 'encoders'
SKEY_FEATURE_NAMES = 'feature_names'
SKEY_X_TEST_SCALED = 'X_test_scaled'
SKEY_Y_TEST = 'y_test'
SKEY_TRAINED_MODELS_LIST = 'trained_models_list'

os.makedirs(MODEL_ARTIFACTS_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@st.cache_data
def load_data(file_input):
    try:
        df = pd.read_csv(file_input)
        logging.info(f"Data loaded successfully from {file_input if isinstance(file_input, str) else file_input.name}")
        return df
    except FileNotFoundError:
        st.error(f"Error: File '{file_input}' not found. Please ensure it's present or upload a new one.")
        logging.error(f"File not found: {file_input}")
        return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        logging.error(f"Error loading data from {file_input}: {e}", exc_info=True)
        return None

def remove_outliers_IQR(dataf, columns):
    if dataf is None: return None, 0, 0
    _df = dataf.copy()
    rows_before = len(_df)
    for col in columns:
        if col in _df.columns and pd.api.types.is_numeric_dtype(_df[col]):
            Q1 = _df[col].quantile(0.25)
            Q3 = _df[col].quantile(0.75)
            IQR = Q3 - Q1
            if IQR > 0:
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                _df = _df[(_df[col] >= lower_bound) & (_df[col] <= upper_bound)]
    rows_after = len(_df)
    removed_count = rows_before - rows_after
    if rows_before > 0:
        removed_percent = (removed_count / rows_before) * 100
        logging.info(f"Outlier removal: {removed_count} rows ({removed_percent:.2f}%) removed.")
    else:
        logging.info("Outlier removal: No rows to process.")
    return _df, removed_count, rows_before

@st.cache_data
def preprocess_data(_df, categorical_cols_to_encode, target_col):
    if _df is None: return None, {}
    df = _df.copy()
    encoders = {}
    for col in categorical_cols_to_encode:
        if col in df.columns:
            le = LabelEncoder()

            if df[col].dtype == 'object' or pd.api.types.is_categorical_dtype(df[col]):

                if df[col].isnull().any():
                    st.warning(f"Column '{col}' contains NaNs. Filling with a placeholder 'Missing' before LabelEncoding.")
                    df[col] = df[col].fillna('Missing') 
                df[col] = le.fit_transform(df[col])
                encoders[col] = le
            else: logging.info(f"Column '{col}' is already numeric or not suitable for LabelEncoding. Skipping.")
        else:
            st.warning(f"Categorical column '{col}' not found. Skipping encoding.")
            logging.warning(f"Categorical column '{col}' not found for encoding.")
    if encoders:
        try:
            joblib.dump(encoders, os.path.join(MODEL_ARTIFACTS_DIR, 'encoders.joblib'))
            logging.info("Encoders saved.")
        except Exception as e:
            st.error(f"Error saving encoders: {e}")
            logging.error(f"Error saving encoders: {e}", exc_info=True)
    return df, encoders

def get_models():
    return {
        "Logistic Regression": LogisticRegression(random_state=44, max_iter=1000, solver='liblinear'),
        "K-NN": KNeighborsClassifier(n_neighbors=5),
        "SVM": SVC(kernel='linear', random_state=44, probability=True),
        "Kernel SVM": SVC(kernel='rbf', random_state=44, probability=True),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(criterion='entropy', random_state=44),
        "Random Forest": RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=44)
    }

def save_training_artifacts(scaler, feature_names):
    try:
        joblib.dump(scaler, os.path.join(MODEL_ARTIFACTS_DIR, 'scaler.joblib'))
        joblib.dump(feature_names, os.path.join(MODEL_ARTIFACTS_DIR, 'feature_names.joblib'))
        logging.info("Scaler and feature names saved.")
    except Exception as e:
        st.error(f"Error saving training artifacts: {e}")
        logging.error(f"Error saving training artifacts: {e}", exc_info=True)

def load_prediction_artifacts():
    try:
        encoders_path = os.path.join(MODEL_ARTIFACTS_DIR, 'encoders.joblib')
        scaler_path = os.path.join(MODEL_ARTIFACTS_DIR, 'scaler.joblib')
        feature_names_path = os.path.join(MODEL_ARTIFACTS_DIR, 'feature_names.joblib')

        encoders = joblib.load(encoders_path) if os.path.exists(encoders_path) else {}
        scaler = joblib.load(scaler_path)
        feature_names = joblib.load(feature_names_path)
        logging.info("Prediction artifacts loaded.")
        return encoders, scaler, feature_names
    except FileNotFoundError as e:
        st.error(f"Essential model artifact not found: {e}. Train models first.")
        logging.warning(f"Prediction artifact not found: {e}")
        return None, None, None
    except Exception as e:
        st.error(f"Error loading prediction artifacts: {e}")
        logging.error(f"Error loading prediction artifacts: {e}", exc_info=True)
        return None, None, None

def load_model(model_name):
    try:
        model_path = os.path.join(MODEL_ARTIFACTS_DIR, f'{model_name.replace(" ", "_")}_model.joblib')
        if not os.path.exists(model_path):
            st.error(f"Model file not found: {model_path}. Train models first.")
            logging.warning(f"Model file not found: {model_path}")
            return None
        model = joblib.load(model_path)
        logging.info(f"Model '{model_name}' loaded from {model_path}")
        return model
    except Exception as e:
        st.error(f"Error loading model '{model_name}': {e}")
        logging.error(f"Error loading model '{model_name}': {e}", exc_info=True)
        return None

def reset_dependent_states():
    logging.info("Resetting data-dependent session states.")
    st.session_state[SKEY_TRAINED] = False
    st.session_state[SKEY_MODEL_RESULTS] = {}
    st.session_state[SKEY_DF_OUTLIER_REMOVED_DATA] = None
    st.session_state[SKEY_ENCODERS] = {}
    st.session_state[SKEY_FEATURE_NAMES] = None
    st.session_state[SKEY_X_TEST_SCALED] = None
    st.session_state[SKEY_Y_TEST] = None
    st.session_state[SKEY_TRAINED_MODELS_LIST] = []

def display_sidebar():
    st.sidebar.header("⚙️ Configuration & Data")
    uploaded_file = st.sidebar.file_uploader("Upload your heart data CSV", type=["csv"], key="data_uploader")

    data_source_changed = False

    if uploaded_file is not None:
        if st.session_state.get(SKEY_ACTIVE_DATA_SOURCE_NAME) != uploaded_file.name:
            df = load_data(uploaded_file)
            if df is not None:
                st.session_state[SKEY_DF_ORIGINAL] = df
                st.session_state[SKEY_ACTIVE_DATA_SOURCE_NAME] = uploaded_file.name
                st.sidebar.success(f"Loaded '{uploaded_file.name}'.")
                data_source_changed = True
            else:
                st.session_state[SKEY_DF_ORIGINAL] = None
                st.session_state[SKEY_ACTIVE_DATA_SOURCE_NAME] = f"failed_load:{uploaded_file.name}"
                data_source_changed = True
    elif st.session_state.get(SKEY_ACTIVE_DATA_SOURCE_NAME) is not None and \
         st.session_state.get(SKEY_ACTIVE_DATA_SOURCE_NAME) != DEFAULT_DATA_FILE and \
         not st.session_state.get(SKEY_ACTIVE_DATA_SOURCE_NAME, "").startswith("failed_load:"):
        logging.info(f"File uploader cleared. Was: {st.session_state[SKEY_ACTIVE_DATA_SOURCE_NAME]}. Reverting to default.")
        df = load_data(DEFAULT_DATA_FILE)
        st.session_state[SKEY_DF_ORIGINAL] = df
        if df is not None:
            st.session_state[SKEY_ACTIVE_DATA_SOURCE_NAME] = DEFAULT_DATA_FILE
            st.sidebar.info(f"Uploader cleared. Loaded default: '{DEFAULT_DATA_FILE}'.")
        else:
            st.session_state[SKEY_ACTIVE_DATA_SOURCE_NAME] = None
        data_source_changed = True

    if SKEY_ACTIVE_DATA_SOURCE_NAME not in st.session_state:
        logging.info("Initial run: No active data source. Attempting to load default.")
        df = load_data(DEFAULT_DATA_FILE)
        st.session_state[SKEY_DF_ORIGINAL] = df
        if df is not None:
            st.session_state[SKEY_ACTIVE_DATA_SOURCE_NAME] = DEFAULT_DATA_FILE
        else:
            st.session_state[SKEY_ACTIVE_DATA_SOURCE_NAME] = None
        data_source_changed = True

    if data_source_changed:
        reset_dependent_states()
        st.rerun()

    test_size = st.sidebar.slider("Test Set Size", 0.1, 0.5, 0.2, 0.05, key="test_size_slider")

    options_for_prediction = st.session_state.get(SKEY_TRAINED_MODELS_LIST, [])
    if not options_for_prediction:
        try:
            model_files = [f for f in os.listdir(MODEL_ARTIFACTS_DIR) if f.endswith("_model.joblib")]
            if model_files:
                options_for_prediction = sorted([mf.replace("_model.joblib", "").replace("_", " ") for mf in model_files])
        except Exception as e:
            logging.warning(f"Could not list model files from disk: {e}")
    if not options_for_prediction:
        options_for_prediction = ["No models available/trained"]

    selected_model_for_prediction = st.sidebar.selectbox(
        "Select Model for Prediction",
        options=options_for_prediction,
        index=0,
        key="prediction_model_selector",
        disabled=(options_for_prediction == ["No models available/trained"])
    )
    return test_size, selected_model_for_prediction

def display_data_exploration(df_original, df_after_outlier_removal_data_tuple):
    st.header("📊 Data Exploration")
    if df_original is None:
        st.warning("Data not loaded. Cannot display exploration. Check sidebar for data source status.")
        return

    df_after_outlier_removal, removed_count, original_count = (None, 0, 0)
    if df_after_outlier_removal_data_tuple:
         df_after_outlier_removal, removed_count, original_count = df_after_outlier_removal_data_tuple

    def prep_df_for_st_display(df_to_display):
        if df_to_display is None: return None
        df_display_copy = df_to_display.copy()
        for col in df_display_copy.columns:
            if df_display_copy[col].dtype == 'object':
                try: df_display_copy[col] = df_display_copy[col].astype(str)
                except Exception as e: st.warning(f"Could not convert column {col} to string for display: {e}")
        return df_display_copy

    tab1, tab2, tab3 = st.tabs(["Overview & Stats", "Distributions", "Correlations"])

    with tab1:
        st.subheader("Original Dataset Sample")
        st.dataframe(prep_df_for_st_display(df_original.head()))
        st.subheader("Original Dataset Statistics")
        st.dataframe(prep_df_for_st_display(df_original.describe(include='all')))

        if df_after_outlier_removal is not None:
            st.subheader("Dataset Sample (After Outlier Removal)")
            st.dataframe(prep_df_for_st_display(df_after_outlier_removal.head()))
            st.subheader("Dataset Statistics (After Outlier Removal)")
            st.dataframe(prep_df_for_st_display(df_after_outlier_removal.describe(include='all')))
            st.info(f"Original shape: {df_original.shape}, After outlier removal: {df_after_outlier_removal.shape}")
            if original_count > 0:
                removed_percent = (removed_count / original_count) * 100
                st.metric(label="Rows Removed by IQR Outlier Treatment", value=removed_count, delta=f"{removed_percent:.2f}% of original")
            else:
                st.info("No outlier removal performed or original data was empty.")
        else:
            st.info("Outlier removal not yet performed or resulted in an empty dataset.")

    with tab2: 
        st.subheader("Feature Distributions")
        data_for_viz = df_after_outlier_removal if df_after_outlier_removal is not None and not df_after_outlier_removal.empty else df_original

        if data_for_viz is None or data_for_viz.empty:
             st.warning("Data not available for distributions.")
             return

        if TARGET_COL in data_for_viz.columns:
            try:
                fig = px.histogram(data_for_viz, x=TARGET_COL, color=TARGET_COL,
                                   title=f"{TARGET_COL} Distribution",
                                   color_discrete_map={0: "#BDB76B", 1: "#CD5C5C"})
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not plot target distribution: {e}")
        else: st.warning(f"Target column '{TARGET_COL}' not found for distribution plot.")

        st.write("**Numerical Features Distribution (Boxplots)**")
        numerical_cols_in_df = [col for col in NUMERICAL_COLS_FOR_OUTLIERS if col in data_for_viz.columns]
        if numerical_cols_in_df:
            try:
                fig_num = px.box(data_for_viz, y=numerical_cols_in_df, title='Numerical Features Boxplots')
                st.plotly_chart(fig_num, use_container_width=True)
            except Exception as e:
                 st.warning(f"Could not plot numerical boxplots: {e}")
        else: st.write("No specified numerical columns found in data for boxplots.")

        st.write("**Categorical Features Distribution (Countplots)**")
        cat_cols_for_viz = [col for col in CATEGORICAL_COLS if col in data_for_viz.columns]
        if cat_cols_for_viz:
            for col_name_viz in cat_cols_for_viz: 
                try:
                    hue_col = TARGET_COL if TARGET_COL in data_for_viz.columns else None
                    fig_cat = px.histogram(data_for_viz, x=col_name_viz, color=hue_col,
                                           title=f'Distribution of {col_name_viz}' + (f' by {TARGET_COL}' if hue_col else ''),
                                           barmode='group' if hue_col else 'overlay'

                                           )
                    fig_cat.update_xaxes(categoryorder='total descending')
                    st.plotly_chart(fig_cat, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not plot distribution for {col_name_viz}: {e}")
        else: st.write("No specified categorical columns found for countplots.")

    with tab3: 
        st.subheader("Correlation Heatmap")
        data_for_corr_source = df_after_outlier_removal if df_after_outlier_removal is not None and not df_after_outlier_removal.empty else df_original

        if data_for_corr_source is not None and not data_for_corr_source.empty:
            df_corr_processed, _ = preprocess_data(data_for_corr_source.copy(), CATEGORICAL_COLS, TARGET_COL)

            if df_corr_processed is not None:
                non_numeric_cols = df_corr_processed.select_dtypes(exclude=np.number).columns
                if len(non_numeric_cols) > 0:
                    st.warning(f"Non-numeric columns still present after attempting encoding for heatmap: {list(non_numeric_cols)}. Dropping them.")
                    df_corr_processed = df_corr_processed.drop(columns=non_numeric_cols)

                if not df_corr_processed.empty and df_corr_processed.shape[1] > 1:
                    corr_matrix = df_corr_processed.corr()
                    fig_corr = px.imshow(corr_matrix, text_auto=".2f", aspect="auto",
                                         color_continuous_scale=px.colors.diverging.RdBu,
                                         title="Correlation Matrix (Processed & Encoded Data)")
                    st.plotly_chart(fig_corr, use_container_width=True)
                else: st.warning("Not enough numeric data to compute correlation matrix after processing.")
            else: st.warning("Preprocessing for correlation heatmap failed.")
        else: st.warning("Data not available for correlation heatmap.")

def display_model_training(df_original, test_size):
    st.header("🏋️ Model Training")
    if df_original is None:
        st.warning("Data not loaded. Cannot train models. Check sidebar.")
        return

    if st.button("🚀 Train All Models", key="train_button"):
        with st.spinner("Preprocessing data and training models..."):
            logging.info("Starting model training process.")
            df_after_outliers, rem_count, orig_count = remove_outliers_IQR(df_original.copy(), NUMERICAL_COLS_FOR_OUTLIERS)
            st.session_state[SKEY_DF_OUTLIER_REMOVED_DATA] = (df_after_outliers, rem_count, orig_count)

            if df_after_outliers is None or df_after_outliers.empty:
                st.error("Dataset is empty after outlier removal. Cannot train.")
                logging.error("Dataset empty after outlier removal for training.")
                return

            df_processed, encoders = preprocess_data(df_after_outliers.copy(), CATEGORICAL_COLS, TARGET_COL)
            st.session_state[SKEY_ENCODERS] = encoders

            if df_processed is None or TARGET_COL not in df_processed.columns:
                st.error("Preprocessing failed or target column missing. Cannot train.")
                return

            X = df_processed.drop(TARGET_COL, axis=1)
            y = df_processed[TARGET_COL]
            feature_names = X.columns.tolist()
            st.session_state[SKEY_FEATURE_NAMES] = feature_names

            if X.empty or len(y.unique()) < 2:
                st.error("Feature set empty or target has < 2 classes. Cannot train.")
                return

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            save_training_artifacts(scaler, feature_names)
            st.session_state[SKEY_X_TEST_SCALED] = X_test_scaled
            st.session_state[SKEY_Y_TEST] = y_test

            models = get_models()
            results = {}
            trained_models_list = []
            progress_bar = st.progress(0)

            for i, (name, model) in enumerate(models.items()):
                logging.info(f"Training model: {name}")
                try:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    accuracy = accuracy_score(y_test, y_pred)
                    results[name] = {"model": model, "accuracy": accuracy, "y_pred": y_pred}
                    joblib.dump(model, os.path.join(MODEL_ARTIFACTS_DIR, f'{name.replace(" ", "_")}_model.joblib'))
                    trained_models_list.append(name)
                    logging.info(f"Trained {name}, Acc: {accuracy:.4f}")
                except Exception as e:
                    st.error(f"Error training {name}: {e}")
                    logging.error(f"Error training {name}: {e}", exc_info=True)
                    results[name] = {"accuracy": "Error", "model": None, "y_pred": None}
                progress_bar.progress((i + 1) / len(models))

            st.session_state[SKEY_MODEL_RESULTS] = results
            st.session_state[SKEY_TRAINED_MODELS_LIST] = trained_models_list
            st.session_state[SKEY_TRAINED] = True
            logging.info("All models trained.")
            st.success("✅ All models trained and artifacts saved!")
            st.balloons()
            st.rerun()

def display_model_performance():
    st.header("📈 Model Performance")
    if not st.session_state.get(SKEY_TRAINED, False) or SKEY_MODEL_RESULTS not in st.session_state:
        st.info("Models not trained yet. Go to 'Model Training' tab.")
        return

    results = st.session_state[SKEY_MODEL_RESULTS]
    y_test = st.session_state.get(SKEY_Y_TEST)
    X_test_scaled = st.session_state.get(SKEY_X_TEST_SCALED)

    if y_test is None or X_test_scaled is None:
        st.warning("Test data not found. Cannot display performance.")
        return

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("🎯 Accuracy Scores")
        accuracy_data = {name: data["accuracy"] for name, data in results.items() if data["accuracy"] != "Error"}
        if not accuracy_data:
            st.warning("No models successfully trained.")
            return
        results_df = pd.DataFrame.from_dict(accuracy_data, orient='index', columns=['Accuracy']).sort_values(by='Accuracy', ascending=False)
        st.dataframe(results_df.style.format({'Accuracy': '{:.2%}'}))

        st.subheader("🔍 Detailed Metrics")
        valid_model_names = [name for name, data in results.items() if data["accuracy"] != "Error"]
        if not valid_model_names:
            st.write("No models for detailed view.")
            return
        selected_eval_model_name = st.selectbox("Select Model for Details", options=valid_model_names, key="eval_model_selector")

    with col2:
        if selected_eval_model_name and selected_eval_model_name in results:
            model_data = results[selected_eval_model_name]
            current_model = model_data["model"]
            if model_data["accuracy"] == "Error" or model_data["y_pred"] is None:
                st.error(f"Cannot display metrics for {selected_eval_model_name}.")
                return
            y_pred = model_data["y_pred"]

            st.write(f"**Confusion Matrix for {selected_eval_model_name}**")
            cm = confusion_matrix(y_test, y_pred)
            labels_cm = sorted(pd.Series(y_test).unique())
            fig_cm = px.imshow(cm, text_auto=True, labels=dict(x="Predicted", y="Actual"),
                               x=[str(l) for l in labels_cm], y=[str(l) for l in labels_cm], title="Confusion Matrix")
            st.plotly_chart(fig_cm)

            st.write(f"**Classification Report for {selected_eval_model_name}**")
            try:
                report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.style.format({'precision': '{:.2f}', 'recall': '{:.2f}', 'f1-score': '{:.2f}', 'support': '{:.0f}'}))
            except ValueError as e: st.error(f"Error generating classification report: {e}")

            if hasattr(current_model, "predict_proba"):
                y_probs = current_model.predict_proba(X_test_scaled)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_probs)
                roc_auc = auc(fpr, tpr)
                fig_roc = px.area(x=fpr, y=tpr, title=f'ROC Curve (AUC={roc_auc:.2f}) - {selected_eval_model_name}',
                                  labels=dict(x='False Positive Rate', y='True Positive Rate'))
                fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
                st.plotly_chart(fig_roc, use_container_width=True)
            else: st.info(f"{selected_eval_model_name} doesn't support probability for ROC.")

            feature_names_perf = st.session_state.get(SKEY_FEATURE_NAMES, [])
            if feature_names_perf:
                importances_data = None
                plot_title = ""
                if hasattr(current_model, "feature_importances_"):
                    importances = current_model.feature_importances_
                    importances_data = pd.DataFrame({'Feature': feature_names_perf, 'Importance': importances})
                    plot_title = f"Feature Importances for {selected_eval_model_name}"
                elif hasattr(current_model, "coef_"):
                    coefs = current_model.coef_[0] if current_model.coef_.ndim > 1 else current_model.coef_
                    importances_data = pd.DataFrame({'Feature': feature_names_perf, 'Coefficient': coefs})
                    importances_data['Importance'] = np.abs(importances_data['Coefficient'])
                    plot_title = f"Feature Coefficients (Abs) for {selected_eval_model_name}"

                if importances_data is not None:
                    importances_data = importances_data.sort_values(by='Importance', ascending=False).head(15)
                    fig_imp = px.bar(importances_data, x='Importance', y='Feature', orientation='h', title=plot_title)
                    fig_imp.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig_imp, use_container_width=True)
                else: st.info(f"Feature importance/coeffs not available for {selected_eval_model_name}.")
            else: st.warning("Feature names not found for importance plot.")

def display_prediction_interface(selected_model_name_for_prediction, df_for_form_options):
    st.header("🔮 Heart Disease Prediction")
    essential_artifacts_exist = (os.path.exists(os.path.join(MODEL_ARTIFACTS_DIR, 'scaler.joblib')) and
                                 os.path.exists(os.path.join(MODEL_ARTIFACTS_DIR, 'feature_names.joblib')))
    if not essential_artifacts_exist:
        st.warning("Core model artifacts missing. Train models first.")
        return

    if selected_model_name_for_prediction == "No models available/trained" or not selected_model_name_for_prediction:
        st.info("Select a trained model to make predictions.")
        return

    model_file_path = os.path.join(MODEL_ARTIFACTS_DIR, f'{selected_model_name_for_prediction.replace(" ", "_")}_model.joblib')
    if not os.path.exists(model_file_path) :
        st.warning(f"Model file for '{selected_model_name_for_prediction}' not found. (Re)train models.")
        return

    encoders_pred, scaler_pred, feature_names_pred = load_prediction_artifacts()
    if not all([scaler_pred, feature_names_pred]): return

    model_pred = load_model(selected_model_name_for_prediction)
    if model_pred is None: return

    st.info(f"Using **{selected_model_name_for_prediction}** for prediction.")

    with st.form("prediction_form"):
        st.subheader("Enter Patient Details:")
        form_cols = st.columns(3)
        input_data_raw = {}

        if df_for_form_options is None or df_for_form_options.empty:
            st.error("Data for form options not available. Load data.")
            return

        def get_num_input_params(col_name, default_val, step_val=1, data_type=int):
            min_v, max_v, val_v = default_val, default_val + 100, default_val
            if col_name in df_for_form_options and pd.api.types.is_numeric_dtype(df_for_form_options[col_name]) and not df_for_form_options[col_name].isnull().all():
                min_v_data = df_for_form_options[col_name].min()
                max_v_data = df_for_form_options[col_name].max()
                min_v = data_type(min_v_data)
                max_v = data_type(max_v_data)
                val_v = default_val if min_v <= default_val <= max_v else min_v
            return min_v, max_v, val_v, data_type(step_val) if isinstance(step_val, (int, float)) else step_val

        with form_cols[0]:
            age_min, age_max, age_val, age_step = get_num_input_params('Age', 50)
            input_data_raw['Age'] = st.number_input("Age", min_value=age_min, max_value=age_max, value=age_val, step=age_step)
            sex_opts = sorted(list(df_for_form_options['Sex'].unique())) if 'Sex' in df_for_form_options and not df_for_form_options['Sex'].empty and 'Sex' in df_for_form_options.columns else ['M', 'F']
            input_data_raw['Sex'] = st.selectbox("Sex", options=sex_opts, index=sex_opts.index('M') if 'M' in sex_opts else 0)
            cpt_opts = sorted(list(df_for_form_options['ChestPainType'].unique())) if 'ChestPainType' in df_for_form_options and not df_for_form_options['ChestPainType'].empty and 'ChestPainType' in df_for_form_options.columns else ['ATA', 'NAP', 'ASY', 'TA']
            input_data_raw['ChestPainType'] = st.selectbox("Chest Pain Type", options=cpt_opts, index=cpt_opts.index('ASY') if 'ASY' in cpt_opts else 0)
            rbp_min, rbp_max, rbp_val, rbp_step = get_num_input_params('RestingBP', 120)
            input_data_raw['RestingBP'] = st.number_input("Resting BP", min_value=rbp_min, max_value=rbp_max, value=rbp_val, step=rbp_step)

        with form_cols[1]:
            chol_min, chol_max, chol_val, chol_step = get_num_input_params('Cholesterol', 200)
            input_data_raw['Cholesterol'] = st.number_input("Cholesterol", min_value=chol_min, max_value=chol_max, value=chol_val, step=chol_step)
            input_data_raw['FastingBS'] = st.selectbox("Fasting BS > 120 mg/dl", options=[0, 1], index=0) 
            recg_opts = sorted(list(df_for_form_options['RestingECG'].unique())) if 'RestingECG' in df_for_form_options and not df_for_form_options['RestingECG'].empty and 'RestingECG' in df_for_form_options.columns else ['Normal', 'ST', 'LVH']
            input_data_raw['RestingECG'] = st.selectbox("Resting ECG", options=recg_opts, index=recg_opts.index('Normal') if 'Normal' in recg_opts else 0)
            mhr_min, mhr_max, mhr_val, mhr_step = get_num_input_params('MaxHR', 150)
            input_data_raw['MaxHR'] = st.number_input("Max HR", min_value=mhr_min, max_value=mhr_max, value=mhr_val, step=mhr_step)

        with form_cols[2]:
            ea_opts = sorted(list(df_for_form_options['ExerciseAngina'].unique())) if 'ExerciseAngina' in df_for_form_options and not df_for_form_options['ExerciseAngina'].empty and 'ExerciseAngina' in df_for_form_options.columns else ['N', 'Y']
            input_data_raw['ExerciseAngina'] = st.selectbox("Exercise Angina", options=ea_opts, index=ea_opts.index('N') if 'N' in ea_opts else 0)
            op_min, op_max, op_val, op_step = get_num_input_params('Oldpeak', 1.0, 0.1, float)
            input_data_raw['Oldpeak'] = st.number_input("Oldpeak", min_value=op_min, max_value=op_max, value=op_val, step=op_step, format="%.1f")
            sts_opts = sorted(list(df_for_form_options['ST_Slope'].unique())) if 'ST_Slope' in df_for_form_options and not df_for_form_options['ST_Slope'].empty and 'ST_Slope' in df_for_form_options.columns else ['Up', 'Flat', 'Down']
            input_data_raw['ST_Slope'] = st.selectbox("ST Slope", options=sts_opts, index=sts_opts.index('Flat') if 'Flat' in sts_opts else 0)

        submitted = st.form_submit_button("🩺 Predict Disease Status")

    if submitted:
        input_df_raw = pd.DataFrame([input_data_raw])
        input_df_processed = input_df_raw.copy()

        if encoders_pred:
            for col, encoder_obj in encoders_pred.items():
                if col in input_df_processed.columns:
                    try:
                        input_val = [input_df_processed[col].iloc[0]]
                        input_df_processed[col] = encoder_obj.transform(input_val)
                    except ValueError as e:
                        st.error(f"Error encoding '{col}': Value '{input_val[0]}' not seen during training. Known: {list(encoder_obj.classes_)}. {e}")
                        return
        try:
            input_df_ordered = input_df_processed[feature_names_pred]
            scaled_input = scaler_pred.transform(input_df_ordered)
            prediction = model_pred.predict(scaled_input)
            probability = model_pred.predict_proba(scaled_input)[0][1] if hasattr(model_pred, "predict_proba") else 0.0

            st.subheader("🩺 Prediction Result")
            if prediction[0] == 1:
                st.error("🚨 Heart Disease Detected")
                st.metric("Risk Score (Prob. of Disease)", f"{probability:.2%}", "High Risk", delta_color="inverse")
            else:
                st.success("✅ No Heart Disease Detected (Low Risk)")
                st.metric("Risk Score (Prob. of Disease)", f"{probability:.2%}", "Low Risk", delta_color="normal")

            with st.expander("Show Input Data"):
                st.json(input_data_raw)
                st.write("Processed input (first 5 features shown if available):", scaled_input[0, :min(5, scaled_input.shape[1])])

        except KeyError as e:
            st.error(f"Feature mismatch during prediction: {e}. Expected features based on training: {feature_names_pred}")
        except Exception as e:
            st.error(f"Error during prediction steps: {e}")
            logging.error(f"Prediction step error: {e}", exc_info=True)

def main():
    st.set_page_config(**PAGE_CONFIG)

    keys_to_initialize = [
        (SKEY_DF_ORIGINAL, None), (SKEY_ACTIVE_DATA_SOURCE_NAME, None),
        (SKEY_TRAINED, False), (SKEY_MODEL_RESULTS, {}),
        (SKEY_DF_OUTLIER_REMOVED_DATA, None), (SKEY_ENCODERS, {}),
        (SKEY_FEATURE_NAMES, None), (SKEY_X_TEST_SCALED, None),
        (SKEY_Y_TEST, None), (SKEY_TRAINED_MODELS_LIST, [])
    ]
    for key, default_value in keys_to_initialize:
        if key not in st.session_state:
            st.session_state[key] = default_value

    st.title("❤️ Advanced Heart Disease Prediction App")
    st.markdown("""
    Welcome! This app predicts heart disease likelihood.
    Use the sidebar to upload data. Tabs for exploration, training, performance, and prediction.
    """)

    test_size, selected_model_for_prediction = display_sidebar()

    df_original_main = st.session_state.get(SKEY_DF_ORIGINAL)

    if df_original_main is not None and st.session_state.get(SKEY_DF_OUTLIER_REMOVED_DATA) is None:
        df_temp, removed_count, original_count = remove_outliers_IQR(df_original_main.copy(), NUMERICAL_COLS_FOR_OUTLIERS)
        st.session_state[SKEY_DF_OUTLIER_REMOVED_DATA] = (df_temp, removed_count, original_count)

    df_for_eda_tuple = st.session_state.get(SKEY_DF_OUTLIER_REMOVED_DATA)

    tab_eda, tab_training, tab_performance, tab_predict = st.tabs(
        ["📊 Data Exploration", "🏋️ Model Training", "📈 Model Performance", "🔮 Prediction"]
    )

    with tab_eda:
        display_data_exploration(df_original_main, df_for_eda_tuple)
    with tab_training:
        display_model_training(df_original_main, test_size)
    with tab_performance:
        display_model_performance()
    with tab_predict:
        display_prediction_interface(selected_model_for_prediction, df_original_main)

    st.sidebar.markdown("---")
    st.sidebar.info("Magdy Mohamed | 20211206\nMohamed Osama | 352151205\nAmr Abdalzeez | 352151201\nSharief Atef | 352151134")

if __name__ == "__main__":
    main()
